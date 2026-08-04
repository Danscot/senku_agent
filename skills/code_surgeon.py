"""
skills/code_surgeon.py — Surgical code editing at the function/class/block level.

Instead of rewriting entire files, code_surgeon:
  1. Reads the target file
  2. Locates the specific symbol (function, class, method, or line range)
  3. Sends ONLY that snippet to the LLM with the instruction
  4. Splices the new snippet back into the original file

This means the LLM never sees (or can corrupt) the rest of the file.

Operations:
  edit_function   — rewrite a named function or method
  edit_class      — rewrite a named class definition (header + body)
  edit_method     — rewrite a method inside a named class
  insert_after    — insert new code after a named symbol
  insert_before   — insert new code before a named symbol
  delete_symbol   — delete a named function/class/method entirely
  edit_lines      — edit an explicit line range (start_line, end_line)
  add_import      — add an import statement at the top of the file
  edit_html_tag   — find+rewrite an HTML element by id or unique selector
  edit_css_rule   — find+rewrite a CSS rule block by selector

Supported languages: Python, JavaScript, TypeScript, HTML, CSS
"""

from __future__ import annotations

import re
import ast
import textwrap
from pathlib import Path
from typing import NamedTuple

from config import MODEL, PROJECT_ROOT, TOKENS_CODE_WRITER, TEMPERATURE_SKILLS


# ══════════════════════════════════════════════════════════════════════════════
#  Data types
# ══════════════════════════════════════════════════════════════════════════════

class Snippet(NamedTuple):
    start:   int    # 0-based line index (inclusive)
    end:     int    # 0-based line index (exclusive)
    text:    str    # the extracted snippet text
    indent:  str    # leading indentation of the first line


# ══════════════════════════════════════════════════════════════════════════════
#  Python symbol locator  (AST-based, precise)
# ══════════════════════════════════════════════════════════════════════════════

def _py_find_symbol(
    source: str,
    symbol: str,
    class_name: str = "",
) -> Snippet | None:
    """
    Use Python's AST to find a function or class by name.
    If class_name is given, look for `symbol` as a method inside that class.
    Returns a Snippet with 0-based line indices.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines(keepends=True)

    def node_snippet(node: ast.AST) -> Snippet:
        start = node.lineno - 1          # ast is 1-based
        end   = node.end_lineno          # end_lineno is inclusive → exclusive slice
        text  = "".join(lines[start:end])
        indent = len(lines[start]) - len(lines[start].lstrip())
        return Snippet(start, end, text, " " * indent)

    if class_name:
        # Find the class first, then the method inside it
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == symbol:
                        return node_snippet(item)
        return None

    # Top-level symbol
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return node_snippet(node)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  JS/TS symbol locator  (regex-based, good-enough)
# ══════════════════════════════════════════════════════════════════════════════

def _js_find_function(source: str, symbol: str) -> Snippet | None:
    """
    Find a JS/TS function by name using brace-counting.
    Handles: function foo(), const foo = () =>, const foo = function(), class Foo
    """
    lines  = source.splitlines(keepends=True)
    # Patterns that start a named function/class/arrow
    patterns = [
        rf"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+{re.escape(symbol)}\s*[\(\{{]",
        rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+{re.escape(symbol)}\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_$]\w*)\s*=>",
        rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+{re.escape(symbol)}\s*=\s*(?:async\s+)?function",
        rf"^(\s*)(?:export\s+)?(?:default\s+)?class\s+{re.escape(symbol)}(?:\s+extends\s+\w+)?\s*\{{",
    ]

    for i, line in enumerate(lines):
        for pat in patterns:
            if re.match(pat, line):
                indent_len = len(line) - len(line.lstrip())
                indent     = " " * indent_len
                # Count braces to find the end
                depth  = 0
                j      = i
                started = False
                for j in range(i, len(lines)):
                    for ch in lines[j]:
                        if ch == "{":
                            depth += 1
                            started = True
                        elif ch == "}":
                            depth -= 1
                    if started and depth == 0:
                        break
                text = "".join(lines[i:j+1])
                return Snippet(i, j + 1, text, indent)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  HTML element locator
# ══════════════════════════════════════════════════════════════════════════════

def _html_find_element(source: str, selector: str) -> Snippet | None:
    """
    Find an HTML element by id (e.g. '#hero') or tag+class (e.g. 'div.card').
    Returns the outermost matching tag as a Snippet.
    """
    lines = source.splitlines(keepends=True)

    if selector.startswith("#"):
        attr_id = selector[1:]
        start_pat = re.compile(rf'<(\w+)[^>]*\bid=["\']?{re.escape(attr_id)}["\']?[^>]*>', re.IGNORECASE)
    elif "." in selector:
        tag, cls = selector.split(".", 1)
        start_pat = re.compile(rf'<({re.escape(tag)})[^>]*\bclass=["\'][^"\']*\b{re.escape(cls)}\b[^"\']*["\'][^>]*>', re.IGNORECASE)
    else:
        start_pat = re.compile(rf'<({re.escape(selector)})[^>]*>', re.IGNORECASE)

    for i, line in enumerate(lines):
        m = start_pat.search(line)
        if not m:
            continue
        tag_name = m.group(1)
        # Walk forward counting open/close tags
        depth = 0
        for j in range(i, len(lines)):
            opens  = len(re.findall(rf'<{tag_name}[\s>]', lines[j], re.IGNORECASE))
            closes = len(re.findall(rf'</{tag_name}>', lines[j], re.IGNORECASE))
            depth += opens - closes
            if depth <= 0:
                text   = "".join(lines[i:j+1])
                indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
                return Snippet(i, j + 1, text, indent)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CSS rule locator
# ══════════════════════════════════════════════════════════════════════════════

def _css_find_rule(source: str, selector: str) -> Snippet | None:
    """Find a CSS rule block by selector string."""
    lines = source.splitlines(keepends=True)
    sel_pat = re.compile(r'^\s*' + re.escape(selector) + r'\s*\{', re.IGNORECASE)

    for i, line in enumerate(lines):
        if sel_pat.match(line):
            depth = 0
            for j in range(i, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0:
                    text   = "".join(lines[i:j+1])
                    indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
                    return Snippet(i, j + 1, text, indent)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Snippet locator dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def _find_snippet(
    op:         str,
    source:     str,
    lang:       str,
    symbol:     str,
    class_name: str,
    selector:   str,
    start_line: int,
    end_line:   int,
) -> Snippet | None:
    """Route to the right locator based on op and language."""

    if op == "edit_lines":
        lines  = source.splitlines(keepends=True)
        s      = max(0, start_line - 1)
        e      = min(len(lines), end_line)
        text   = "".join(lines[s:e])
        indent = " " * (len(lines[s]) - len(lines[s].lstrip())) if lines[s:] else ""
        return Snippet(s, e, text, indent)

    if op == "edit_html_tag":
        return _html_find_element(source, selector or symbol)

    if op == "edit_css_rule":
        return _css_find_rule(source, selector or symbol)

    if lang == "python":
        if op in ("edit_function", "insert_after", "insert_before", "delete_symbol"):
            return _py_find_symbol(source, symbol)
        if op in ("edit_method",):
            return _py_find_symbol(source, symbol, class_name)
        if op == "edit_class":
            return _py_find_symbol(source, symbol)  # symbol = class name

    if lang in ("javascript", "typescript", "jsx", "tsx"):
        return _js_find_function(source, symbol)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Splice helpers
# ══════════════════════════════════════════════════════════════════════════════

def _splice(source: str, snippet: Snippet, replacement: str) -> str:
    """Replace snippet.start..snippet.end lines with replacement text."""
    lines = source.splitlines(keepends=True)
    # Preserve trailing newline on replacement
    if replacement and not replacement.endswith("\n"):
        replacement += "\n"
    return "".join(lines[:snippet.start]) + replacement + "".join(lines[snippet.end:])


def _insert_after(source: str, snippet: Snippet, new_code: str) -> str:
    lines = source.splitlines(keepends=True)
    if new_code and not new_code.endswith("\n"):
        new_code += "\n"
    return "".join(lines[:snippet.end]) + "\n" + new_code + "".join(lines[snippet.end:])


def _insert_before(source: str, snippet: Snippet, new_code: str) -> str:
    lines = source.splitlines(keepends=True)
    if new_code and not new_code.endswith("\n"):
        new_code += "\n"
    return "".join(lines[:snippet.start]) + new_code + "\n" + "".join(lines[snippet.start:])


def _add_import(source: str, import_stmt: str) -> str:
    """Insert an import at the end of the existing import block."""
    lines  = source.splitlines(keepends=True)
    last_i = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) or stripped == "":
            if stripped:
                last_i = i
    stmt = import_stmt.strip() + "\n"
    return "".join(lines[:last_i+1]) + stmt + "".join(lines[last_i+1:])


# ══════════════════════════════════════════════════════════════════════════════
#  LLM call — edit only the snippet
# ══════════════════════════════════════════════════════════════════════════════

def _llm_edit_snippet(
    client,
    snippet:     Snippet,
    instruction: str,
    lang:        str,
    file_path:   str,
    context:     str,
    params:      dict,
) -> str:
    """
    Ask the LLM to rewrite ONLY the given snippet.
    Returns the raw replacement text (no fences).
    """
    system = f"""\
You are Senku, a surgical code editor with 10 billion percent precision.
You will be given a SINGLE code snippet extracted from a larger file.
Your ONLY job is to apply the instruction to that snippet and return the modified version.

Rules:
  - Return ONLY the modified snippet — nothing else, no explanations, no fences
  - Preserve the original indentation exactly (base indent: {repr(snippet.indent)})
  - Do NOT rewrite any code outside the snippet
  - Do NOT add imports or global changes (use add_import op for that)
  - If the instruction is unclear, do the most conservative correct thing

File: {file_path}
Language: {lang or 'unknown'}
"""

    user_prompt = f"""\
Snippet to modify (lines {snippet.start + 1}–{snippet.end}):
```{lang}
{snippet.text}
```

Instruction: {instruction}
{f'Additional context: {context}' if context else ''}

Return ONLY the modified snippet with the same base indentation. No fences, no prose.
"""

    from skills._llm import skill_llm_call
    raw = skill_llm_call(
        client, MODEL,
        [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=TOKENS_CODE_WRITER,
        provider=params.get("provider", ""),
    )

    # Strip accidental fences the LLM adds anyway
    raw = re.sub(r"^```[a-z]*\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$",        "", raw)
    return raw.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(params: dict, client) -> str:
    op          = params.get("op", "")
    path        = params.get("path", "")
    symbol      = params.get("symbol", "")        # function / class / method name
    class_name  = params.get("class_name", "")    # for edit_method
    selector    = params.get("selector", "")      # for HTML/CSS
    instruction = params.get("instruction", "")
    context     = params.get("context", "")
    start_line  = int(params.get("start_line", 0))
    end_line    = int(params.get("end_line", 0))
    root        = params.get("root", params.get("project_root", PROJECT_ROOT))
    import_stmt = params.get("import_stmt", "")   # for add_import op

    VALID_OPS = {
        "edit_function", "edit_class", "edit_method",
        "insert_after", "insert_before", "delete_symbol",
        "edit_lines", "add_import",
        "edit_html_tag", "edit_css_rule",
        "replace_text",   # literal find-and-replace, no LLM needed
    }

    if not op:
        return f"ERROR: 'op' is required. Choose: {', '.join(sorted(VALID_OPS))}"
    if op not in VALID_OPS:
        return f"ERROR: unknown op '{op}'. Choose: {', '.join(sorted(VALID_OPS))}"
    if not path:
        return "ERROR: 'path' is required."

    resolved = path if Path(path).is_absolute() else str(Path(root) / path)

    try:
        source = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file '{resolved}' not found."
    except Exception as e:
        return f"ERROR reading '{resolved}': {e}"

    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html",
        ".css": "css", ".scss": "css",
    }
    lang = ext_map.get(Path(resolved).suffix.lower(), "")

    # ── replace_text — pure string substitution, zero LLM ─────────────────
    if op == "replace_text":
        old_text = params.get("old_text", "")
        new_text = params.get("new_text", "")
        if not old_text:
            return "ERROR: 'old_text' is required for replace_text op."
        count = source.count(old_text)
        if count == 0:
            return f"ERROR: could not find {repr(old_text)} in '{resolved}'."
        new_source = source.replace(old_text, new_text)
        Path(resolved).write_text(new_source, encoding="utf-8")
        return (
            f"✔ Replaced {count} occurrence(s) of {repr(old_text)} "
            f"→ {repr(new_text)} in '{resolved}'"
        )

    # ── add_import — no LLM needed ─────────────────────────────────────────
    if op == "add_import":
        if not import_stmt:
            return "ERROR: 'import_stmt' is required for add_import op."
        new_source = _add_import(source, import_stmt)
        Path(resolved).write_text(new_source, encoding="utf-8")
        return f"✔ Added import to '{resolved}': {import_stmt}"

    # ── All other ops need a snippet ───────────────────────────────────────
    target = symbol or selector
    if not target and op != "edit_lines":
        return f"ERROR: 'symbol' (or 'selector' for HTML/CSS) is required for op '{op}'."
    if op == "edit_lines" and (not start_line or not end_line):
        return "ERROR: 'start_line' and 'end_line' are required for edit_lines op."
    if not instruction and op != "delete_symbol":
        return "ERROR: 'instruction' is required."

    snippet = _find_snippet(op, source, lang, symbol, class_name, selector, start_line, end_line)
    if snippet is None:
        loc = f"'{symbol}'" + (f" in class '{class_name}'" if class_name else "")
        return (
            f"ERROR: could not locate {loc} in '{resolved}'.\n"
            f"Make sure the name matches exactly (case-sensitive).\n"
            f"Tip: use op='edit_lines' with explicit line numbers as a fallback."
        )

    original_snippet = snippet.text

    # ── delete_symbol — no LLM needed ─────────────────────────────────────
    if op == "delete_symbol":
        new_source = _splice(source, snippet, "")
        Path(resolved).write_text(new_source, encoding="utf-8")
        lines_removed = snippet.end - snippet.start
        return (
            f"✔ Deleted '{symbol}' from '{resolved}' "
            f"(removed lines {snippet.start+1}–{snippet.end}, {lines_removed} lines)"
        )

    # ── LLM edits the snippet ─────────────────────────────────────────────
    new_snippet_text = _llm_edit_snippet(client, snippet, instruction, lang, resolved, context, params)

    if not new_snippet_text:
        return "ERROR: LLM returned empty snippet."

    # Restore base indentation if LLM stripped it
    if snippet.indent and not new_snippet_text.startswith(snippet.indent):
        new_snippet_text = textwrap.indent(
            textwrap.dedent(new_snippet_text),
            snippet.indent,
        )

    # ── Splice back into file ─────────────────────────────────────────────
    if op == "insert_after":
        new_source = _insert_after(source, snippet, new_snippet_text)
        action = f"Inserted after '{symbol}'"
    elif op == "insert_before":
        new_source = _insert_before(source, snippet, new_snippet_text)
        action = f"Inserted before '{symbol}'"
    else:
        new_source = _splice(source, snippet, new_snippet_text)
        action = f"Updated '{symbol or selector or f'lines {start_line}-{end_line}'}'"

    try:
        Path(resolved).write_text(new_source, encoding="utf-8")
    except Exception as e:
        return f"ERROR writing '{resolved}': {e}"

    # ── Summary ───────────────────────────────────────────────────────────
    old_lines = snippet.text.count("\n") + 1
    new_lines = new_snippet_text.count("\n") + 1
    delta     = new_lines - old_lines
    delta_str = f"+{delta}" if delta >= 0 else str(delta)

    return (
        f"✔ {action} in '{resolved}'\n"
        f"  Lines {snippet.start+1}–{snippet.end}  ({old_lines} → {new_lines} lines, {delta_str})\n\n"
        f"Before:\n```{lang}\n{original_snippet.strip()}\n```\n\n"
        f"After:\n```{lang}\n{new_snippet_text.strip()}\n```"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Skill registration
# ══════════════════════════════════════════════════════════════════════════════

SKILL_DEF = {
    "name": "code_surgeon",
    "description": (
        "Surgical code editing — modify a specific function, class, method, or line range "
        "WITHOUT rewriting the whole file. "
        "Use this instead of code_writer whenever editing existing code. "
        "Supports Python (AST-precise), JS/TS (regex), HTML (tag by id/class), CSS (rule by selector). "
        "Ops: edit_function | edit_class | edit_method | insert_after | insert_before | "
        "delete_symbol | edit_lines | add_import | edit_html_tag | edit_css_rule | "
        "replace_text (literal find-and-replace, no LLM — fastest for simple substitutions)"
    ),
    "params": [
        {"name": "op",          "type": "str", "required": True,
         "description": "edit_function | edit_class | edit_method | insert_after | insert_before | delete_symbol | edit_lines | add_import | edit_html_tag | edit_css_rule | replace_text"},
        {"name": "path",        "type": "str", "required": True,
         "description": "File path (absolute or relative to project root)"},
        {"name": "symbol",      "type": "str", "required": False,
         "description": "Exact name of the function/class/method to target"},
        {"name": "class_name",  "type": "str", "required": False,
         "description": "Class name when using edit_method"},
        {"name": "selector",    "type": "str", "required": False,
         "description": "CSS/HTML selector e.g. '#hero' or 'div.card' or '.navbar' for HTML/CSS ops"},
        {"name": "instruction", "type": "str", "required": False,
         "description": "What change to make to the snippet"},
        {"name": "old_text",    "type": "str", "required": False,
         "description": "Exact string to find — required for replace_text op"},
        {"name": "new_text",    "type": "str", "required": False,
         "description": "Replacement string — required for replace_text op"},
        {"name": "start_line",  "type": "int", "required": False,
         "description": "Start line number (1-based) for edit_lines op"},
        {"name": "end_line",    "type": "int", "required": False,
         "description": "End line number (1-based, inclusive) for edit_lines op"},
        {"name": "import_stmt", "type": "str", "required": False,
         "description": "Import statement to add e.g. 'from django.db import models'"},
        {"name": "context",     "type": "str", "required": False,
         "description": "Extra context about the change for the LLM"},
    ],
    "fn": run,
    "needs_client": True,
}
