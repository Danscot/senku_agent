"""
skills/code_writer.py
Write or edit a code file intelligently.

Before generating code, it reads structure.md to understand the dependency
graph — so it knows which other files will be affected by this change and
can produce coherent, project-aware code.

Modes:
  create  — write a brand new file
  edit    — modify an existing file (patch or full rewrite)
  patch   — apply a specific targeted change described in natural language
"""

from __future__ import annotations

import re
from pathlib import Path
from config import MODEL, PROJECT_ROOT, TOKENS_CODE_WRITER, TEMPERATURE_SKILLS, THINKING_ENABLED, THINKING_BUDGET


def _safe_extract(resp, default="") -> str:
    if not resp or not resp.choices:
        return default
    content = resp.choices[0].message.content
    if content is None:
        return default
    return content.strip() if isinstance(content, str) else str(content).strip()


def _load_structure_context(root: str) -> str:
    """Load structure.md if it exists, return a trimmed context block."""
    smd = Path(root) / "structure.md"
    if not smd.exists():
        return ""
    try:
        text = smd.read_text(encoding="utf-8")
        if len(text) > 4000:
            text = text[:4000] + "\n\n[... structure.md truncated]"
        return text
    except Exception:
        return ""


def _read_file(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
        if len(text) > 8000:
            text = text[:8000] + "\n\n# [... file truncated for context]"
        return text
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"# ERROR reading file: {e}"


def _infer_lang(path: str) -> str:
    ext_map = {
        ".py":   "python",  ".js":   "javascript", ".ts":  "typescript",
        ".jsx":  "jsx",     ".tsx":  "tsx",         ".html":"html",
        ".css":  "css",     ".scss": "scss",        ".sh":  "bash",
        ".yaml": "yaml",    ".yml":  "yaml",        ".toml":"toml",
        ".json": "json",    ".md":   "markdown",
    }
    return ext_map.get(Path(path).suffix.lower(), "")


def _extract_code_block(text: str, lang: str) -> str:
    """Extract the first fenced code block from LLM output."""
    if lang:
        pattern = rf"```{re.escape(lang)}\s*\n([\s\S]*?)```"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip()
    m = re.search(r"```[a-z]*\s*\n([\s\S]*?)```", text)
    if m:
        return m.group(1).rstrip()
    return text.strip()


def run(params: dict, client) -> str:
    mode        = params.get("mode", "create")
    path        = params.get("path", "")
    instruction = params.get("instruction", "")
    root        = params.get("root", params.get("project_root", PROJECT_ROOT))
    context     = params.get("context", "")

    if not path:
        return "ERROR: 'path' is required."
    if not instruction:
        return "ERROR: 'instruction' is required — describe what to write or change."

    # Resolve relative paths against the project root so files land in the right place
    resolved_path = path if Path(path).is_absolute() else str(Path(root) / path)

    lang           = _infer_lang(resolved_path)
    structure_ctx  = _load_structure_context(root)
    existing_code  = _read_file(resolved_path) if mode in ("edit", "patch") else ""

    system = f"""\
You are Senku, an expert software engineer with 10 billion percent scientific precision.
Tech stack: Django (backend), HTML/CSS/JavaScript (frontend).
Project root: {root}
Always produce clean, idiomatic, well-commented code.
Return ONLY the complete file content inside a fenced code block — no prose outside it.

{"Project dependency map (structure.md):" if structure_ctx else ""}
{structure_ctx}
"""

    if mode == "create":
        user_prompt = f"""\
Create a new {lang or 'code'} file at: {resolved_path}

Instruction:
{instruction}

{f"Additional context:{chr(10)}{context}" if context else ""}

Return the complete file content inside a fenced ```{lang}``` block.
"""
    elif mode in ("edit", "patch"):
        if not existing_code:
            return f"ERROR: file '{resolved_path}' does not exist. Use mode='create' instead."
        user_prompt = f"""\
{"Edit" if mode == "edit" else "Patch"} the {lang or 'code'} file at: {resolved_path}

Current file content:
```{lang}
{existing_code}
```

Instruction:
{instruction}

{f"Additional context:{chr(10)}{context}" if context else ""}

Return the COMPLETE updated file content inside a fenced ```{lang}``` block.
"""
    else:
        return f"ERROR: unknown mode '{mode}'. Use: create | edit | patch"

    try:
        from skills._llm import skill_llm_call
        raw = skill_llm_call(
            client, MODEL,
            [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            max_tokens=TOKENS_CODE_WRITER,
        )
    except Exception as e:
        return f"ERROR: LLM call failed — {e}"

    code = _extract_code_block(raw, lang)

    if not code:
        return f"ERROR: LLM returned no code.\nRaw response:\n{raw[:500]}"

    try:
        p = Path(resolved_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")
    except Exception as e:
        return f"ERROR writing file '{resolved_path}': {e}"

    action = "Created" if mode == "create" else "Updated"
    return (
        f"✔ {action} '{resolved_path}' ({len(code)} chars, {code.count(chr(10))+1} lines)\n\n"
        f"```{lang}\n{code[:1200]}{'...' if len(code) > 1200 else ''}\n```"
    )


SKILL_DEF = {
    "name":        "code_writer",
    "description": (
        "Write or edit a code file (create / edit / patch). "
        "Uses structure.md as context so edits are project-aware. "
        "Prefers Django for backend, HTML/CSS/JS for frontend."
    ),
    "params": [
        {"name": "mode",        "type": "str", "required": True,  "description": "create | edit | patch"},
        {"name": "path",        "type": "str", "required": True,  "description": "Absolute or relative path to the file"},
        {"name": "instruction", "type": "str", "required": True,  "description": "What to write or what change to make"},
        {"name": "root",        "type": "str", "required": False, "description": "Project root for structure.md lookup (default: PROJECT_ROOT)"},
        {"name": "context",     "type": "str", "required": False, "description": "Extra context to pass to the LLM"},
    ],
    "fn": run,
    "needs_client": True,
}
