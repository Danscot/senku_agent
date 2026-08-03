"""
skills/code_reviewer.py
Review code files for bugs, security issues, style problems, and
Django/JS best practices. Uses structure.md to understand impact.

Modes:
  bugs      — find bugs and logical errors
  security  — find security vulnerabilities (Django-specific: CSRF, SQL injection, etc.)
  style     — check code style and Django/JS conventions
  full      — all of the above in one report
"""

from __future__ import annotations

from pathlib import Path
from config import MODEL, PROJECT_ROOT, TOKENS_CODE_REVIEWER, TEMPERATURE_SKILLS


def _safe_extract(resp, default="") -> str:
    if not resp or not resp.choices:
        return default
    content = resp.choices[0].message.content
    if content is None:
        return default
    return content.strip() if isinstance(content, str) else str(content).strip()


def _load_structure_context(root: str) -> str:
    smd = Path(root) / "structure.md"
    if not smd.exists():
        return ""
    try:
        text = smd.read_text(encoding="utf-8")
        return text[:3000] + ("\n[... truncated]" if len(text) > 3000 else "")
    except Exception:
        return ""


def _infer_lang(path: str) -> str:
    ext_map = {
        ".py": "Python/Django", ".js": "JavaScript", ".ts": "TypeScript",
        ".html": "HTML/Django template", ".css": "CSS", ".scss": "SCSS",
    }
    return ext_map.get(Path(path).suffix.lower(), "code")


def run(params: dict, client) -> str:
    path   = params.get("path", "")
    code   = params.get("code", "")
    mode   = params.get("mode", "full").lower()
    root   = params.get("root", PROJECT_ROOT)
    target = params.get("target", "")

    if path and not code:
        try:
            code = Path(path).read_text(encoding="utf-8")
            if len(code) > 10000:
                code = code[:10000] + "\n# [... truncated]"
        except Exception as e:
            return f"ERROR reading '{path}': {e}"

    if not code:
        return "ERROR: provide 'path' or 'code'."

    lang          = _infer_lang(path or "file.py")
    structure_ctx = _load_structure_context(root)
    focus         = f"\nFocus specifically on: {target}" if target else ""

    mode_instructions = {
        "bugs": (
            "Find all bugs, logical errors, off-by-one errors, null-pointer risks, "
            "incorrect assumptions, and runtime exceptions. For each: location, severity "
            "(critical/warning/minor), description, and suggested fix."
        ),
        "security": (
            "Audit for security vulnerabilities. For Django: SQL injection, XSS, CSRF bypass, "
            "missing authentication/permission checks, exposed secrets, insecure file uploads, "
            "mass assignment. For JS: XSS, eval(), unsafe innerHTML, prototype pollution. "
            "For each: location, CWE if applicable, severity, and fix."
        ),
        "style": (
            "Review code style, readability, naming conventions, and framework best practices. "
            "For Django: PEP-8, class-based vs function-based views, ORM usage, serializer patterns. "
            "For JS/CSS: naming, separation of concerns, accessibility. "
            "List improvements with before/after examples where helpful."
        ),
        "full": (
            "Perform a complete code review covering:\n"
            "1. BUGS — logical errors, runtime risks\n"
            "2. SECURITY — vulnerabilities specific to Django/JS stack\n"
            "3. STYLE — conventions and best practices\n"
            "4. IMPACT — given the project structure below, which other files could be affected "
            "if bugs in this file go unfixed?\n"
            "Format as clear sections with severity labels."
        ),
    }

    instruction = mode_instructions.get(mode, mode_instructions["full"])

    system = f"""\
You are a senior software engineer specializing in Django backends and JavaScript/HTML/CSS frontends.
You write clear, actionable code reviews with specific line references.
{f"Project dependency map:{chr(10)}{structure_ctx}" if structure_ctx else ""}
"""

    user_prompt = f"""\
Review this {lang} file: {path or "(snippet)"}
{focus}

Code:
```
{code}
```

{instruction}
"""

    try:
        from skills._llm import skill_llm_call
        review = skill_llm_call(
            client, MODEL,
            [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            max_tokens=TOKENS_CODE_REVIEWER,
            provider=params.get("provider", ""),
        )
    except Exception as e:
        return f"ERROR: LLM call failed — {e}"

    return f"## Code Review — `{path or 'snippet'}` [{mode}]\n\n{review}"


SKILL_DEF = {
    "name":        "code_reviewer",
    "description": (
        "Review code for bugs, security issues, or style problems. "
        "Django and JS/HTML/CSS aware. Uses structure.md to assess cross-file impact."
    ),
    "params": [
        {"name": "path",   "type": "str", "required": False, "description": "Path to the file to review"},
        {"name": "code",   "type": "str", "required": False, "description": "Raw code snippet (alternative to path)"},
        {"name": "mode",   "type": "str", "required": False, "description": "bugs | security | style | full (default: full)"},
        {"name": "root",   "type": "str", "required": False, "description": "Project root for structure.md (default: PROJECT_ROOT)"},
        {"name": "target", "type": "str", "required": False, "description": "Focus on a specific function/class"},
    ],
    "fn": run,
    "needs_client": True,
}
