"""
skills/test_runner.py
Run a test suite (pytest / Django test runner / Jest / npm test),
capture output, then use the LLM to interpret failures and suggest fixes.

The agent reads structure.md to understand which files are covered by
failing tests, so its fix suggestions are project-aware.
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from config import MODEL, PROJECT_ROOT


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
        return text[:2500] + ("\n[... truncated]" if len(text) > 2500 else "")
    except Exception:
        return ""


def _run_cmd(args: list[str], cwd: str, timeout: int = 120) -> tuple[str, int]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = (result.stdout or "") + (("\n[stderr]\n" + result.stderr) if result.stderr.strip() else "")
        out = out.strip()
        if len(out) > 6000:
            # Keep the end — failures are usually at the bottom
            out = "[... output trimmed ...]\n" + out[-6000:]
        return out, result.returncode
    except subprocess.TimeoutExpired:
        return f"ERROR: test command timed out after {timeout}s.", 1
    except FileNotFoundError as e:
        return f"ERROR: command not found — {e}", 1
    except Exception as e:
        return f"ERROR: {e}", 1


def _find_manage_py(cwd: str) -> str | None:
    p = Path(cwd)
    for candidate in [p, *p.parents]:
        mp = candidate / "manage.py"
        if mp.exists():
            return str(mp)
    return None


def run(params: dict, client) -> str:
    runner  = params.get("runner", "auto")   # auto | pytest | django | jest | npm
    cwd     = params.get("cwd", PROJECT_ROOT)
    target  = params.get("target", "")       # specific test file/module/label
    root    = params.get("root", PROJECT_ROOT)
    verbose = params.get("verbose", False)
    python  = sys.executable

    if not os.path.isdir(cwd):
        return f"ERROR: directory '{cwd}' does not exist."

    # ── Auto-detect runner ────────────────────────────────────────────────────
    if runner == "auto":
        if _find_manage_py(cwd):
            runner = "django"
        elif (Path(cwd) / "package.json").exists():
            runner = "npm"
        else:
            runner = "pytest"

    # ── Build command ─────────────────────────────────────────────────────────
    if runner == "pytest":
        args = [python, "-m", "pytest", "-v" if verbose else "-q"]
        if target:
            args.append(target)
        run_cwd = cwd

    elif runner == "django":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found. Set cwd to the Django project root."
        args = [python, manage, "test"]
        if verbose:
            args.append("--verbosity=2")
        if target:
            args.append(target)
        run_cwd = os.path.dirname(manage)

    elif runner == "jest":
        args = ["npx", "jest", "--no-coverage"]
        if verbose:
            args.append("--verbose")
        if target:
            args.extend(["--testPathPattern", target])
        run_cwd = cwd

    elif runner == "npm":
        args = ["npm", "test", "--", "--watchAll=false"]
        if target:
            args.extend(["--testPathPattern", target])
        run_cwd = cwd

    else:
        return f"ERROR: unknown runner '{runner}'. Use: auto | pytest | django | jest | npm"

    # ── Run the tests ─────────────────────────────────────────────────────────
    output, returncode = _run_cmd(args, run_cwd, timeout=180)
    passed = returncode == 0

    status_line = "✔ All tests passed." if passed else "✖ Tests failed."

    if passed:
        return f"{status_line}\n\n```\n{output}\n```"

    # ── Tests failed — ask LLM to interpret and suggest fixes ─────────────────
    structure_ctx = _load_structure_context(root)

    prompt = f"""\
Test suite failed. Analyze the output and suggest concrete fixes.

Runner: {runner}
Working directory: {run_cwd}

Test output:
```
{output}
```

{f"Project dependency map (structure.md):{chr(10)}{structure_ctx}" if structure_ctx else ""}

Provide:
1. A summary of what failed and why (be specific about file/line/function)
2. The root cause for each failure
3. Concrete code fixes — show the actual change needed, not just a description
4. If a failure in one file likely cascades (based on structure.md), mention it
"""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        analysis = "".join(chunks).strip()
    except Exception as e:
        analysis = f"(LLM analysis failed: {e})"

    return (
        f"{status_line}\n\n"
        f"```\n{output}\n```\n\n"
        f"## Agent Analysis & Fix Suggestions\n\n{analysis}"
    )


SKILL_DEF = {
    "name":        "test_runner",
    "description": (
        "Run tests (pytest, Django test runner, Jest, npm test) and "
        "interpret failures with fix suggestions. Uses structure.md to trace cascading failures."
    ),
    "params": [
        {"name": "runner",  "type": "str",  "required": False, "description": "auto | pytest | django | jest | npm (default: auto-detect)"},
        {"name": "cwd",     "type": "str",  "required": False, "description": "Directory to run tests from (default: PROJECT_ROOT)"},
        {"name": "target",  "type": "str",  "required": False, "description": "Specific test file, module, or label to run"},
        {"name": "verbose", "type": "bool", "required": False, "description": "Enable verbose output (default: False)"},
        {"name": "root",    "type": "str",  "required": False, "description": "Project root for structure.md (default: PROJECT_ROOT)"},
    ],
    "fn": run,
    "needs_client": True,
}
