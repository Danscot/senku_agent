"""
tools/git_tool.py
Git operations: status, diff, log, add, commit, checkout, branch, stash.
Runs git commands safely inside the configured project root.
"""

import subprocess
import shlex
import os
from config import PROJECT_ROOT

_MAX_OUTPUT = 6000  # chars


def _git(args: list[str], cwd: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = (result.stdout or "") + (("\n[stderr]\n" + result.stderr) if result.stderr.strip() else "")
        out = out.strip()
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n\n[... truncated — {len(out)} total chars]"
        if not out:
            out = f"(no output — exit code {result.returncode})"
        if result.returncode != 0:
            out += f"\n[exit {result.returncode}]"
        return out
    except FileNotFoundError:
        return "ERROR: git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return f"ERROR: git command timed out after {timeout}s."
    except Exception as e:
        return f"ERROR: {e}"


def _run(params: dict) -> str:
    op      = params.get("op", "status")
    cwd     = params.get("cwd", PROJECT_ROOT)
    message = params.get("message", "")
    files   = params.get("files", ".")
    branch  = params.get("branch", "")
    ref     = params.get("ref", "HEAD")

    if not os.path.isdir(cwd):
        return f"ERROR: directory '{cwd}' does not exist."

    # ── status ────────────────────────────────────────────────────────────────
    if op == "status":
        return _git(["status", "--short", "--branch"], cwd)

    # ── diff ──────────────────────────────────────────────────────────────────
    if op == "diff":
        args = ["diff"]
        if params.get("staged"):
            args.append("--cached")
        if files and files != ".":
            args.append("--")
            args.extend(files.split() if isinstance(files, str) else files)
        return _git(args, cwd)

    # ── log ───────────────────────────────────────────────────────────────────
    if op == "log":
        n = int(params.get("n", 10))
        return _git(["log", f"--oneline", f"-{n}"], cwd)

    # ── add ───────────────────────────────────────────────────────────────────
    if op == "add":
        targets = files if isinstance(files, list) else files.split()
        return _git(["add"] + targets, cwd)

    # ── commit ────────────────────────────────────────────────────────────────
    if op == "commit":
        if not message:
            return "ERROR: 'message' is required for commit."
        return _git(["commit", "-m", message], cwd)

    # ── checkout ──────────────────────────────────────────────────────────────
    if op == "checkout":
        if not branch:
            return "ERROR: 'branch' is required for checkout."
        args = ["checkout"]
        if params.get("create"):
            args.append("-b")
        args.append(branch)
        return _git(args, cwd)

    # ── branch ────────────────────────────────────────────────────────────────
    if op == "branch":
        return _git(["branch", "-a"], cwd)

    # ── stash ─────────────────────────────────────────────────────────────────
    if op == "stash":
        action = params.get("action", "push")  # push | pop | list
        return _git(["stash", action], cwd)

    # ── show ──────────────────────────────────────────────────────────────────
    if op == "show":
        return _git(["show", ref, "--stat"], cwd)

    return f"ERROR: unknown git op '{op}'. Use: status | diff | log | add | commit | checkout | branch | stash | show"


TOOL_DEF = {
    "name":        "git",
    "description": "Run git operations (status, diff, log, add, commit, checkout, branch, stash, show) inside the project.",
    "params": [
        {"name": "op",      "type": "str", "required": True,  "description": "status | diff | log | add | commit | checkout | branch | stash | show"},
        {"name": "cwd",     "type": "str", "required": False, "description": "Working directory (default: PROJECT_ROOT)"},
        {"name": "message", "type": "str", "required": False, "description": "Commit message (commit op)"},
        {"name": "files",   "type": "str", "required": False, "description": "File(s) to add/diff (space-separated or '.' for all)"},
        {"name": "branch",  "type": "str", "required": False, "description": "Branch name (checkout op)"},
        {"name": "create",  "type": "bool","required": False, "description": "Create branch if it doesn't exist (checkout -b)"},
        {"name": "staged",  "type": "bool","required": False, "description": "Diff staged changes (diff op)"},
        {"name": "n",       "type": "int", "required": False, "description": "Number of log entries (default 10)"},
        {"name": "ref",     "type": "str", "required": False, "description": "Commit ref for show op (default HEAD)"},
        {"name": "action",  "type": "str", "required": False, "description": "Stash action: push | pop | list"},
    ],
    "fn": _run,
}
