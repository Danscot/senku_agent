"""
tools/file_tool.py
Read and write files on the local filesystem.
"""

import os
from pathlib import Path


def _run(params: dict) -> str:
    op           = params.get("op", "read")          # "read" | "write" | "list" | "delete"
    path         = params.get("path", "")
    content      = params.get("content", "")
    encoding     = params.get("encoding", "utf-8")
    project_root = params.get("project_root", "")

    if not path:
        return "ERROR: 'path' is required."

    raw_path = Path(path).expanduser()
    # Resolve relative paths against project_root so the LLM doesn't have to
    # know the agent's working directory
    if not raw_path.is_absolute() and project_root:
        p = Path(project_root).expanduser() / raw_path
    else:
        p = raw_path

    # ── LIST ─────────────────────────────────────────────────────────────────
    if op == "list":
        target = p if p.is_dir() else p.parent
        if not target.exists():
            return f"ERROR: directory '{target}' does not exist."
        entries = sorted(target.iterdir())
        lines = []
        for e in entries:
            kind = "DIR " if e.is_dir() else "FILE"
            size = f"{e.stat().st_size:>10} B" if e.is_file() else ""
            lines.append(f"  {kind}  {size}  {e.name}")
        return f"Contents of {target}:\n" + "\n".join(lines)

    # ── READ ──────────────────────────────────────────────────────────────────
    if op == "read":
        if not p.exists():
            return f"ERROR: file '{p}' does not exist."
        try:
            text = p.read_text(encoding=encoding)
            # Cap at 8 000 chars to avoid blowing context
            if len(text) > 8000:
                text = text[:8000] + f"\n\n[... truncated — {len(text)} total chars]"
            return text
        except Exception as e:
            return f"ERROR reading file: {e}"

    # ── WRITE ─────────────────────────────────────────────────────────────────
    if op == "write":
        if not content:
            return "ERROR: 'content' is required for write op."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)
            return f"✔ Wrote {len(content)} chars to '{p}'"
        except Exception as e:
            return f"ERROR writing file: {e}"

    # ── APPEND ────────────────────────────────────────────────────────────────
    if op == "append":
        if not content:
            return "ERROR: 'content' is required for append op."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding=encoding) as f:
                f.write(content)
            return f"✔ Appended {len(content)} chars to '{p}'"
        except Exception as e:
            return f"ERROR appending to file: {e}"

    # ── DELETE ────────────────────────────────────────────────────────────────
    if op == "delete":
        if not p.exists():
            return f"ERROR: file '{p}' does not exist."
        try:
            p.unlink()
            return f"✔ Deleted '{p}'"
        except Exception as e:
            return f"ERROR deleting file: {e}"

    return f"ERROR: unknown op '{op}'. Use read | write | append | list | delete."


TOOL_DEF = {
    "name":        "file",
    "description": "Read, write, append, list, or delete files on the local filesystem.",
    "params": [
        {"name": "op",       "type": "str",  "required": True,  "description": "read | write | append | list | delete"},
        {"name": "path",     "type": "str",  "required": True,  "description": "Absolute or relative file / directory path"},
        {"name": "content",  "type": "str",  "required": False, "description": "Content to write (write / append only)"},
        {"name": "encoding", "type": "str",  "required": False, "description": "File encoding (default utf-8)"},
    ],
    "fn": _run,
}
