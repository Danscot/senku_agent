"""
tools/file_tool.py
Read and write files on the local filesystem.

Truncation strategy:
  Instead of silently cutting content, the tool tells the LLM exactly:
    - Total file size
    - How many chars were returned
    - What chunk index to request next
  This lets the LLM make multiple read calls to page through large files
  without losing data or being silently blinded.
"""

import os
from pathlib import Path

# Default chunk size — tuned to leave room in context for the rest of the prompt.
# The LLM can override this with the 'chunk_size' param.
_DEFAULT_CHUNK = 12_000
_MAX_CHUNK     = 40_000   # hard ceiling — protects context window


def _run(params: dict) -> str:
    op           = params.get("op", "read")
    path         = params.get("path", "")
    content      = params.get("content", "")
    encoding     = params.get("encoding", "utf-8")
    project_root = params.get("project_root", "")
    chunk_size   = min(int(params.get("chunk_size", _DEFAULT_CHUNK)), _MAX_CHUNK)
    chunk_index  = int(params.get("chunk_index", 0))   # 0-based page number

    if not path:
        return "ERROR: 'path' is required."

    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute() and project_root:
        p = Path(project_root).expanduser() / raw_path
    else:
        p = raw_path

    # ── LIST ──────────────────────────────────────────────────────────────────
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
            text       = p.read_text(encoding=encoding)
            total_chars = len(text)
            total_chunks = max(1, -(-total_chars // chunk_size))  # ceiling div

            start = chunk_index * chunk_size
            if start >= total_chars:
                return (
                    f"ERROR: chunk_index {chunk_index} is out of range. "
                    f"File has {total_chunks} chunk(s) (0-based: 0–{total_chunks-1})."
                )

            end   = min(start + chunk_size, total_chars)
            chunk = text[start:end]

            # Build a clear header so the LLM always knows where it is
            if total_chars <= chunk_size:
                # Fits in one chunk — no pagination header needed
                return chunk
            else:
                remaining = total_chunks - chunk_index - 1
                header = (
                    f"[FILE: {p.name} | "
                    f"chars {start+1}–{end} of {total_chars} | "
                    f"chunk {chunk_index+1}/{total_chunks}"
                    + (f" | {remaining} more chunk(s) — call with chunk_index={chunk_index+1} to continue" if remaining > 0 else " | FINAL CHUNK")
                    + "]\n\n"
                )
                return header + chunk

        except Exception as e:
            return f"ERROR reading file: {e}"

    # ── READ_ALL — read entire file regardless of size (use carefully) ─────────
    if op == "read_all":
        if not p.exists():
            return f"ERROR: file '{p}' does not exist."
        try:
            text = p.read_text(encoding=encoding)
            size = len(text)
            warn = (
                f"\n\n[WARNING: file is {size:,} chars — this may consume significant context]\n"
                if size > _DEFAULT_CHUNK else ""
            )
            return text + warn
        except Exception as e:
            return f"ERROR reading file: {e}"

    # ── WRITE ─────────────────────────────────────────────────────────────────
    if op == "write":
        if not content:
            return "ERROR: 'content' is required for write op."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)
            return f"✔ Wrote {len(content):,} chars to '{p}'"
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
            return f"✔ Appended {len(content):,} chars to '{p}'"
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

    # ── INFO — file metadata without reading content ───────────────────────────
    if op == "info":
        if not p.exists():
            return f"ERROR: path '{p}' does not exist."
        stat = p.stat()
        size = stat.st_size
        chunks = max(1, -(-size // chunk_size))
        return (
            f"File: {p}\n"
            f"Size: {size:,} bytes\n"
            f"Chunks at chunk_size={chunk_size}: {chunks}\n"
            f"To read: op=read, chunk_index=0 through {chunks-1}"
        )

    return f"ERROR: unknown op '{op}'. Use: read | read_all | write | append | list | delete | info"


TOOL_DEF = {
    "name":        "file",
    "description": (
        "Read, write, append, list, or delete files. "
        "Large files are returned in chunks — the response header tells you the total size "
        "and which chunk_index to request next. Always read all chunks before writing a report. "
        "Use op=info to check file size before reading. Use op=read_all only for small files."
    ),
    "params": [
        {"name": "op",          "type": "str",  "required": True,
         "description": "read | read_all | write | append | list | delete | info"},
        {"name": "path",        "type": "str",  "required": True,
         "description": "Absolute or relative file / directory path"},
        {"name": "content",     "type": "str",  "required": False,
         "description": "Content to write (write / append only)"},
        {"name": "encoding",    "type": "str",  "required": False,
         "description": "File encoding (default utf-8)"},
        {"name": "chunk_size",  "type": "int",  "required": False,
         "description": f"Chars per chunk (default {_DEFAULT_CHUNK}, max {_MAX_CHUNK})"},
        {"name": "chunk_index", "type": "int",  "required": False,
         "description": "0-based chunk index for paginated reads (default 0)"},
    ],
    "fn": _run,
}
