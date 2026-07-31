"""
tools/shell_tool.py
Execute shell commands in a subprocess with a configurable timeout.
Output is capped to avoid context overflow.
"""

import subprocess
import shlex
import os


_MAX_OUTPUT = 4000  # chars


def _run(params: dict) -> str:
    cmd     = params.get("cmd", "").strip()
    cwd     = params.get("cwd", os.getcwd())
    timeout = int(params.get("timeout", 30))

    if not cmd:
        return "ERROR: 'cmd' is required."

    # Basic safety: block obviously destructive patterns
    danger = ["rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero of=/dev/"]
    for d in danger:
        if d in cmd:
            return f"ERROR: command blocked — contains dangerous pattern '{d}'."

    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        out = ""
        if stdout:
            out += stdout
        if stderr:
            out += f"\n[stderr]\n{stderr}"

        out = out.strip()

        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n\n[... output truncated — {len(out)} total chars]"

        if not out:
            out = f"(no output — exit code {result.returncode})"

        rc_label = "" if result.returncode == 0 else f"\n[exit {result.returncode}]"
        return out + rc_label

    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s."
    except FileNotFoundError as e:
        return f"ERROR: command not found — {e}"
    except Exception as e:
        return f"ERROR: {e}"


TOOL_DEF = {
    "name":        "shell",
    "description": "Run a shell command and return stdout/stderr. cwd defaults to the project root.",
    "params": [
        {"name": "cmd",          "type": "str", "required": True,  "description": "The shell command to run"},
        {"name": "cwd",          "type": "str", "required": False, "description": "Working directory (default: project root)"},
        {"name": "timeout",      "type": "int", "required": False, "description": "Timeout in seconds (default 30)"},
        {"name": "project_root", "type": "str", "required": False, "description": "Injected automatically — do not set manually"},
    ],
    "fn": _run,
}
