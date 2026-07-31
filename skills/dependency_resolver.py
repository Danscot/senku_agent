"""
skills/dependency_resolver.py
Manage project dependencies:
  - Install packages (pip / npm)
  - Check for outdated packages
  - Generate / update requirements.txt or package.json
  - Detect and resolve conflicts
"""

from __future__ import annotations

import subprocess
import sys
import json
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


def _run(args: list[str], cwd: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        out = (result.stdout or "") + (("\n[stderr]\n" + result.stderr) if result.stderr.strip() else "")
        out = out.strip()
        if len(out) > 5000:
            out = out[:5000] + "\n[... truncated]"
        if not out:
            out = f"(no output — exit code {result.returncode})"
        if result.returncode != 0:
            out += f"\n[exit {result.returncode}]"
        return out
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s."
    except FileNotFoundError as e:
        return f"ERROR: command not found — {e}"
    except Exception as e:
        return f"ERROR: {e}"


def _python():
    return sys.executable


def run(params: dict, client) -> str:
    op       = params.get("op", "install")
    packages = params.get("packages", "")
    cwd      = params.get("cwd", PROJECT_ROOT)
    manager  = params.get("manager", "auto")   # auto | pip | npm

    if not os.path.isdir(cwd):
        return f"ERROR: directory '{cwd}' does not exist."

    # Auto-detect manager
    if manager == "auto":
        if (Path(cwd) / "package.json").exists():
            manager = "npm"
        else:
            manager = "pip"

    py = _python()

    # ── install ───────────────────────────────────────────────────────────────
    if op == "install":
        if not packages:
            # Install from requirements.txt / package.json
            if manager == "pip":
                req = Path(cwd) / "requirements.txt"
                if req.exists():
                    return _run([py, "-m", "pip", "install", "-r", str(req)], cwd, 180)
                return "ERROR: no packages specified and no requirements.txt found."
            else:
                return _run(["npm", "install"], cwd, 180)
        if manager == "pip":
            pkgs = packages.split()
            return _run([py, "-m", "pip", "install"] + pkgs, cwd, 180)
        else:
            pkgs = packages.split()
            return _run(["npm", "install"] + pkgs, cwd, 180)

    # ── freeze / generate requirements ────────────────────────────────────────
    if op == "freeze":
        if manager == "pip":
            out = _run([py, "-m", "pip", "freeze"], cwd)
            if "ERROR" not in out:
                req_path = Path(cwd) / "requirements.txt"
                req_path.write_text(out, encoding="utf-8")
                return f"✔ requirements.txt written ({len(out.splitlines())} packages):\n{out}"
            return out
        else:
            try:
                pj = json.loads((Path(cwd) / "package.json").read_text())
                deps = list(pj.get("dependencies", {}).keys()) + list(pj.get("devDependencies", {}).keys())
                return f"package.json has {len(deps)} dependencies: {', '.join(deps)}"
            except Exception as e:
                return f"ERROR reading package.json: {e}"

    # ── outdated ──────────────────────────────────────────────────────────────
    if op == "outdated":
        if manager == "pip":
            out = _run([py, "-m", "pip", "list", "--outdated", "--format=columns"], cwd)
        else:
            out = _run(["npm", "outdated"], cwd)

        # Ask LLM to explain which outdated packages matter
        if client and out and "ERROR" not in out:
            prompt = f"""\
These packages are outdated in the project. Which ones should be updated urgently
(security fixes, breaking changes) vs which can wait? Be concise.

{out}
"""
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    stream=True,
                )
                chunks = []
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        chunks.append(delta)
                advice = "".join(chunks).strip()
                return f"## Outdated packages\n\n```\n{out}\n```\n\n## Upgrade priority\n\n{advice}"
            except Exception:
                pass
        return out

    # ── uninstall ─────────────────────────────────────────────────────────────
    if op == "uninstall":
        if not packages:
            return "ERROR: 'packages' is required for uninstall."
        pkgs = packages.split()
        if manager == "pip":
            return _run([py, "-m", "pip", "uninstall", "-y"] + pkgs, cwd)
        else:
            return _run(["npm", "uninstall"] + pkgs, cwd)

    # ── list ──────────────────────────────────────────────────────────────────
    if op == "list":
        if manager == "pip":
            return _run([py, "-m", "pip", "list", "--format=columns"], cwd)
        else:
            return _run(["npm", "list", "--depth=0"], cwd)

    # ── check (validate install is consistent) ────────────────────────────────
    if op == "check":
        if manager == "pip":
            return _run([py, "-m", "pip", "check"], cwd)
        else:
            return _run(["npm", "audit", "--audit-level=moderate"], cwd)

    return (
        f"ERROR: unknown op '{op}'. "
        "Use: install | freeze | outdated | uninstall | list | check"
    )


SKILL_DEF = {
    "name":        "dependency_resolver",
    "description": (
        "Manage project dependencies: install, freeze requirements.txt, "
        "check for outdated packages, audit, or uninstall. Supports pip and npm."
    ),
    "params": [
        {"name": "op",       "type": "str", "required": True,  "description": "install | freeze | outdated | uninstall | list | check"},
        {"name": "packages", "type": "str", "required": False, "description": "Space-separated package names"},
        {"name": "cwd",      "type": "str", "required": False, "description": "Project directory (default: PROJECT_ROOT)"},
        {"name": "manager",  "type": "str", "required": False, "description": "auto | pip | npm (default: auto-detect)"},
    ],
    "fn": run,
    "needs_client": True,
}
