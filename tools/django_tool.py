"""
tools/django_tool.py
Django project operations: scaffold, run server, make/run migrations,
create apps, run management commands, check setup.

Works by invoking manage.py or django-admin via shell.
"""

import subprocess
import os
import sys
from pathlib import Path
from config import PROJECT_ROOT

_MAX_OUTPUT = 5000


def _run_cmd(args: list[str], cwd: str, timeout: int = 60, env_extra: dict | None = None) -> str:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        out = (result.stdout or "") + (("\n[stderr]\n" + result.stderr) if result.stderr.strip() else "")
        out = out.strip()
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n[... truncated]"
        if not out:
            out = f"(no output — exit code {result.returncode})"
        if result.returncode != 0:
            out += f"\n[exit {result.returncode}]"
        return out
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s."
    except FileNotFoundError as e:
        return f"ERROR: command not found — {e}"
    except Exception as e:
        return f"ERROR: {e}"


def _find_manage_py(cwd: str) -> str | None:
    """Search for manage.py upward from cwd."""
    p = Path(cwd)
    for candidate in [p, *p.parents]:
        mp = candidate / "manage.py"
        if mp.exists():
            return str(mp)
    return None


def _python() -> str:
    return sys.executable


def _run(params: dict) -> str:
    op      = params.get("op", "check")
    cwd     = params.get("cwd", PROJECT_ROOT)
    name    = params.get("name", "")          # project/app name
    port    = params.get("port", "8000")
    app     = params.get("app", "")
    cmd     = params.get("cmd", "")           # raw manage.py command

    python  = _python()

    # ── scaffold: create new django project ───────────────────────────────────
    if op == "scaffold":
        if not name:
            return "ERROR: 'name' is required to scaffold a Django project."
        # Install django if missing
        check = subprocess.run([python, "-c", "import django"], capture_output=True)
        if check.returncode != 0:
            inst = _run_cmd([python, "-m", "pip", "install", "django"], cwd, timeout=120)
            if "ERROR" in inst or "error" in inst.lower():
                return f"ERROR: could not install Django.\n{inst}"

        out = _run_cmd(
            [python, "-m", "django", "startproject", name, "."],
            cwd,
        )
        if "ERROR" not in out and "[exit" not in out:
            # Also scaffold a basic structure.md
            structure_path = Path(cwd) / "structure.md"
            if not structure_path.exists():
                structure_path.write_text(
                    f"# {name} — Project Structure\n\n"
                    "This file is maintained by the coding agent.\n"
                    "It describes the dependency graph between files.\n\n"
                    "## Files\n\n"
                    "_Run `/scan` to auto-generate the dependency tree._\n",
                    encoding="utf-8",
                )
            out += f"\n✔ Created structure.md at {structure_path}"
        return out

    # ── startapp: create a new app inside the project ─────────────────────────
    if op == "startapp":
        if not name:
            return "ERROR: 'name' is required for startapp."
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found. Run 'scaffold' first or set 'cwd' to the project root."
        return _run_cmd([python, manage, "startapp", name], os.path.dirname(manage))

    # ── migrate ───────────────────────────────────────────────────────────────
    if op == "migrate":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        args = [python, manage, "migrate"]
        if app:
            args.append(app)
        return _run_cmd(args, os.path.dirname(manage))

    # ── makemigrations ────────────────────────────────────────────────────────
    if op == "makemigrations":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        args = [python, manage, "makemigrations"]
        if app:
            args.append(app)
        return _run_cmd(args, os.path.dirname(manage))

    # ── runserver (non-blocking, just validates it starts) ────────────────────
    if op == "runserver":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        # We do a system check instead of actually starting the server
        # (the server would block forever)
        return _run_cmd([python, manage, "check", "--deploy"], os.path.dirname(manage))

    # ── check ─────────────────────────────────────────────────────────────────
    if op == "check":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found. Is this a Django project?"
        return _run_cmd([python, manage, "check"], os.path.dirname(manage))

    # ── showmigrations ────────────────────────────────────────────────────────
    if op == "showmigrations":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        return _run_cmd([python, manage, "showmigrations"], os.path.dirname(manage))

    # ── test ──────────────────────────────────────────────────────────────────
    if op == "test":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        args = [python, manage, "test"]
        if app:
            args.append(app)
        return _run_cmd(args, os.path.dirname(manage), timeout=120)

    # ── createsuperuser (non-interactive via env vars) ────────────────────────
    if op == "createsuperuser":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        user     = params.get("username", "admin")
        email    = params.get("email", "admin@example.com")
        password = params.get("password", "admin123")
        return _run_cmd(
            [python, manage, "createsuperuser", "--noinput",
             f"--username={user}", f"--email={email}"],
            os.path.dirname(manage),
            env_extra={"DJANGO_SUPERUSER_PASSWORD": password},
        )

    # ── collectstatic ─────────────────────────────────────────────────────────
    if op == "collectstatic":
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        return _run_cmd([python, manage, "collectstatic", "--noinput"], os.path.dirname(manage))

    # ── raw management command ────────────────────────────────────────────────
    if op == "manage":
        if not cmd:
            return "ERROR: 'cmd' is required for manage op. E.g. cmd='shell --command=\"print(1)\"'"
        manage = _find_manage_py(cwd)
        if not manage:
            return "ERROR: manage.py not found."
        import shlex
        return _run_cmd([python, manage] + shlex.split(cmd), os.path.dirname(manage))

    # ── install django + deps ─────────────────────────────────────────────────
    if op == "install":
        pkg = params.get("packages", "django djangorestframework django-cors-headers")
        pkgs = pkg.split()
        return _run_cmd([python, "-m", "pip", "install"] + pkgs, cwd, timeout=180)

    return (
        f"ERROR: unknown django op '{op}'. "
        "Use: scaffold | startapp | migrate | makemigrations | runserver | "
        "check | showmigrations | test | createsuperuser | collectstatic | manage | install"
    )


TOOL_DEF = {
    "name":        "django",
    "description": (
        "Manage a Django project: scaffold new project, create apps, "
        "run migrations, run tests, check setup, or execute manage.py commands."
    ),
    "params": [
        {"name": "op",       "type": "str", "required": True,  "description": "scaffold | startapp | migrate | makemigrations | check | showmigrations | test | createsuperuser | collectstatic | manage | install | runserver"},
        {"name": "cwd",      "type": "str", "required": False, "description": "Project directory (default: PROJECT_ROOT)"},
        {"name": "name",     "type": "str", "required": False, "description": "Project or app name (scaffold / startapp)"},
        {"name": "app",      "type": "str", "required": False, "description": "App label for migrate / test"},
        {"name": "port",     "type": "str", "required": False, "description": "Port for runserver (default 8000)"},
        {"name": "cmd",      "type": "str", "required": False, "description": "Raw manage.py sub-command string (manage op)"},
        {"name": "packages", "type": "str", "required": False, "description": "Space-separated packages to pip install (install op)"},
        {"name": "username", "type": "str", "required": False, "description": "Superuser username (createsuperuser)"},
        {"name": "email",    "type": "str", "required": False, "description": "Superuser email (createsuperuser)"},
        {"name": "password", "type": "str", "required": False, "description": "Superuser password (createsuperuser)"},
    ],
    "fn": _run,
}
