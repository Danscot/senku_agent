"""
tools/__init__.py
Auto-registers all tools. Each tool module must expose a TOOL_DEF dict.
"""

from .file_tool    import TOOL_DEF as FILE_DEF
from .shell_tool   import TOOL_DEF as SHELL_DEF
from .git_tool     import TOOL_DEF as GIT_DEF
from .django_tool  import TOOL_DEF as DJANGO_DEF
from .search_tool  import TOOL_DEF as SEARCH_DEF
from .ask_tool     import TOOL_DEF as ASK_DEF

REGISTRY: dict[str, dict] = {}


def _register(tool_def: dict):
    REGISTRY[tool_def["name"]] = tool_def


_register(FILE_DEF)
_register(SHELL_DEF)
_register(GIT_DEF)
_register(DJANGO_DEF)
_register(SEARCH_DEF)
_register(ASK_DEF)


def get(name: str) -> dict | None:
    return REGISTRY.get(name)


def all_tools() -> list[dict]:
    return list(REGISTRY.values())


def describe_all() -> str:
    """Return a compact description string injected into the system prompt."""
    lines = []
    for t in REGISTRY.values():
        params = ", ".join(
            f"{p['name']} ({p['type']})" + (" [optional]" if not p.get("required", True) else "")
            for p in t.get("params", [])
        )
        lines.append(f"  - {t['name']}: {t['description']}  |  params: {params}")
    return "\n".join(lines)
