"""
mcp/__init__.py
MCP (Model Context Protocol) server registry.
Each server module exposes an MCP_DEF dict.

To add a new MCP server:
  1. Create mcp/my_server.py with an MCP_DEF dict
  2. Import and _register it here
"""

# ── Load MCP servers ──────────────────────────────────────────────────────────
REGISTRY: dict[str, dict] = {}


def _register(mcp_def: dict):
    if mcp_def.get("enabled", False):
        REGISTRY[mcp_def["name"]] = mcp_def


# Wikipedia — always available, no API key
try:
    from .wikipedia import MCP_DEF as WIKI_DEF
    _register(WIKI_DEF)
except ImportError:
    pass


def get(name: str) -> dict | None:
    return REGISTRY.get(name)


def all_servers() -> list[dict]:
    return list(REGISTRY.values())


def describe_all() -> str:
    if not REGISTRY:
        return "  (no MCP servers enabled)"
    lines = []
    for s in REGISTRY.values():
        lines.append(f"  - mcp:{s['name']}: {s['description']}")
    return "\n".join(lines)
