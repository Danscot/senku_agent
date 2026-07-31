# Mahogara — MCP Servers Reference

MCP (Model Context Protocol) servers are optional external integrations.
They are only loaded when `enabled: True` (usually gated on an env var).

Call syntax in JSON plans: `"tool": "mcp:<name>"`

---

## brave_search  *(optional)*

Web search via the Brave Search API — higher quality than DuckDuckGo.

**Setup:** Add `BRAVE_API_KEY=<key>` to your `.env`. The server auto-enables.

| Param         | Type | Required | Description          |
|---------------|------|----------|----------------------|
| `query`       | str  | ✔        | Search query         |
| `max_results` | int  | –        | Max results (default 5) |

---

## Adding a New MCP Server

1. Create `mcp/my_server.py`:
   ```python
   import os

   def _run(params: dict) -> str:
       api_key = os.getenv("MY_API_KEY", "")
       if not api_key:
           return "ERROR: MY_API_KEY not set."
       # ... call external service
       return result

   MCP_DEF = {
       "name":        "my_server",
       "description": "Calls the My Service API.",
       "enabled":     bool(os.getenv("MY_API_KEY")),
       "params":      [{"name": "query", "type": "str", "required": True}],
       "fn":          _run,
   }
   ```
2. Import and `_register` it in `mcp/__init__.py`.

The agent discovers enabled servers at startup and injects them into the
system prompt catalogue automatically.
