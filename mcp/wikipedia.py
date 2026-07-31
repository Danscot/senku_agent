"""
mcp/wikipedia.py
Free Wikipedia search and article fetching via the public MediaWiki API.
No API key required. Always enabled.
"""

import urllib.request
import urllib.parse
import json


def _api(params: dict) -> dict:
    base = "https://en.wikipedia.org/w/api.php"
    qs   = urllib.parse.urlencode({**params, "format": "json"})
    url  = f"{base}?{qs}"
    req  = urllib.request.Request(url, headers={"User-Agent": "Mahogara-Agent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _run(params: dict) -> str:
    op    = params.get("op", "search")     # "search" | "summary" | "content"
    query = params.get("query", "").strip()

    if not query:
        return "ERROR: 'query' is required."

    # ── SEARCH ────────────────────────────────────────────────────────────────
    if op == "search":
        limit = int(params.get("limit", 5))
        data  = _api({
            "action":   "query",
            "list":     "search",
            "srsearch": query,
            "srlimit":  limit,
        })
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results for: {query}"
        lines = [f"Wikipedia search: \"{query}\"\n"]
        for r in results:
            title   = r.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", r.get("snippet", ""))
            lines.append(f"• {title}")
            lines.append(f"  {snippet}")
        return "\n".join(lines)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    if op == "summary":
        title_enc = urllib.parse.quote(query.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_enc}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mahogara-Agent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return f"ERROR: {e}"
        title   = data.get("title", query)
        extract = data.get("extract", "No summary available.")
        url_out = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"# {title}\n\n{extract}\n\n{url_out}"

    # ── FULL CONTENT ──────────────────────────────────────────────────────────
    if op == "content":
        data = _api({
            "action":    "query",
            "prop":      "extracts",
            "exintro":   False,
            "titles":    query,
            "explaintext": True,
        })
        pages = data.get("query", {}).get("pages", {})
        page  = next(iter(pages.values()))
        if page.get("missing") is not None:
            return f"No Wikipedia page found for: {query}"
        title   = page.get("title", query)
        extract = page.get("extract", "")
        if len(extract) > 8000:
            extract = extract[:8000] + "\n\n[... content truncated]"
        return f"# {title}\n\n{extract}"

    return f"ERROR: unknown op '{op}'. Use search | summary | content."


# lazy import
import re

MCP_DEF = {
    "name":        "wikipedia",
    "description": "Search Wikipedia or fetch article summaries/content. No API key needed.",
    "enabled":     True,
    "params": [
        {"name": "op",    "type": "str", "required": False, "description": "search (default) | summary | content"},
        {"name": "query", "type": "str", "required": True,  "description": "Search query or article title"},
        {"name": "limit", "type": "int", "required": False, "description": "Max search results (default 5)"},
    ],
    "fn": _run,
}
