"""
tools/search_tool.py
Web search via DuckDuckGo HTML scraping (no API key required).

Key fix: DuckDuckGo wraps real URLs inside redirect links like:
  //duckduckgo.com/l/?uddg=ENCODED_REAL_URL&...
This module decodes those to return proper clickable URLs.
"""

import urllib.request
import urllib.parse
import html
import re


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _decode_ddg_href(href: str) -> str:
    """
    DDG result links look like:
      //duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.youtube.com%2F...&rut=...
    Extract and decode the real URL from the uddg param.
    """
    href = html.unescape(href)
    if "uddg=" in href:
        m = re.search(r"[?&]uddg=([^&]+)", href)
        if m:
            return urllib.parse.unquote(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    encoded = urllib.parse.quote_plus(query)
    url     = f"https://html.duckduckgo.com/html/?q={encoded}"

    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as resp:
        body = resp.read().decode("utf-8", errors="ignore")

    results = []

    # Each web result lives in a <div class="result results_links ..."> block
    block_re   = re.compile(r'<div class="result results_links[^"]*".*?(?=<div class="result|$)', re.DOTALL)
    title_re   = re.compile(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)>', re.DOTALL)

    for block_m in block_re.finditer(body):
        if len(results) >= max_results:
            break
        block = block_m.group(0)

        t_m = title_re.search(block)
        s_m = snippet_re.search(block)

        if not t_m:
            continue

        raw_href = t_m.group(1)
        title    = html.unescape(_strip_tags(t_m.group(2))).strip()
        snippet  = html.unescape(_strip_tags(s_m.group(1))).strip() if s_m else ""
        real_url = _decode_ddg_href(raw_href)

        # Skip DDG internal pages and empty results
        if not title or "duckduckgo.com" in real_url:
            continue

        results.append({"title": title, "url": real_url, "snippet": snippet})

    return results


def _run(params: dict) -> str:
    query       = params.get("query", "").strip()
    max_results = int(params.get("max_results", 6))

    if not query:
        return "ERROR: 'query' is required."

    try:
        results = _ddg_search(query, max_results)
    except Exception as e:
        return f"ERROR: search failed — {e}"

    if not results:
        return f"No results found for: {query}"

    lines = [f'Search: "{query}"\n']
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        if r["url"]:
            lines.append(f"    URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet']}")
        lines.append("")

    return "\n".join(lines)


TOOL_DEF = {
    "name":        "search",
    "description": "Search the web using DuckDuckGo. Returns titles, real clickable URLs, and snippets.",
    "params": [
        {"name": "query",       "type": "str", "required": True,  "description": "Search query"},
        {"name": "max_results", "type": "int", "required": False, "description": "Max results (default 6)"},
    ],
    "fn": _run,
}
