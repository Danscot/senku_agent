"""
skills/web_crawler.py — Web content fetching via Firecrawl.

Unlike the search tool (which returns snippets), this skill fetches the full
content of web pages and converts them to clean markdown — ideal for feeding
detailed documentation, articles, or API references directly to the LLM.

Operations:
  scrape   — fetch a single URL, return its markdown content
  crawl    — crawl a site up to `limit` pages, return all as markdown docs
  search   — Firecrawl's built-in web search with full content extraction
  extract  — extract structured data from a URL using a schema/prompt

The API key is read from FIRECRAWL_API_KEY env var (or config.py fallback).
"""

from __future__ import annotations

import os
import json
import textwrap
from pathlib import Path
from config import PROJECT_ROOT


# ── API key resolution ─────────────────────────────────────────────────────────

def _get_api_key(params: dict) -> str:
    return (
        params.get("api_key")
        or os.getenv("FIRECRAWL_API_KEY", "")
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Ops
# ══════════════════════════════════════════════════════════════════════════════

def _op_scrape(client, params: dict) -> str:
    """Fetch a single URL and return its markdown, with chunked pagination."""
    url        = params.get("url", "").strip()
    formats    = params.get("formats", ["markdown"])
    chunk_size = int(params.get("chunk_size", 24_000))
    chunk_index= int(params.get("chunk_index", 0))

    if not url:
        return "ERROR: 'url' is required for scrape op."

    result = client.scrape_url(url, formats=formats)

    md = getattr(result, "markdown", None) or getattr(result, "content", None) or ""
    if not md:
        return f"ERROR: no markdown content returned for {url}"

    title       = getattr(result, "title", url)
    total_chars = len(md)
    total_chunks = max(1, -(-total_chars // chunk_size))

    start = chunk_index * chunk_size
    if start >= total_chars:
        return (
            f"ERROR: chunk_index {chunk_index} out of range. "
            f"Page has {total_chunks} chunk(s) (0–{total_chunks-1})."
        )

    end   = min(start + chunk_size, total_chars)
    chunk = md[start:end]

    if total_chunks == 1:
        return f"# {title}\nSource: {url}\n\n{chunk}"

    remaining = total_chunks - chunk_index - 1
    header = (
        f"# {title}\n"
        f"Source: {url}\n"
        f"[CONTENT: chars {start+1}–{end} of {total_chars} | "
        f"chunk {chunk_index+1}/{total_chunks}"
        + (f" | {remaining} more chunk(s) remaining — call scrape again with chunk_index={chunk_index+1}" if remaining > 0 else " | FINAL CHUNK — all content retrieved")
        + "]\n\n"
    )
    return header + chunk


def _op_crawl(client, params: dict) -> str:
    """Crawl a site up to limit pages and return all pages as markdown."""
    from firecrawl.types import ScrapeOptions

    url          = params.get("url", "").strip()
    limit        = int(params.get("limit", 10))
    poll_interval= int(params.get("poll_interval", 30))
    chunk_size   = int(params.get("chunk_size", 8_000))   # per page

    if not url:
        return "ERROR: 'url' is required for crawl op."
    if limit > 100:
        return "ERROR: limit must be ≤ 100 to stay within API constraints."

    result = client.crawl(
        url,
        limit=limit,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        poll_interval=poll_interval,
    )

    docs = [
        doc.markdown
        for doc in result.data
        if hasattr(doc, "markdown") and doc.markdown
    ]

    if not docs:
        return f"ERROR: crawl returned no documents for {url}"

    total_chars = sum(len(d) for d in docs)
    parts = [f"Crawled {len(docs)} page(s) from {url} (limit={limit}, total ~{total_chars:,} chars)\n"]

    for i, doc_md in enumerate(docs, 1):
        doc_len = len(doc_md)
        if doc_len > chunk_size:
            # Show the doc but warn that it's clipped and how to get more
            note = (
                f"\n[PAGE {i} CLIPPED at {chunk_size:,}/{doc_len:,} chars — "
                f"use op=scrape on the source URL with chunk_index to read the rest]"
            )
            parts.append(f"## Page {i} ({doc_len:,} chars)\n{doc_md[:chunk_size]}{note}")
        else:
            parts.append(f"## Page {i} ({doc_len:,} chars)\n{doc_md}")

    return "\n\n---\n\n".join(parts)


def _op_search(client, params: dict) -> str:
    """
    Firecrawl's built-in search — full page content per result, not just snippets.
    """
    query      = params.get("query", "").strip()
    limit      = int(params.get("limit", 5))
    chunk_size = int(params.get("chunk_size", 6_000))   # per result

    if not query:
        return "ERROR: 'query' is required for search op."

    result = client.search(query, limit=limit)

    if not result or not getattr(result, "data", None):
        return f"No results found for: {query}"

    parts = [f'Web search: "{query}" — {len(result.data)} result(s)\n']
    for i, item in enumerate(result.data, 1):
        title   = getattr(item, "title",       "") or ""
        url     = getattr(item, "url",         "") or ""
        md      = getattr(item, "markdown",    "") or getattr(item, "content", "") or ""
        snippet = getattr(item, "description", "") or ""

        doc_len = len(md)
        if doc_len > chunk_size:
            note = (
                f"\n[RESULT {i} CLIPPED at {chunk_size:,}/{doc_len:,} chars — "
                f"use op=scrape, url={url} with chunk_index to read the full page]"
            )
            display = md[:chunk_size] + note
        else:
            display = md if md else snippet

        parts.append(f"### [{i}] {title}")
        if url:
            parts.append(f"URL: {url}")
        if display:
            parts.append(display)
        parts.append("")

    return "\n".join(parts)


def _op_extract(client, params: dict) -> str:
    """
    Extract structured data from a URL using a natural-language prompt or schema.
    Returns JSON.
    """
    url    = params.get("url", "").strip()
    prompt = params.get("prompt", "").strip()
    schema = params.get("schema", None)    # optional JSON schema dict

    if not url:
        return "ERROR: 'url' is required for extract op."
    if not prompt and not schema:
        return "ERROR: 'prompt' or 'schema' is required for extract op."

    kwargs: dict = {"url": url}
    if prompt:
        kwargs["prompt"] = prompt
    if schema:
        kwargs["schema"] = schema if isinstance(schema, dict) else json.loads(schema)

    result = client.extract(**kwargs)

    data = getattr(result, "data", result)
    return json.dumps(data, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(params: dict, client_llm=None) -> str:
    op = params.get("op", "scrape").strip()

    VALID_OPS = {"scrape", "crawl", "search", "extract"}
    if op not in VALID_OPS:
        return f"ERROR: unknown op '{op}'. Choose: {', '.join(sorted(VALID_OPS))}"

    api_key = _get_api_key(params)
    if not api_key:
        return (
            "ERROR: Firecrawl API key not found.\n"
            "Set FIRECRAWL_API_KEY in your .env file, or pass 'api_key' in params.\n"
            "Get a free key at https://firecrawl.dev"
        )

    try:
        from firecrawl import FirecrawlApp
        fc = FirecrawlApp(api_key=api_key)
    except ImportError:
        return (
            "ERROR: firecrawl-py not installed.\n"
            "Run: pip install firecrawl-py"
        )
    except Exception as e:
        return f"ERROR: could not initialise Firecrawl client — {e}"

    try:
        if op == "scrape":
            return _op_scrape(fc, params)
        elif op == "crawl":
            return _op_crawl(fc, params)
        elif op == "search":
            return _op_search(fc, params)
        elif op == "extract":
            return _op_extract(fc, params)
    except Exception as e:
        return f"ERROR: Firecrawl {op} failed — {e}"

    return "ERROR: unreachable"


# ══════════════════════════════════════════════════════════════════════════════
#  Skill registration
# ══════════════════════════════════════════════════════════════════════════════

SKILL_DEF = {
    "name": "web_crawler",
    "description": (
        "Fetch full web page content as clean markdown using Firecrawl. "
        "Far richer than the search tool — returns complete page text, not just snippets. "
        "Use for: reading docs, fetching API references, researching topics, extracting structured data. "
        "Ops: "
        "scrape (single URL → markdown) | "
        "crawl (whole site → list of markdown docs) | "
        "search (web search with full content, not just snippets) | "
        "extract (structured JSON extraction from a URL)"
    ),
    "params": [
        {"name": "op",           "type": "str",  "required": True,
         "description": "scrape | crawl | search | extract"},
        {"name": "url",          "type": "str",  "required": False,
         "description": "Target URL — required for scrape, crawl, extract"},
        {"name": "query",        "type": "str",  "required": False,
         "description": "Search query — required for search op"},
        {"name": "limit",        "type": "int",  "required": False,
         "description": "Max pages to crawl or search results to return (default: crawl=10, search=5)"},
        {"name": "chunk_size",   "type": "int",  "required": False,
         "description": "Chars per chunk for paginated reads (default: scrape=24000, crawl=8000, search=6000). Response header tells you the total and next chunk_index."},
        {"name": "chunk_index",  "type": "int",  "required": False,
         "description": "0-based page number for scrape op (default 0). Increment to read the next chunk of a large page."},
        {"name": "prompt",       "type": "str",  "required": False,
         "description": "Natural-language extraction prompt — for extract op"},
        {"name": "schema",       "type": "dict", "required": False,
         "description": "JSON schema for structured extraction — for extract op"},
        {"name": "poll_interval","type": "int",  "required": False,
         "description": "Polling interval in seconds for crawl op (default 30)"},
        {"name": "api_key",      "type": "str",  "required": False,
         "description": "Firecrawl API key (overrides FIRECRAWL_API_KEY env var)"},
    ],
    "fn": run,
    "needs_client": False,   # uses Firecrawl, not the LLM client
}
