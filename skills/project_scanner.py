"""
skills/project_scanner.py
Scans the entire project, reads every source file, uses the LLM to determine
what each file does and which other files it imports/depends on, then builds
a full dependency graph and writes it to structure.md.

This structure.md is injected as context whenever the agent plans edits,
so it knows that touching file A will also affect file B.

Output format in structure.md:
  ## <relative/path/to/file.py>
  **Purpose:** one-sentence description
  **Imports:** list of project-internal files this file depends on
  **Imported by:** list of project files that import this one
  **Key exports:** main classes / functions / variables exposed
  ---
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from pathlib import Path
from config import MODEL, PROJECT_ROOT, TOKENS_PROJECT_SCANNER, TEMPERATURE_SKILLS

_LOG = logging.getLogger("skill.project_scanner")


# ── API rate limiter ───────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Sliding-window rate limiter for LLM API calls.

    Tracks the timestamp of every call made. Before each call, if the number
    of calls in the last `window_seconds` has reached `max_calls`, it sleeps
    until the oldest call falls outside the window — then proceeds.

    Shows a live countdown in the terminal so the user knows what's happening.

    Args:
        max_calls:       max calls allowed per window (default: 10)
        window_seconds:  rolling window size in seconds (default: 60)
    """

    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0):
        self.max_calls      = max_calls
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def wait(self, console) -> None:
        """
        Block until a call slot is available. Shows a countdown if waiting.
        Call this immediately before each LLM request.
        """
        now = time.monotonic()

        # Evict timestamps older than the window
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) < self.max_calls:
            # Slot available — record and proceed
            self._timestamps.append(now)
            return

        # Slot full — wait until the oldest timestamp leaves the window
        oldest   = self._timestamps[0]
        wait_for = (oldest + self.window_seconds) - now

        _LOG.info(f"[RATE LIMIT] {self.max_calls} calls in {self.window_seconds}s — waiting {wait_for:.1f}s")

        # Live countdown display
        deadline = time.monotonic() + wait_for
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            m, s = divmod(int(remaining) + 1, 60)
            console.print(
                f"  [bold yellow]⏳[/]  [dim]API rate limit ({self.max_calls} calls/{int(self.window_seconds)}s) — "
                f"cooldown [bold white]{m:02d}:{s:02d}[/][/]",
                end="\r",
            )
            time.sleep(min(1.0, remaining))

        # Clear the countdown line
        console.print(" " * 72, end="\r")

        # Slot now available
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        self._timestamps.append(now)

# ── File extensions we care about ─────────────────────────────────────────────
_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".scss", ".sass",
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".env.example",
    ".sh", ".dockerfile",
}

# Media/binary files — catalogued by path only, never sent to the LLM for analysis
_MEDIA_EXTS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Audio / video
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".avi", ".mov",
    # Documents / data blobs
    ".pdf", ".zip", ".tar", ".gz", ".sqlite", ".db",
}

# ── Directories to never recurse into ─────────────────────────────────────────
_SKIP_DIRS: set[str] = {
    ".venv", "venv", "env", ".env",
    "virtualenv", ".virtualenv",
    "pyenv", ".pyenv",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".eggs", "*.egg-info",
    "dist", "build", "sdist", "wheels",
    "site-packages",
    "migrations",
    "staticfiles", "static_root",
    "media",
    "collected_static",
    "node_modules",
    ".npm", ".yarn", ".pnp",
    ".next", ".nuxt", ".svelte-kit",
    "out", ".output",
    ".parcel-cache", ".turbo",
    "coverage",
    ".git", ".svn", ".hg",
    ".idea", ".vscode", ".vs",
    ".fleet", ".cursor",
    ".DS_Store", "__MACOSX",
    "Thumbs.db",
    ".docker",
    "logs", "log",
    "tmp", "temp", ".tmp",
    "vendor",
    "bower_components",
    ".sass-cache",
    "htmlcov",
    ".hypothesis",
}

_SKIP_PREFIXES: tuple[str, ...] = (
    "venv", "env", ".env",
    ".venv", "virtualenv",
)

_MAX_SINGLE_FILE_BYTES = 200_000
_MAX_FILE_CHARS = 6000
_MAX_FILES = 80


def _safe_extract(resp, default="") -> str:
    if not resp or not resp.choices:
        return default
    content = resp.choices[0].message.content
    if content is None:
        return default
    return content.strip() if isinstance(content, str) else str(content).strip()


def _should_skip_dir(name: str) -> bool:
    if name in _SKIP_DIRS:
        return True
    lower = name.lower()
    if any(lower.startswith(p) for p in _SKIP_PREFIXES):
        return True
    if name.startswith(".") and name not in {".github", ".husky"}:
        return True
    return False


def _collect_files(root: str) -> list[Path]:
    root_path = Path(root)
    collected = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        for fname in filenames:
            if fname.startswith(".") and fname not in {
                ".env.example", ".gitignore", ".editorconfig",
            }:
                continue

            fpath = Path(dirpath) / fname

            is_named_special = fname in {
                "Dockerfile", "Makefile", "manage.py",
                ".env.example", ".gitignore",
            }
            if not is_named_special and fpath.suffix.lower() not in _CODE_EXTS:
                continue

            try:
                if fpath.stat().st_size > _MAX_SINGLE_FILE_BYTES:
                    continue
            except OSError:
                continue

            stem = fpath.stem.lower()
            if any(stem.endswith(s) for s in (".min", ".bundle", ".chunk", "-lock")):
                continue
            if fname in {
                "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                "poetry.lock", "Pipfile.lock",
            }:
                continue

            collected.append(fpath)

    return sorted(collected)[:_MAX_FILES]


def _collect_media(root: str) -> list[Path]:
    """
    Walk the project tree and return media/binary files.
    These are catalogued by path only — never read or sent to the LLM.
    """
    root_path = Path(root)
    collected = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in _MEDIA_EXTS:
                collected.append(fpath)

    return sorted(collected)


def _read_truncated(fpath: Path) -> str:
    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + f"\n\n# [... truncated at {_MAX_FILE_CHARS} chars]"
        return text
    except Exception as e:
        return f"# ERROR reading file: {e}"


def _analyze_file(fpath: Path, content: str, root: str, client, emit, console, n_str: str, total: int, provider: str = "", rate_limiter: "_RateLimiter | None" = None) -> dict:
    """
    Stream the LLM analysis of a single file, printing tokens live as they
    arrive so the user sees activity immediately instead of waiting for the
    full response before anything appears.
    """
    from rich.text import Text

    rel = str(fpath.relative_to(root))

    prompt = f"""\
You are analyzing a source file inside a software project.
File: {rel}

Content:
```
{content}
```

Return ONLY valid JSON (no prose, no markdown fences):
{{
  "purpose": "<one concise sentence: what this file does>",
  "key_exports": ["<class/function/var name>", ...],
  "imports": ["<relative/path/in/project>", ...],
  "notes": "<optional: anything important about this file's role>"
}}

For "imports": list only OTHER FILES IN THIS PROJECT that this file imports/requires/includes.
Use relative paths from the project root. Do NOT list stdlib or third-party packages.
If none, return an empty list.
"""
    t0 = time.monotonic()

    # Print the file header line — stays on its own line (no \r trick)
    console.print(
        f"  [dim cyan]⠿[/]  [dim][{n_str}/{total}][/]  [white]{rel}[/]"
    )
    # Print the streaming prefix on the same indentation, then stream inline
    console.print("       [dim]↳ [/]", end="")

    # Honour rate limit before making the LLM call
    if rate_limiter is not None:
        rate_limiter.wait(console)

    chunks: list[str] = []
    try:
        from agent.thinking import build_thinking_kwargs, extract_chunk, strip_thought_tags
        thinking_kwargs = build_thinking_kwargs(provider, MODEL)

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=TOKENS_PROJECT_SCANNER,
            temperature=TEMPERATURE_SKILLS,
            stream=True,
            **thinking_kwargs,
        )
        for chunk in stream:
            content, reasoning = extract_chunk(chunk)
            if reasoning:
                # Show reasoning dimmed — user can see the model thinking per-file
                console.print(f"[dim]{reasoning}[/]", end="")
            if content:
                chunks.append(content)
                console.print(f"[dim]{content}[/]", end="")

        # Move to a new line after the stream finishes
        console.print()

        raw = "".join(chunks).strip()

        # Strip <thought>...</thought> blocks (some models emit chain-of-thought before JSON)
        raw = re.sub(r"<thought>.*?</thought>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # Advance to the first { in case any prose still precedes the JSON
        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]

        data = json.loads(raw)
        data["path"] = rel

        elapsed = time.monotonic() - t0
        purpose = data.get("purpose", "")
        purpose_short = purpose[:72] + "…" if len(purpose) > 72 else purpose
        console.print(
            f"  [bright_green]✔[/]  [dim]{purpose_short}[/]  [dim cyan]{elapsed:.1f}s[/]\n"
        )
        emit("ok", rel, purpose, elapsed)
        return data

    except Exception as e:
        console.print()  # close the streaming line
        elapsed = time.monotonic() - t0
        err_short = str(e)[:72]
        console.print(
            f"  [bold red]✖[/]  [dim red]ERROR: {err_short}[/]  [dim cyan]{elapsed:.1f}s[/]\n"
        )
        emit("err", rel, str(e), elapsed)
        return {
            "path": rel,
            "purpose": f"(analysis failed: {e})",
            "key_exports": [],
            "imports": [],
            "notes": "",
        }


def _build_imported_by(analyses: list[dict]) -> dict[str, list[str]]:
    imported_by: dict[str, list[str]] = {}
    for info in analyses:
        for dep in info.get("imports", []):
            if dep not in imported_by:
                imported_by[dep] = []
            imported_by[dep].append(info["path"])
    return imported_by


def _render_structure_md(
    analyses: list[dict],
    imported_by: dict[str, list[str]],
    media_files: list[Path],
    root: str,
    project_name: str,
) -> str:
    lines = [
        f"# {project_name} — Project Dependency Map",
        "",
        "> Auto-generated by the coding agent. Re-run `/scan` to refresh.",
        "",
        "## Legend",
        "- **Imports** → files this file depends on (within the project)",
        "- **Imported by** → files that depend on this one",
        "",
        "---",
        "",
    ]

    for info in analyses:
        path     = info.get("path", "?")
        purpose  = info.get("purpose", "")
        exports  = info.get("key_exports", [])
        imports  = info.get("imports", [])
        notes    = info.get("notes", "")
        rev_deps = imported_by.get(path, [])

        lines.append(f"## `{path}`")
        lines.append(f"**Purpose:** {purpose}")

        if exports:
            lines.append(f"**Key exports:** {', '.join(f'`{e}`' for e in exports)}")

        if imports:
            lines.append(f"**Imports:** {', '.join(f'`{i}`' for i in imports)}")
        else:
            lines.append("**Imports:** *(none within project)*")

        if rev_deps:
            lines.append(f"**Imported by:** {', '.join(f'`{d}`' for d in rev_deps)}")
        else:
            lines.append("**Imported by:** *(nothing — leaf file or entry point)*")

        if notes:
            lines.append(f"**Notes:** {notes}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Media / binary asset catalogue ────────────────────────────────────────
    if media_files:
        lines.append("## 🗂 Media & Binary Assets")
        lines.append("> Not analysed — catalogued by path only for reference.")
        lines.append("")
        root_path = Path(root)
        for mf in media_files:
            try:
                rel  = str(mf.relative_to(root_path))
                size = mf.stat().st_size
                size_str = (
                    f"{size / 1024:.1f} KB" if size < 1_048_576
                    else f"{size / 1_048_576:.1f} MB"
                )
            except Exception:
                rel, size_str = str(mf), "?"
            lines.append(f"- `{rel}`  ({size_str})")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def run(params: dict, client) -> str:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rbox

    console = Console(highlight=False)

    root         = params.get("root", PROJECT_ROOT)
    output_path  = params.get("output", os.path.join(root, "structure.md"))
    project_name = params.get("name", Path(root).name)
    provider     = params.get("provider", "")

    # Rate limiting — configurable via params or falls back to config default
    # Google Gemini free tier: 10 RPM. Set rpm=0 to disable.
    from config import SCANNER_RATE_LIMIT_RPM
    rpm = int(params.get("rpm", SCANNER_RATE_LIMIT_RPM))
    rate_limiter = _RateLimiter(max_calls=rpm, window_seconds=60.0) if rpm > 0 else None

    if not os.path.isdir(root):
        return f"ERROR: project root '{root}' does not exist."

    # ── Discovery phase ───────────────────────────────────────────────────────
    console.print(f"\n  [dim cyan]⊕[/]  [dim]Collecting files in[/] [white]{root}[/] …")
    files       = _collect_files(root)
    media_files = _collect_media(root)

    if not files and not media_files:
        return f"ERROR: no source files found in '{root}'."

    total = len(files)
    console.print(
        f"  [dim]  Found [bold white]{total}[/] source files"
        f"{f' + [bold white]{len(media_files)}[/] media assets' if media_files else ''}"
        f" to process.[/]\n"
    )

    # ── Per-file log table (printed as we go) ─────────────────────────────────
    #   We print one line per file as it completes so the user sees live progress.

    results: list[dict] = []
    pad = len(str(total))

    def emit(status: str, rel: str, detail: str, elapsed: float) -> None:
        idx   = len(results)            # called just before append
        n_str = str(idx + 1).rjust(pad)
        if status == "ok":
            icon  = "[bright_green]✔[/]"
            color = "dim white"
        else:
            icon  = "[bold red]✖[/]"
            color = "dim red"
            detail = f"ERROR: {detail}"

        # Truncate purpose/error so line stays tidy
        detail_short = detail[:72] + "…" if len(detail) > 72 else detail
        elapsed_str  = f"{elapsed:.1f}s"

        console.print(
            f"  {icon}  [{color}][{n_str}/{total}][/]"
            f"  [white]{rel}[/]"
            f"  [dim]{detail_short}[/]"
            f"  [dim cyan]{elapsed_str}[/]"
        )
        _LOG.info(f"[SCAN] {status.upper()} {rel} ({elapsed:.2f}s) — {detail[:120]}")

    # ── Analyse each file (blocking, sequential) ──────────────────────────────
    t_scan_start = time.monotonic()

    for fpath in files:
        content = _read_truncated(fpath)
        n_str   = str(len(results) + 1).rjust(pad)
        info    = _analyze_file(fpath, content, root, client, emit, console, n_str, total, provider, rate_limiter)
        results.append(info)

    total_elapsed = time.monotonic() - t_scan_start

    # ── Build graph & write output ─────────────────────────────────────────────
    console.print(f"\n  [dim]Building dependency graph…[/]")
    imported_by = _build_imported_by(results)
    md          = _render_structure_md(results, imported_by, media_files, root, project_name)

    try:
        Path(output_path).write_text(md, encoding="utf-8")
    except Exception as e:
        return f"ERROR writing structure.md: {e}"

    failed = [r for r in results if r["purpose"].startswith("(analysis failed")]
    ok     = len(results) - len(failed)

    console.print(
        f"\n  [bright_green]◆[/]  Scan complete — "
        f"[white]{ok}[/] analysed, [{'bold red' if failed else 'dim'}]{len(failed)}[/] failed"
        f"{f', [white]{len(media_files)}[/] media assets catalogued' if media_files else ''}"
        f"  [dim cyan]({total_elapsed:.1f}s)[/]\n"
    )

    media_summary = (
        f"\n\nMedia assets ({len(media_files)}):\n" +
        "\n".join(f"  - {mf.relative_to(root)}" for mf in media_files)
    ) if media_files else ""

    return (
        f"✔ Scanned {len(results)} source files in '{root}' ({total_elapsed:.1f}s).\n"
        f"✔ Dependency graph written to '{output_path}'.\n"
        f"✔ {len(media_files)} media assets catalogued (paths only).\n\n"
        f"Summary:\n" +
        "\n".join(f"  - {r['path']}: {r['purpose']}" for r in results) +
        media_summary
    )


SKILL_DEF = {
    "name":        "project_scanner",
    "description": (
        "Scan the entire project, analyze each file's purpose and dependencies, "
        "build a dependency graph, and write structure.md. "
        "Run this first when starting work on an existing project."
    ),
    "params": [
        {"name": "root",   "type": "str", "required": False, "description": "Project root directory (default: PROJECT_ROOT)"},
        {"name": "output", "type": "str", "required": False, "description": "Output path for structure.md (default: <root>/structure.md)"},
        {"name": "name",   "type": "str", "required": False, "description": "Project name for the header"},
    ],
    "fn": run,
    "needs_client": True,
}
