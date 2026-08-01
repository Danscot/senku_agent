"""
agent/core.py  —  Coding Agent Brain

Workflow per request:
  1. THINK   — analyze the problem, read structure.md if present, plan steps
  2. ACT     — execute each step (tool / skill)
  3. CHECK   — verify the step result; on error → repair loop (up to 3 retries)
  4. ADVANCE — move to next planned step or synthesize final answer

Changes in this version:
  - Structured stage logging via ui.logger (stage_block, LLMCallLogger, log_stage_done)
  - Multi-provider LLM clients: each stage (think/act/respond/compress) can use
    a different provider + model, configured in config.py
  - All LLM calls use streaming (no silent non-streaming fallbacks)
  - Tool call timing logged via log_tool_start / log_tool_done
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import tools  as tool_registry
import skills as skill_registry
import mcp    as mcp_registry

from config import (
    PROJECT_ROOT,
    STAGE_THINK, STAGE_ACT, STAGE_RESPOND, STAGE_COMPRESS,
    parse_stage,
)
from memory import ConversationBuffer
from agent.correction_tracker import (
    find_correction, record_correction, describe_corrections,
)
from agent.healer import try_heal
from ui import (
    C, console,
    show_mode, show_plan, show_step_start, show_step_result,
    show_tool_call, show_error, show_response, spinner, live_task,
    show_thought, show_thinking_header, show_thinking_footer,
    show_gap_report, show_clarify,
)
from ui.logger import (
    stage_block, LLMCallLogger, log_stage_done,
    log_tool_start, log_tool_done, log_retry,
    log_warning, log_error, log_info,
)

import logging
_LOG = logging.getLogger("agent.core")


# ── Heuristic fast-path ────────────────────────────────────────────────────────

_CHAT_RE = re.compile(
    r"^("
    r"hi+|hello+|hey+|yo+|sup|howdy|"
    r"what'?s up|good (morning|afternoon|evening|night)|"
    r"thanks?|thank you|thx|ty|"
    r"bye|goodbye|see ya|"
    r"who are you|what are you|what can you do"
    r")[!?., ]*$",
    re.IGNORECASE,
)

_TASK_WORDS = {
    "file", "read", "write", "create", "save", "delete", "list",
    "run", "execute", "shell", "command", "script",
    "search", "find", "code", "build", "make", "generate",
    "install", "clone", "git", "compile", "test", "debug",
    "fix", "refactor", "review", "scan", "migrate", "deploy",
    "django", "model", "view", "template", "url", "api",
    "function", "class", "import", "error", "bug",
}


def _is_obvious_chat(text: str) -> bool:
    t = text.strip()
    if _CHAT_RE.match(t):
        return True
    words = set(t.lower().split())
    if len(t) < 25 and not (words & _TASK_WORDS):
        return True
    return False


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | list | None:
    text = text.strip()
    # Strip <thought>...</thought> blocks emitted by reasoning models before JSON
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to finding the first {...} or [...] block in the remaining text
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ── URL extractor ──────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _extract_first_url(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("URL:"):
            url = line[4:].strip()
            if url.startswith("http"):
                return url
    m = _URL_RE.search(text)
    return m.group(0).rstrip(".,)") if m else ""


# ── structure.md loader ───────────────────────────────────────────────────────

def _load_structure_md(root: str = PROJECT_ROOT) -> str:
    smd = Path(root) / "structure.md"
    if not smd.exists():
        return ""
    try:
        text = smd.read_text(encoding="utf-8")
        if len(text) > 5000:
            text = text[:5000] + "\n\n[... structure.md truncated — run /scan to regenerate]"
        return text
    except Exception:
        return ""


# ── Tool catalogue ─────────────────────────────────────────────────────────────

def _tool_catalogue() -> str:
    parts = ["TOOLS (call with tool:<n>):\n" + tool_registry.describe_all()]
    skills = skill_registry.describe_all()
    if skills:
        parts.append("SKILLS (call with skill:<n>):\n" + skills)
    mcps = mcp_registry.describe_all()
    if mcps and "(no MCP" not in mcps:
        parts.append("MCP SERVERS (call with mcp:<n>):\n" + mcps)
    return "\n\n".join(parts)


# ── Thought emitter ────────────────────────────────────────────────────────────

def _emit_thoughts(thoughts: list[dict]) -> None:
    if not thoughts:
        return
    show_thinking_header()
    for t in thoughts:
        phase = t.get("phase", "reason")
        text  = t.get("text",  "")
        if text:
            show_thought(phase, text)
    show_thinking_footer()


# ── Think system prompt ────────────────────────────────────────────────────────

_THINK_SYSTEM = """\
You are a coding agent's reasoning core. You build things with the right tool for the job.

DEFAULT STACK (when the user hasn't specified):
  - Websites, landing pages, simple sites → plain HTML + CSS + JavaScript
  - Only reach for Django/React/Vue when the user EXPLICITLY mentions it,
    OR when the task clearly requires a backend (auth, database, API, forms that save data).
  - "Make me a website" = HTML/CSS/JS. Full stop. Do NOT scaffold Django.
  - "Make me a landing page" = single HTML file + CSS. No framework.
  - "Make me a web app with login" = ask first (clarify mode).

Your job: In ONE JSON response, analyze the coding request, decide execution mode, and produce the full plan.

PROJECT ROOT: {project_root}
All file paths in tool params MUST be absolute paths under the project root above, OR relative paths \
that will be resolved relative to it. Shell commands run with cwd set to this root automatically.
When creating a new project, use this root as the target directory unless the user specifies otherwise.

{catalogue}

{structure_section}

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "thoughts": [
    {{"phase": "observe",  "text": "<what is the user asking for?>"}},
    {{"phase": "reason",   "text": "<what files/components are involved?>"}},
    {{"phase": "reason",   "text": "<which tools/skills are needed?>"}},
    {{"phase": "decide",   "text": "<mode chosen and why>"}}
  ],

  "mode": "chat" | "clarify" | "single" | "plan" | "gap",

  // if mode == "clarify"
  "questions": [
    {{"id": 1, "question": "<concise question>", "options": ["<opt A>", "<opt B>", "..."]}}
  ],
  "clarify_reason": "<one sentence: what ambiguity blocks you from acting>",

  // if mode == "single"
  "tool":   "tool:<n> | skill:<n> | mcp:<n>",
  "params": {{}},

  // if mode == "plan"
  "goal": "<one sentence goal>",
  "steps": [
    {{
      "id": 1,
      "tool": "tool:<n> | skill:<n> | mcp:<n>",
      "description": "<what this step does and why>",
      "params": {{}},
      "check": "<how to verify this step succeeded>"
    }}
  ],

  // if mode == "gap"
  "gap_report": {{
    "task_summary": "<one sentence>",
    "requires": [{{"need": "<capability>", "covered_by": "<tool or null>"}}],
    "gaps": [{{"need": "<missing>", "suggested_tool": "<real library>", "reason": "<why>"}}],
    "partial": false
  }}
}}

MODE RULES:
  chat    — conversation, general questions — NO tools needed
  clarify — STOP and ask the user before acting; use when the answer would change
            WHAT you build or HOW you build it significantly
  single  — exactly ONE tool/skill call completes the task
  plan    — multiple sequential steps needed (max 8 steps)
  gap     — task needs capabilities NOT in the tool set

CLARIFY RULES — when to ask vs when to just act:
  ASK when genuinely ambiguous and the wrong guess wastes significant work:
    ✔ "website" with backend signals (login, database, forms) → ask: plain HTML or Django/React?
    ✔ "add auth" → ask: session-based, JWT, or OAuth?
    ✔ "deploy" → ask: local, VPS, Docker, or cloud?
  DO NOT ASK when the default is obvious:
    ✗ "make me a landing page" → just build HTML/CSS/JS, no question needed
    ✗ "make me a website" → HTML/CSS/JS is the default, build it
    ✗ File names you can choose sensibly yourself
    ✗ Minor style details (pick a reasonable default)
    ✗ Anything obvious from context or prior conversation
  MAX 2 questions per clarify response. Each question MUST have 2-4 short options.
  Prefer acting with a sensible default over asking about trivial details.

TOOL KEY FORMAT — CRITICAL — READ CAREFULLY:
  The "tool" field value must be ONLY "tool:<name>" or "skill:<name>". Params go in "params".
  ✔ CORRECT:  "tool": "tool:django",  "params": {{"op": "scaffold", "name": "demo_site"}}
  ✔ CORRECT:  "tool": "tool:django",  "params": {{"op": "startapp", "name": "kaizen_app"}}
  ✔ CORRECT:  "tool": "skill:code_writer", "params": {{"mode": "create", "path": "...", "instruction": "..."}}
  ✗ WRONG:    "tool": "django:op=scaffold"   — NEVER embed params in the tool key
  ✗ WRONG:    "tool": "django:scaffold"      — NEVER put op values in the key
  ✗ WRONG:    "tool": "tool:django:scaffold" — NEVER use multiple colons

DJANGO TOOL OPERATIONS (op param values):
  scaffold     — create a new Django project: params {{"op":"scaffold","name":"<project_name>"}}
  startapp     — create a new app:            params {{"op":"startapp","name":"<app_name>"}}
  migrate      — run migrations:              params {{"op":"migrate"}}
  makemigrations — make migrations:           params {{"op":"makemigrations"}}
  check        — django system check:         params {{"op":"check"}}
  install      — pip install django+deps:     params {{"op":"install"}}

CODING-SPECIFIC RULES:
  - ALWAYS check structure.md before editing files
  - For new Django projects: scaffold first, then startapp, then migrate
  - For frontend changes: check if there's a base template first
  - For bug fixes: review the file first (skill:code_reviewer), then edit (skill:code_writer)
  - After editing Python files: run tests if a test file exists (skill:test_runner)
  - Prefer skill:code_writer over tool:file for writing code

ASK TOOL — mid-execution questions:
  Use tool:ask as a PLAN STEP when you hit a genuine decision point mid-task:
    ✔ You find conflicting config values and don't know which to keep
    ✔ A file you expected to exist is missing and the path is ambiguous
    ✔ The user asked to "add auth" but the codebase has two auth patterns already
  Insert tool:ask EARLY in the plan (before the steps that need the answer).
  Use {{step_N}} to pass the user's answer into downstream step params.
  DO NOT use tool:ask for decisions you can make yourself with a sensible default.

STEP PARAM PLACEHOLDERS:
  {{step_N}}       — full text result of step N (truncated to 1500 chars)
  {{step_N.url}}   — first URL in step N's result
  {{step_N.line1}} — first non-empty line of step N's result
"""


# ── Multi-provider client factory ──────────────────────────────────────────────

def _make_client(stage_str: str) -> tuple[OpenAI, str, str]:
    """
    Return (client, provider, model) for a given stage string.
    All providers are accessed through an OpenAI-compatible client.
    """
    provider, model, base_url, api_key = parse_stage(stage_str)
    if not api_key:
        log_warning(f"No API key configured for provider '{provider}' — calls may fail.")
        api_key = "placeholder"
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, provider, model


# ── Structure.md auto-update helpers ──────────────────────────────────────────

# Tools/skills that write files to disk — trigger a structure.md refresh
_FILE_WRITING_TOOLS = {
    "tool:file", "skill:code_writer", "skill:code_reviewer",
    "tool:django", "tool:shell",
}

def _step_writes_files(tool_key: str, params: dict) -> bool:
    """Return True if this step likely wrote new files to disk."""
    key = tool_key.strip()

    # skill:code_writer always writes
    if key == "skill:code_writer":
        return True

    # tool:file with op=write/create
    if key == "tool:file":
        return params.get("op", "read") in ("write", "create", "append")

    # tool:shell — heuristic: cmd contains redirection or file-creating commands
    if key == "tool:shell":
        cmd = params.get("cmd", "")
        return any(tok in cmd for tok in (">", "touch", "mkdir", "cp ", "mv ", "tee "))

    # tool:django scaffold / startapp always creates directories and files
    if key == "tool:django":
        return params.get("op", "") in ("scaffold", "startapp")

    return False


def _update_structure_md(root: str) -> None:
    """
    Lightweight structure.md updater that just appends newly discovered paths
    without re-running the full LLM analysis scan. Reads current structure.md,
    walks the directory, and adds entries for files not yet listed.
    """
    from pathlib import Path as _Path

    struct_path = _Path(root) / "structure.md"
    existing_md = struct_path.read_text(encoding="utf-8") if struct_path.exists() else ""

    # Collect all current source + media files
    code_exts = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
        ".json", ".yaml", ".yml", ".toml", ".md", ".sh",
    }
    media_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
        ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".pdf", ".sqlite", ".db",
    }
    skip_dirs = {
        ".venv", "venv", "env", "__pycache__", ".git", "node_modules",
        "migrations", "staticfiles", "media", "dist", "build",
    }

    new_code:  list[str] = []
    new_media: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            fpath  = Path(dirpath) / fname
            try:
                rel = str(fpath.relative_to(root))
            except ValueError:
                continue

            # Skip if already in structure.md
            if f"`{rel}`" in existing_md:
                continue

            ext = fpath.suffix.lower()
            if ext in code_exts:
                new_code.append(rel)
            elif ext in media_exts:
                new_media.append(rel)

    if not new_code and not new_media:
        return  # nothing new to add

    additions: list[str] = []
    for rel in sorted(new_code):
        additions.append(f"## `{rel}`")
        additions.append("**Purpose:** *(newly created — run /scan for full analysis)*")
        additions.append("**Imports:** *(unknown)*")
        additions.append("**Imported by:** *(unknown)*")
        additions.append("")
        additions.append("---")
        additions.append("")

    if new_media:
        # Find the existing media section or append a new one
        media_header = "## 🗂 Media & Binary Assets"
        media_lines  = [f"- `{rel}`" for rel in sorted(new_media)]
        if media_header in existing_md:
            # Insert after the header line
            existing_md = existing_md.replace(
                media_header + "\n",
                media_header + "\n" + "\n".join(media_lines) + "\n",
            )
            additions_text = "\n".join(additions)
            struct_path.write_text(
                existing_md.rstrip() + ("\n\n" + additions_text if additions_text else ""),
                encoding="utf-8",
            )
            return
        else:
            additions.append(media_header)
            additions.append("> Not analysed — catalogued by path only.")
            additions.append("")
            additions.extend(media_lines)
            additions.append("")

    struct_path.write_text(
        existing_md.rstrip() + "\n\n" + "\n".join(additions),
        encoding="utf-8",
    )


# ── Agent ──────────────────────────────────────────────────────────────────────

class Agent:

    def __init__(self, client: OpenAI, project_root: str = PROJECT_ROOT):
        # primary client kept for backward compat (used by skills that need_client)
        self.client       = client
        self.project_root = project_root
        self.memory       = ConversationBuffer(max_turns=40, keep_recent=10)
        self.catalogue    = _tool_catalogue()

        # Per-stage clients
        self._think_client,    self._think_provider,    self._think_model    = _make_client(STAGE_THINK)
        self._act_client,      self._act_provider,      self._act_model      = _make_client(STAGE_ACT)
        self._respond_client,  self._respond_provider,  self._respond_model  = _make_client(STAGE_RESPOND)
        self._compress_client, self._compress_provider, self._compress_model = _make_client(STAGE_COMPRESS)

        self._rebuild_think_system()

    def _rebuild_think_system(self) -> None:
        corrections_block = describe_corrections()
        extra = f"\n\n{corrections_block}" if corrections_block else ""
        structure_md = _load_structure_md(self.project_root)
        structure_section = (
            f"PROJECT STRUCTURE (structure.md):\n{structure_md}"
            if structure_md
            else "PROJECT STRUCTURE: No structure.md found. Run skill:project_scanner to generate one."
        )
        self._think_system = _THINK_SYSTEM.format(
            project_root=self.project_root,
            catalogue=self.catalogue + extra,
            structure_section=structure_section,
        )

    # ── Core streaming LLM call ────────────────────────────────────────────────

    def _stream(
        self,
        messages: list[dict],
        *,
        stage: str,
        client: OpenAI,
        provider: str,
        model: str,
        max_tokens: int = 12000,
        attempt: int = 1,
        max_attempts: int = 1,
        echo: bool = False,          # if True: print tokens live to console
    ) -> str:
        """
        Universal streaming LLM call with structured logging.
        All internal + user-visible calls go through here.
        """
        with LLMCallLogger(
            stage,
            provider=provider,
            model=model,
            messages=messages,
            attempt=attempt,
            max_attempts=max_attempts,
        ) as lcl:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=True,
                )

                if echo:
                    console.print(f"\n  [{C['accent']}]◆  Agent[/]  ", end="")

                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        lcl.on_chunk(delta)
                        if echo:
                            console.print(f"[{C['agent']}]{delta}[/]", end="", highlight=False)

                if echo:
                    console.print("\n")

                return lcl.finish()

            except Exception as exc:
                return lcl.finish(error=str(exc))

    # ── Stage-specific call wrappers ───────────────────────────────────────────

    def _llm_think(self, messages: list[dict], max_tokens: int = 11000, attempt: int = 1) -> str:
        return self._stream(
            messages, stage="think",
            client=self._think_client, provider=self._think_provider, model=self._think_model,
            max_tokens=max_tokens, attempt=attempt, max_attempts=3,
        )

    def _llm_act(self, messages: list[dict], max_tokens: int = 3000, attempt: int = 1) -> str:
        return self._stream(
            messages, stage="act",
            client=self._act_client, provider=self._act_provider, model=self._act_model,
            max_tokens=max_tokens, attempt=attempt, max_attempts=3,
        )

    def _llm_respond(self, messages: list[dict], max_tokens: int = 12000) -> str:
        return self._stream(
            messages, stage="respond",
            client=self._respond_client, provider=self._respond_provider, model=self._respond_model,
            max_tokens=max_tokens, echo=True,
        )

    def _llm_compress(self, messages: list[dict], max_tokens: int = 4000) -> str:
        return self._stream(
            messages, stage="compress",
            client=self._compress_client, provider=self._compress_provider, model=self._compress_model,
            max_tokens=max_tokens,
        )

    # Kept for backward compat (skills/healer call self.client directly)
    def _llm(self, messages: list[dict], max_tokens: int = 12000) -> str:
        return self._llm_act(messages, max_tokens)

    def _llm_stream(self, messages: list[dict], max_tokens: int = 12000) -> str:
        return self._llm_respond(messages, max_tokens)

    # ── Think ───────────────────────────────────────────────────────────────────

    def _think(self, user_input: str) -> dict:
        self._rebuild_think_system()

        messages = [{"role": "system", "content": self._think_system}]
        messages += self.memory.to_messages()
        messages.append({"role": "user", "content": user_input})

        with stage_block("THINK", input_text=user_input):
            raw  = self._llm_think(messages)
            data = _extract_json(raw)

            if not isinstance(data, dict):
                log_warning("THINK stage returned non-JSON — falling back to chat mode")
                data = {"mode": "chat", "thoughts": []}

            mode  = data.get("mode", "chat")
            steps = data.get("steps", [])
            log_stage_done("think", mode=mode, steps=len(steps))

        return data

    # ── Param resolver ──────────────────────────────────────────────────────────

    def _resolve_params(self, params: dict, results: dict[int, str]) -> dict:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str):
                for step_id, result in results.items():
                    ph_url  = "{step_" + str(step_id) + ".url}"
                    ph_line = "{step_" + str(step_id) + ".line1}"
                    ph_full = "{step_" + str(step_id) + "}"
                    if ph_url  in v: v = v.replace(ph_url,  _extract_first_url(result))
                    if ph_line in v:
                        line1 = next((l.strip() for l in result.splitlines() if l.strip()), result[:200])
                        v = v.replace(ph_line, line1)
                    if ph_full in v: v = v.replace(ph_full, result[:1500])
            resolved[k] = v
        return resolved

    # ── Dispatch with repair loop ───────────────────────────────────────────────

    def _normalize_tool_key(self, tool_key: str, params: dict) -> tuple[str, dict]:
        """
        The LLM sometimes encodes params directly into the tool key, e.g.:
          "django:op=scaffold"  →  tool="tool:django", params["op"]="scaffold"
          "django:app=kaizen"   →  tool="tool:django", params["app"]="kaizen"

        This normalizer catches that pattern and promotes the embedded k=v pairs
        into the params dict before dispatch.
        """
        key = tool_key.strip()

        # Already properly prefixed (tool:/skill:/mcp:) with no = in the name
        if key.startswith(("skill:", "mcp:")):
            return key, params
        if key.startswith("tool:"):
            name = key[len("tool:"):]
        else:
            name = key  # bare name like "django" or "file"

        # Detect k=v encoding: "django:op=scaffold" → name="django", extras={"op":"scaffold"}
        # Split on the FIRST colon to get the base tool name
        if ":" in name:
            base, rest = name.split(":", 1)
            # rest might be "op=scaffold" or "op=scaffold&name=demo"
            extras: dict = {}
            for kv in rest.replace("&", ",").split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    extras[k.strip()] = v.strip()
                elif kv.strip():
                    # bare word after colon — treat as "op" value for django/shell
                    extras["op"] = kv.strip()
            if extras:
                # Merge extras into params (params from planner take precedence)
                merged = {**extras, **params}
                log_warning(
                    f"Normalized malformed tool key '{tool_key}' → 'tool:{base}' "
                    f"with injected params {extras}"
                )
                return f"tool:{base}", merged
            name = base

        return f"tool:{name}", params

    def _dispatch(self, tool_key: str, params: dict, max_retries: int = 3) -> Any:
        # Fix malformed keys like "django:op=scaffold" before anything else
        tool_key, params = self._normalize_tool_key(tool_key, params)

        t0 = log_tool_start(tool_key, params)

        for attempt in range(1, max_retries + 1):
            result = self._dispatch_raw(tool_key, params)
            is_err = isinstance(result, str) and result.startswith("ERROR")

            if not is_err:
                log_tool_done(tool_key, str(result), t0)
                return result

            # ── Stage 1: self-healing ─────────────────────────────────────────
            if attempt == 1:
                healed, _ = try_heal(str(result))
                if healed:
                    log_retry(tool_key, attempt, "self-heal: missing dep")
                    retry = self._dispatch_raw(tool_key, params)
                    if not (isinstance(retry, str) and retry.startswith("ERROR")):
                        log_tool_done(tool_key, str(retry), t0)
                        return retry
                    result = retry

            # ── Stage 2: known correction ─────────────────────────────────────
            known = find_correction(tool_key, params, str(result))
            if known:
                fix_params = known["fix_params"]
                log_retry(tool_key, attempt, f"known correction: {known['explanation']}")
                retry = self._dispatch_raw(tool_key, fix_params)
                if not (isinstance(retry, str) and retry.startswith("ERROR")):
                    record_correction(tool_key, params, fix_params, str(result), known["explanation"])
                    self._rebuild_think_system()
                    log_tool_done(tool_key, str(retry), t0)
                    return retry
                params = fix_params

            # ── Stage 3: LLM-derived fix ──────────────────────────────────────
            fix = self._ask_llm_fix(tool_key, params, str(result))
            if fix:
                fix_params  = fix.get("params", params)
                explanation = fix.get("explanation", "LLM-derived fix")
                log_retry(tool_key, attempt, f"LLM fix: {explanation}")
                retry = self._dispatch_raw(tool_key, fix_params)
                if not (isinstance(retry, str) and retry.startswith("ERROR")):
                    record_correction(tool_key, params, fix_params, str(result), explanation)
                    self._rebuild_think_system()
                    log_tool_done(tool_key, str(retry), t0)
                    return retry
                params = fix_params

        log_tool_done(tool_key, str(result), t0, error=True)
        return result

    def _inject_project_root(self, params: dict) -> dict:
        """
        Inject project_root into params so tools/skills always operate
        in the correct project directory rather than the agent's own cwd.

        Rules:
          - shell/git tools: set 'cwd' if not already provided
          - file/code tools:  set 'root' if not already provided
          - all tools:        always inject 'project_root' as a hint
        """
        p = dict(params)
        root = self.project_root
        p.setdefault("project_root", root)
        # shell + git tools use 'cwd' for subprocess working directory
        if "cwd" not in p:
            p["cwd"] = root
        # code_writer, project_scanner, etc. use 'root' for structure.md lookup
        if "root" not in p:
            p["root"] = root
        return p

    def _dispatch_raw(self, tool_key: str, params: dict) -> Any:
        # Skills that do their own rich streaming output must not be wrapped
        # in live_task — the Live context would fight with their console prints.
        _NO_LIVE_TIMER = {"project_scanner", "code_writer"}
        skill_name = tool_key[len("skill:"):] if tool_key.startswith("skill:") else ""
        use_timer  = skill_name not in _NO_LIVE_TIMER

        label = tool_key.replace("tool:", "").replace("skill:", "").replace("mcp:", "")

        def _run():
            try:
                params_injected = self._inject_project_root(params)

                if tool_key.startswith("skill:"):
                    name = tool_key[len("skill:"):]
                    defn = skill_registry.get(name)
                    if not defn:
                        return f"ERROR: skill '{name}' not found."
                    fn = defn["fn"]
                    return fn(params_injected, self.client) if defn.get("needs_client") else fn(params_injected)

                if tool_key.startswith("mcp:"):
                    name = tool_key[len("mcp:"):]
                    defn = mcp_registry.get(name)
                    if not defn:
                        return f"ERROR: mcp server '{name}' not enabled."
                    return defn["fn"](params_injected)

                name = tool_key.replace("tool:", "").strip()
                if name in ("chat", "think", "reason", "none", ""):
                    return (
                        f"ERROR: '{tool_key}' is not a real tool. "
                        f"Available: {', '.join(tool_registry.REGISTRY.keys())}"
                    )
                defn = tool_registry.get(name)
                if not defn:
                    available = ", ".join(tool_registry.REGISTRY.keys())
                    return f"ERROR: tool '{name}' not found. Available: {available}"
                return defn["fn"](params_injected)

            except Exception as e:
                return f"ERROR: exception in {tool_key} — {e}"

        if use_timer:
            with live_task(label):
                return _run()
        else:
            return _run()

    def _ask_llm_fix(self, tool_key: str, params: dict, error: str) -> dict | None:
        prompt = f"""A tool call failed. Suggest corrected parameters.

Project root: {self.project_root}
All file paths must be absolute or relative to the project root above.

Tool: {tool_key}
Original params: {json.dumps(params)}
Error: {error[:400]}

Available tools and their params:
{_tool_catalogue()}

Return ONLY JSON:
{{
  "params": {{...corrected params...}},
  "explanation": "<one sentence: what was wrong and what you changed>"
}}

If you cannot determine a fix, return: {{"params": null, "explanation": "cannot fix"}}
"""
        with stage_block("ACT"):
            raw  = self._llm_act([{"role": "user", "content": prompt}])
            data = _extract_json(raw)
            log_stage_done("act", result="fix_found" if (isinstance(data, dict) and data.get("params")) else "no_fix")
        if isinstance(data, dict) and data.get("params"):
            return data
        return None

    # ── CHECK phase ───────────────────────────────────────────────────────────

    def _check_step(self, step: dict, result: str) -> bool:
        """
        A step passes if its result does not start with ERROR.
        We deliberately avoid asking the LLM to judge success — it's unreliable
        because it can flag a step as failed just because a *prior* step failed,
        not because the current tool actually returned bad output.
        The result string itself is ground truth: tools always prefix errors with ERROR:.
        """
        return not (isinstance(result, str) and result.strip().startswith("ERROR"))

    # ── Mode handlers ──────────────────────────────────────────────────────────

    def _handle_chat(self, user_input: str) -> str:
        structure_md = _load_structure_md(self.project_root)
        system = (
            "You are Senku, a coding agent that uses the right tool for the job. "
            "Default to plain HTML/CSS/JS unless the user explicitly asks for a framework. "
            "Be concise, technically precise, and occasionally reference your scientific reasoning.\n\n"
            f"Project root: {self.project_root}\n"
            + (f"Project structure:\n{structure_md}" if structure_md else "No structure.md yet.")
        )
        messages = [{"role": "system", "content": system}]
        messages += self.memory.to_messages()
        messages.append({"role": "user", "content": user_input})

        with stage_block("RESPOND", input_text=user_input):
            result = self._llm_respond(messages, max_tokens=6000)
            log_stage_done("respond", chars=len(result))
        return result

    def _handle_single(self, user_input: str, think: dict) -> str:
        tool_key = think.get("tool", "")
        params   = think.get("params", {})

        if not tool_key:
            return self._handle_chat(user_input)

        show_tool_call(tool_key, params)
        result = self._dispatch(tool_key, params)
        is_err = isinstance(result, str) and result.startswith("ERROR")
        show_step_result(str(result), error=is_err)

        with stage_block("RESPOND"):
            answer = self._llm_respond([
                {"role": "system",    "content": "You are a coding agent. Answer naturally using the tool result. Be concise and technical."},
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": f"Tool result:\n{str(result)[:2000]}"},
                {"role": "user",      "content": "Give your final answer."},
            ], max_tokens=5000)
            log_stage_done("respond", chars=len(answer))
        return answer

    def _handle_plan(self, user_input: str, think: dict) -> str:
        steps = think.get("steps", [])
        if not steps:
            log_warning("Planner produced no steps — falling back to chat.")
            return self._handle_chat(user_input)

        show_plan(think)

        step_results: dict[int, str] = {}
        failed_steps: list[str]      = []

        for step in steps:
            show_step_start(step)
            params   = self._resolve_params(step.get("params", {}), step_results)
            tool_key = step["tool"]

            show_tool_call(tool_key, params)
            result = self._dispatch(tool_key, params)
            is_err = isinstance(result, str) and result.startswith("ERROR")
            show_step_result(str(result), error=is_err)

            step_ok = self._check_step(step, str(result))
            if not step_ok:
                failed_steps.append(
                    f"Step {step['id']} ({step.get('description', tool_key)}): {str(result)[:300]}"
                )
                log_error(f"Step {step['id']} failed — continuing to synthesize")

            # ── Auto-update structure.md after any successful file/code write ──
            if step_ok and _step_writes_files(tool_key, params):
                try:
                    _update_structure_md(self.project_root)
                    self._rebuild_think_system()
                    log_info("structure.md updated — new files registered")
                except Exception as e:
                    log_warning(f"structure.md auto-update failed: {e}")

            step_results[step["id"]] = str(result)

        results_block = "\n\n".join(
            f"Step {i} ({steps[i-1].get('tool','?')}):\n{r[:800]}"
            for i, r in step_results.items()
        )

        failure_note = ""
        if failed_steps:
            failure_note = (
                "\n\nNOTE: The following steps had issues:\n" +
                "\n".join(f"  - {f}" for f in failed_steps) +
                "\nExplain what went wrong and what the user should do next."
            )

        with stage_block("RESPOND"):
            answer = self._llm_respond([
                {"role": "system",    "content": "You are a coding agent. Write a clear, complete final answer."},
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": f"Completed steps:\n{results_block}{failure_note}"},
                {"role": "user",      "content": "Give your final answer."},
            ], max_tokens=8000)
            log_stage_done("respond", chars=len(answer), failed_steps=len(failed_steps))
        return answer

    def _handle_clarify(self, think: dict) -> str:
        """
        Pause execution and surface the agent's questions to the user.
        The questions + reason are shown via the UI; the agent returns
        a prompt string so the next user message provides the answers.
        """
        questions     = think.get("questions", [])
        clarify_reason = think.get("clarify_reason", "I need a bit more info before I act.")

        show_clarify(clarify_reason, questions)

        # Build a summary so the LLM can remember what it asked
        q_lines = "\n".join(
            f"  {q['id']}. {q['question']}"
            + (f"  [{' / '.join(q['options'])}]" if q.get("options") else "")
            for q in questions
        )
        return f"Before I proceed, I have a couple of quick questions:\n{q_lines}"

    def _handle_gap(self, think: dict) -> str:
        report = think.get("gap_report", {})
        show_gap_report(report)
        if report.get("partial", False):
            log_warning("Proceeding with available tools (partial result)")
        return ""

    # ── Main entry ─────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        if self.memory.should_compress():
            _LOG.info("[COMPRESS] Compressing conversation buffer…")
            self.memory.compress(self._compress_client)

        t0 = time.monotonic()

        if _is_obvious_chat(user_input):
            show_mode("chat")
            result = self._handle_chat(user_input)
        else:
            think = self._think(user_input)
            _emit_thoughts(think.get("thoughts", []))

            mode = think.get("mode", "chat")
            show_mode(mode)

            if   mode == "clarify": result = self._handle_clarify(think)
            elif mode == "single":  result = self._handle_single(user_input, think)
            elif mode == "plan":    result = self._handle_plan(user_input, think)
            elif mode == "gap":     result = self._handle_gap(think)
            else:                   result = self._handle_chat(user_input)

        elapsed = time.monotonic() - t0
        console.print(
            f"  [{C['muted']}]⏱  {elapsed:.1f}s[/]\n",
            highlight=False,
        )
        _LOG.info(f"[RUN] total_elapsed={elapsed:.2f}s")

        if result:
            self.memory.push("user",      user_input)
            self.memory.push("assistant", result)

        return result
