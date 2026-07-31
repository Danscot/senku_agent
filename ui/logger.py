"""
ui/logger.py — Structured stage-aware logger.

Produces output like:

  ────────────────────────────────────────────────────────────
    ▶  THINK STAGE   input: 42 chars
  ────────────────────────────────────────────────────────────
  INFO     agent.core: === THINK STAGE START === input_chars=42
  INFO     agent.core: [CALL] stage=think provider=nvidia model=kimi-k2-thinking attempt=1/3 tokens_est=1200
    ⠋  Think LLM call              12:04:01  ℹ  Model: nvidia:kimi-k2-thinking | Tokens (est.): 1200 | Mode: streaming
  INFO     agent.core: [INFO] Model: nvidia:kimi-k2-thinking | Tokens (est.): 1200 | Mode: streaming
  INFO     agent.core:   ↳ Done — 842 chars in 4s
    ✔  Think LLM call              12:04:05  ℹ    ↳ Done — 842 chars in 4s
  INFO     agent.core: [OK] stage=think provider=nvidia elapsed=4.12s chars=842
    ✔  Think done — mode=plan | steps=3

All output goes to both:
  - Rich console   (coloured, human-readable)
  - Python logging (structured INFO/WARNING/ERROR — useful for file capture)
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.rule import Rule

# ── shared console (imported by display.py too) ──────────────────────────────
console = Console(stderr=False, highlight=False)

# ── colour palette (matches display.py C dict) ───────────────────────────────
_C = {
    "div":       "dim white",
    "stage_hdr": "bold white",
    "arrow":     "bold cyan",
    "label":     "cyan",
    "muted":     "dim white",
    "ok":        "bright_green",
    "warn":      "yellow",
    "error":     "bold red",
    "info":      "dim cyan",
    "kv_key":    "dim white",
    "kv_val":    "white",
    "time":      "dim white",
    "spinner":   "cyan",
    "tick":      "bright_green",
    "cross":     "bold red",
    "dot":       "dim white",
    "provider":  "magenta",
    "model":     "bold white",
    "tokens":    "yellow",
    "elapsed":   "bright_yellow",
    "chars":     "cyan",
}

# ── Python stdlib logger ──────────────────────────────────────────────────────
_LOG_FMT  = "%(levelname)-8s %(name)s: %(message)s"
_DATE_FMT = "%H:%M:%S"

def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy third-party SDK internals
    for noisy in (
        "openai",
        "openai._base_client",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

_setup_logging()


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _kv(*pairs: tuple[str, Any]) -> str:
    """Render key=value pairs as a string."""
    return "  ".join(f"[{_C['kv_key']}]{k}[/]=[{_C['kv_val']}]{v}[/]" for k, v in pairs)


def _tokens_est(messages: list[dict] | str | None) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    if messages is None:
        return 0
    if isinstance(messages, str):
        return max(1, len(messages) // 4)
    total = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, total // 4)


# ── divider ───────────────────────────────────────────────────────────────────

def _divider() -> None:
    console.print(f"[{_C['div']}]{'─' * 60}[/]")


# ── Stage context manager ─────────────────────────────────────────────────────

@contextmanager
def stage_block(
    name: str,
    *,
    logger_name: str = "agent.core",
    input_text: str = "",
) -> Generator[None, None, None]:
    """
    Context manager that prints a stage header + footer divider.

    Usage:
        with stage_block("THINK", input_text=user_input):
            ...
    """
    log = logging.getLogger(logger_name)
    input_chars = len(input_text)

    _divider()
    console.print(
        f"  [{_C['arrow']}]▶[/]  [{_C['stage_hdr']}]{name} STAGE[/]"
        + (f"   [{_C['muted']}]input: {input_chars} chars[/]" if input_chars else ""),
    )
    _divider()

    log.info(f"=== {name} STAGE START ===" + (f" input_chars={input_chars}" if input_chars else ""))

    try:
        yield
    finally:
        console.print()


# ── LLM call logger ───────────────────────────────────────────────────────────

class LLMCallLogger:
    """
    Wraps an LLM streaming call and emits structured logs + a live spinner.

    Usage:
        with LLMCallLogger("think", provider="nvidia", model="kimi-k2-thinking",
                            messages=messages, attempt=1, max_attempts=3) as lcl:
            for chunk in stream:
                lcl.on_chunk(chunk)
            result = lcl.finish()
    """

    def __init__(
        self,
        stage: str,
        *,
        provider: str,
        model: str,
        messages: list[dict] | None = None,
        attempt: int = 1,
        max_attempts: int = 1,
        logger_name: str = "agent.core",
    ):
        self.stage        = stage.lower()
        self.provider     = provider
        self.model        = model
        self.attempt      = attempt
        self.max_attempts = max_attempts
        self.log          = logging.getLogger(logger_name)
        self._tokens      = _tokens_est(messages)
        self._chunks: list[str] = []
        self._t0: float   = 0.0
        self._live: Live | None = None

    # ── private helpers ───────────────────────────────────────────────────────

    def _label(self) -> str:
        return f"{self.stage.title()} LLM call"

    def _provider_model(self) -> str:
        return f"{self.provider}:{self.model}"

    def _spinner_text(self) -> Text:
        t = Text()
        t.append(f"  {self._label():<28}", style="white")
        t.append(f"  {_now()}  ", style=_C["time"])
        t.append("ℹ  ", style=_C["info"])
        t.append(f"Model: ", style=_C["kv_key"])
        t.append(f"{self._provider_model()}", style=_C["model"])
        t.append(f" | Tokens (est.): ", style=_C["kv_key"])
        t.append(str(self._tokens), style=_C["tokens"])
        t.append(f" | Mode: streaming", style=_C["kv_key"])
        return t

    # ── public API ────────────────────────────────────────────────────────────

    def __enter__(self) -> "LLMCallLogger":
        self._t0 = time.monotonic()

        # structured log
        self.log.info(
            f"[CALL] stage={self.stage} provider={self.provider} model={self.model} "
            f"attempt={self.attempt}/{self.max_attempts} tokens_est={self._tokens}"
        )

        # rich info line (mirrors what the spinner shows)
        info_line = (
            f"  [{_C['spinner']}]⠋[/]  [{_C['label']}]{self._label():<28}[/]"
            f"  [{_C['time']}]{_now()}[/]  [{_C['info']}]ℹ  [/]"
            f"[{_C['kv_key']}]Model:[/] [{_C['model']}]{self._provider_model()}[/]"
            f" [{_C['kv_key']}]| Tokens (est.):[/] [{_C['tokens']}]{self._tokens}[/]"
            f" [{_C['kv_key']}]| Mode:[/] [{_C['kv_val']}]streaming[/]"
        )
        console.print(info_line)
        self.log.info(
            f"[INFO] Model: {self._provider_model()} | Tokens (est.): {self._tokens} | Mode: streaming"
        )

        return self

    def on_chunk(self, text: str) -> None:
        """Call for each streamed token/chunk."""
        if text:
            self._chunks.append(text)

    def finish(self, *, error: str = "") -> str:
        """
        Call when the stream is done (or failed).
        Prints the ✔/✖ line and returns the assembled text.
        """
        elapsed = time.monotonic() - self._t0
        result  = "".join(self._chunks)
        chars   = len(result)

        if error:
            # ── error path ────────────────────────────────────────────────────
            self.log.error(
                f"[ERROR] stage={self.stage} provider={self.provider} "
                f"model={self.model} elapsed={elapsed:.2f}s error={error[:120]}"
            )
            console.print(
                f"  [{_C['cross']}]✖[/]  [{_C['label']}]{self._label():<28}[/]"
                f"  [{_C['time']}]{_now()}[/]  [{_C['error']}]✖  {error[:100]}[/]"
            )
        else:
            # ── success path ──────────────────────────────────────────────────
            done_msg = f"Done — {chars} chars in {elapsed:.0f}s"
            self.log.info(f"[INFO]   ↳ {done_msg}")
            console.print(
                f"  [{_C['tick']}]✔[/]  [{_C['label']}]{self._label():<28}[/]"
                f"  [{_C['time']}]{_now()}[/]  [{_C['info']}]ℹ    ↳ {done_msg}[/]"
            )
            self.log.info(
                f"[OK] stage={self.stage} provider={self.provider} "
                f"model={self.model} elapsed={elapsed:.2f}s chars={chars}"
            )

        return result

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and exc_type is not KeyboardInterrupt:
            self.finish(error=str(exc_val))
            return True   # suppress so caller gets "" back
        return False


# ── Stage-done summary line ────────────────────────────────────────────────────

def log_stage_done(
    stage: str,
    *,
    logger_name: str = "agent.core",
    **kwargs: Any,
) -> None:
    """
    Print the compact "✔  Think done — mode=plan | steps=3" summary.

    Usage:
        log_stage_done("think", mode=think_mode, steps=len(steps))
        log_stage_done("parser", title=title, mains=3, complements=2, drinks=0)
    """
    log  = logging.getLogger(logger_name)
    kv   = " | ".join(f"{k}={v!r}" for k, v in kwargs.items())
    label = f"{stage.title()} done"

    console.print(
        f"  [{_C['tick']}]✔[/]  [{_C['ok']}]{label}[/]"
        + (f" [dim]—[/] [{_C['muted']}]{kv}[/]" if kv else "")
    )
    log.info(f"=== {stage.upper()} STAGE DONE === {kv}")


# ── Tool/step event helpers ───────────────────────────────────────────────────

def log_tool_start(tool_key: str, params: dict, *, logger_name: str = "agent.core") -> float:
    """Log the start of a tool call. Returns t0."""
    log = logging.getLogger(logger_name)
    safe = {k: (str(v)[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
            for k, v in params.items()}
    log.info(f"[TOOL] {tool_key}  params={safe}")
    return time.monotonic()


def log_tool_done(
    tool_key: str,
    result: str,
    t0: float,
    *,
    error: bool = False,
    logger_name: str = "agent.core",
) -> None:
    """Log the result of a tool call."""
    log     = logging.getLogger(logger_name)
    elapsed = time.monotonic() - t0
    chars   = len(result)
    if error:
        log.error(f"[TOOL-ERR] {tool_key} elapsed={elapsed:.2f}s | {result[:200]}")
    else:
        log.info(f"[TOOL-OK]  {tool_key} elapsed={elapsed:.2f}s chars={chars}")


def log_retry(tool_key: str, attempt: int, reason: str, *, logger_name: str = "agent.core") -> None:
    log = logging.getLogger(logger_name)
    log.warning(f"[RETRY] {tool_key} attempt={attempt} reason={reason[:120]}")
    console.print(
        f"  [{_C['warn']}]↻[/]  [{_C['label']}]{tool_key}[/]"
        f"  [{_C['warn']}]retry #{attempt}[/]  [{_C['muted']}]{reason[:80]}[/]"
    )


def log_warning(msg: str, *, logger_name: str = "agent.core") -> None:
    logging.getLogger(logger_name).warning(msg)
    console.print(f"  [{_C['warn']}]⚠[/]  [{_C['warn']}]{msg}[/]")


def log_error(msg: str, *, logger_name: str = "agent.core") -> None:
    logging.getLogger(logger_name).error(msg)
    console.print(f"  [{_C['cross']}]✖[/]  [{_C['error']}]{msg}[/]")


def log_info(msg: str, *, logger_name: str = "agent.core") -> None:
    logging.getLogger(logger_name).info(msg)
    console.print(f"  [{_C['dot']}]·[/]  [{_C['muted']}]{msg}[/]")
