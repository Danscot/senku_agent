"""
tools/ask_tool.py

Lets the agent pause mid-execution and ask the user a question.
The plan halts, the question is shown, the user types an answer,
and execution resumes with the answer injected into the next step's context.

Use this when you hit a genuine decision point mid-task that you cannot
resolve yourself — e.g. conflicting files, missing config values,
ambiguous requirements discovered while reading code.

DO NOT use this for things you could reasonably decide yourself.
"""

from __future__ import annotations
from rich.console import Console
from rich.panel import Panel

_console = Console(highlight=False)

_C = {
    "accent": "bold yellow",
    "muted":  "dim white",
    "input":  "bold white",
    "border": "yellow",
}


def _run(params: dict) -> str:
    question = params.get("question", "").strip()
    context  = params.get("context", "").strip()   # why the agent is asking
    options  = params.get("options", [])            # optional suggestions

    if not question:
        return "ERROR: 'question' param is required."

    # ── Render the question panel ─────────────────────────────────────────────
    body_lines = []
    if context:
        body_lines.append(f"[{_C['muted']}]{context}[/]\n")
    body_lines.append(f"[{_C['input']}]{question}[/]")
    if options:
        body_lines.append("")
        body_lines.append(f"[{_C['muted']}]Options: " + "  ".join(f"[{o}]" for o in options) + "[/]")

    _console.print(
        Panel(
            "\n".join(body_lines),
            title=f"[{_C['accent']}]?  Senku needs your input[/]",
            border_style=_C["border"],
            padding=(1, 2),
        )
    )

    # ── Prompt for input ──────────────────────────────────────────────────────
    try:
        answer = _console.input(f"  [{_C['accent']}]Your answer >[/] ").strip()
    except (KeyboardInterrupt, EOFError):
        answer = ""

    if not answer:
        answer = "(no answer provided — use your best judgment)"

    _console.print(f"  [{_C['muted']}]Got it. Continuing…[/]\n")
    return f"User answered: {answer}"


TOOL_DEF = {
    "name":        "ask",
    "description": (
        "Pause execution and ask the user a question when you hit a decision point "
        "you cannot resolve yourself. The answer is returned as a string and available "
        "to subsequent steps via {{step_N}}. "
        "Use ONLY when genuinely stuck — not for things you can decide yourself."
    ),
    "params": [
        {
            "name": "question", "type": "str", "required": True,
            "description": "The specific question to ask the user",
        },
        {
            "name": "context", "type": "str", "required": False,
            "description": "One sentence explaining why you need to ask (shown to user)",
        },
        {
            "name": "options", "type": "list", "required": False,
            "description": "Optional list of suggested answers to show as hints",
        },
    ],
    "fn": _run,
}
