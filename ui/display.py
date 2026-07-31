from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich import box
from typing import List, Tuple
from .ascii_art import BANNER, SUBTITLE, STATUS_ICONS

console = Console()

# ── Palette ─────────────────────────────────────────────────────────────────
C = {
    "primary":   "bold white",
    "accent":    "bold cyan",
    "muted":     "dim white",
    "user":      "bold green",
    "agent":     "cyan",
    "tool":      "yellow",
    "skill":     "magenta",
    "plan":      "bold magenta",
    "result":    "bright_green",
    "error":     "bold red",
    "border":    "cyan",
    "step":      "bright_yellow",
    "think":     "dim cyan",
    "observe":   "dim blue",
    "reason":    "dim magenta",
    "decide":    "dim yellow",
    "reflect":   "dim green",
    "gap":       "bold red",
    "gap_cover": "green",
    "gap_miss":  "red",
}


# ── Startup ──────────────────────────────────────────────────────────────────
def render_banner():
    console.clear()
    console.print(f"[{C['accent']}]{BANNER}[/]")
    console.print(f"[{C['muted']}]{SUBTITLE}[/]\n")
    console.print(Rule(style="dim cyan"))
    console.print()


# ── Chat history ─────────────────────────────────────────────────────────────
def render_history(history: List[Tuple[str, str]], tail: int = 20):
    console.clear()
    console.print(f"[{C['accent']}]{BANNER}[/]")
    console.print(f"[{C['muted']}]{SUBTITLE}[/]\n")
    console.print(Rule(style="dim cyan"))
    console.print()

    if not history:
        console.print(f"  [{C['muted']}]No messages yet — start typing below[/]\n")
        return

    for role, msg in history[-tail:]:
        if role == "user":
            console.print(
                f"  [{C['user']}]You[/]  [{C['muted']}]›[/]  {msg}"
            )
        else:
            _print_agent_msg(msg)
        console.print()


def _print_agent_msg(msg: str):
    icon = STATUS_ICONS["done"]
    console.print(
        f"  [{C['accent']}]Senku[/]  [{C['muted']}]{icon}[/]  [{C['agent']}]{msg}[/]"
    )


# ── Mode banners ─────────────────────────────────────────────────────────────
def show_mode(mode: str):
    labels = {
        "chat":    (STATUS_ICONS["chat"],    "dim cyan",     "CHAT"),
        "single":  (STATUS_ICONS["execute"], "yellow",       "SINGLE TOOL"),
        "plan":    (STATUS_ICONS["plan"],    "bold magenta", "PLAN MODE"),
        "clarify": ("?",                     "bold yellow",  "CLARIFY"),
    }
    icon, style, label = labels.get(mode, ("◌", "white", mode.upper()))
    console.print(
        f"\n  [{style}]{icon}  {label}[/]\n",
        highlight=False,
    )


# ── Clarify display ───────────────────────────────────────────────────────────
def show_clarify(reason: str, questions: list[dict]):
    """Render the agent's clarification questions before it acts."""
    from rich.padding import Padding

    console.print(
        Panel(
            f"[{C['muted']}]{reason}[/]",
            title=f"[bold yellow]?  Before I act — I need your input[/]",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()

    for q in questions:
        num  = q.get("id", "")
        text = q.get("question", "")
        opts = q.get("options", [])

        console.print(f"  [bold yellow]{num}.[/] [white]{text}[/]")
        if opts:
            opt_str = "  ".join(f"[dim cyan][{o}][/]" for o in opts)
            console.print(f"     {opt_str}")
        console.print()


# ── Plan display ─────────────────────────────────────────────────────────────
def show_plan(plan: dict):
    goal = plan.get("goal", "")
    steps = plan.get("steps", [])

    console.print(
        Panel(
            f"[{C['muted']}]{goal}[/]",
            title=f"[{C['plan']}]{STATUS_ICONS['plan']}  Goal[/]",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    console.print()

    table = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"])
    table.add_column("#",    style="dim", width=3)
    table.add_column("Tool", style=C["tool"], width=14)
    table.add_column("Description", style=C["primary"])

    for s in steps:
        tool_label = s.get("tool", "—")
        if tool_label.startswith("skill:"):
            tool_label = f"[{C['skill']}]{tool_label}[/]"
        else:
            tool_label = f"[{C['tool']}]{tool_label}[/]"

        table.add_row(str(s.get("id", "?")), tool_label, s.get("description", ""))

    console.print(table)
    console.print(Rule(style="dim"))
    console.print()


# ── Step execution ────────────────────────────────────────────────────────────
def show_step_start(step: dict):
    icon = STATUS_ICONS["step"]
    tool = step.get("tool", "?")
    desc = step.get("description", "")
    console.print(
        f"  [{C['step']}]{icon} Step {step.get('id')}[/]  [{C['muted']}]{tool}[/]  —  {desc}"
    )


def show_step_result(result: str, error: bool = False):
    color = C["error"] if error else C["result"]
    icon  = STATUS_ICONS["error"] if error else STATUS_ICONS["done"]
    # Truncate long results for display
    preview = result[:300] + "…" if len(result) > 300 else result
    console.print(
        Panel(
            f"[{color}]{preview}[/]",
            title=f"[{color}]{icon}  Result[/]",
            border_style="dim green" if not error else "dim red",
            padding=(0, 2),
        )
    )
    console.print()


# ── Tool call ─────────────────────────────────────────────────────────────────
def show_tool_call(tool: str, inp: dict):
    icon = STATUS_ICONS.get(tool.split(":")[0], STATUS_ICONS["tool"])
    safe_inp = {k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
                for k, v in inp.items()}
    console.print(
        f"  [{C['tool']}]{icon}  {tool}[/]  [{C['muted']}]{safe_inp}[/]"
    )


# ── Spinners / progress ───────────────────────────────────────────────────────
def spinner(label: str) -> Progress:
    p = Progress(
        SpinnerColumn(style=C["accent"]),
        TextColumn(f"[{C['muted']}]{label}[/]"),
        transient=True,
    )
    return p


# ── Error ─────────────────────────────────────────────────────────────────────
def show_error(msg: str):
    console.print(
        Panel(
            f"[{C['error']}]{msg}[/]",
            title=f"[{C['error']}]{STATUS_ICONS['error']}  Error[/]",
            border_style="red",
            padding=(0, 2),
        )
    )


# ── Final response ────────────────────────────────────────────────────────────
def show_response(msg: str):
    console.print(
        Panel(
            f"[{C['agent']}]{msg}[/]",
            title=f"[{C['accent']}]{STATUS_ICONS['done']}  Senku[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


# ── Thinking log ──────────────────────────────────────────────────────────────
# Phase → (icon, color, label)
_THINK_PHASES = {
    "observe": ("○", C["observe"],  "OBSERVE"),
    "reason":  ("◈", C["reason"],   "REASON"),
    "decide":  ("◇", C["decide"],   "DECIDE"),
    "reflect": ("◆", C["reflect"],  "REFLECT"),
    "assess":  ("⊛", C["think"],    "ASSESS"),
    "gap":     ("✖", C["gap"],      "GAP"),
}

def show_thought(phase: str, text: str):
    """Render one thought line. Called live as the agent reasons."""
    icon, color, label = _THINK_PHASES.get(phase, ("·", C["muted"], phase.upper()))
    label_str = f"[{color}]{icon} {label:<8}[/]"
    # Wrap long thoughts at 100 chars
    if len(text) > 120:
        text = text[:117] + "…"
    console.print(f"  {label_str}  [{C['muted']}]{text}[/]", highlight=False)


def show_thinking_header():
    console.print(f"\n  [{C['think']}]◈  Thinking[/]", highlight=False)
    console.print(f"  [{C['muted']}]{'─' * 60}[/]")


def show_thinking_footer():
    console.print(f"  [{C['muted']}]{'─' * 60}[/]\n")


# ── Capability gap report ─────────────────────────────────────────────────────
def show_gap_report(report: dict):
    """
    report = {
      "task_summary": str,
      "requires": [{"need": str, "covered_by": str | None}],
      "gaps": [{"need": str, "suggested_tool": str, "reason": str}],
      "partial": bool,
    }
    """
    task    = report.get("task_summary", "")
    reqs    = report.get("requires", [])
    gaps    = report.get("gaps", [])
    partial = report.get("partial", False)

    # ── Header ────────────────────────────────────────────────────────────────
    status_color = "yellow" if partial else "red"
    status_label = "PARTIALLY SOLVABLE" if partial else "CANNOT COMPLETE"
    console.print(
        Panel(
            f"[{C['muted']}]{task}[/]",
            title=f"[bold {status_color}]✖  {status_label}[/]",
            border_style=status_color,
            padding=(0, 2),
        )
    )
    console.print()

    # ── Coverage table ────────────────────────────────────────────────────────
    if reqs:
        cov_table = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"], padding=(0, 1))
        cov_table.add_column("Requirement",  style=C["primary"],    min_width=28)
        cov_table.add_column("Status",       style=C["muted"],      width=10)
        cov_table.add_column("Covered by",   style=C["tool"],       min_width=20)

        for r in reqs:
            covered = r.get("covered_by")
            if covered:
                status = f"[{C['gap_cover']}]✔ covered[/]"
                by     = f"[{C['gap_cover']}]{covered}[/]"
            else:
                status = f"[{C['gap_miss']}]✖ missing[/]"
                by     = f"[dim]—[/]"
            cov_table.add_row(r.get("need", ""), status, by)

        console.print("  [dim]Requirements coverage[/]")
        console.print(cov_table)

    # ── Gap detail ────────────────────────────────────────────────────────────
    if gaps:
        console.print()
        console.print("  [bold red]Missing capabilities[/]")
        gap_table = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"], padding=(0, 1))
        gap_table.add_column("Gap",              style=C["primary"],  min_width=24)
        gap_table.add_column("Suggested tool",   style="bold yellow", min_width=24)
        gap_table.add_column("Why it's needed",  style=C["muted"],    min_width=32)

        for g in gaps:
            gap_table.add_row(
                g.get("need", ""),
                g.get("suggested_tool", ""),
                g.get("reason", ""),
            )

        console.print(gap_table)

    console.print(
        f"  [{C['muted']}]Add the suggested tools to [cyan]tools/[/] or [magenta]skills/[/] "
        f"then re-run your request.[/]\n"
    )


# ── Corrections log ───────────────────────────────────────────────────────────
def show_corrections(corrections: list[dict]):
    if not corrections:
        console.print("  [dim]No corrections recorded yet.[/]\n")
        return

    console.print(f"\n  [{C['step']}]◈  Tool Correction Memory[/]")
    console.print(f"  [{C['muted']}]{'─' * 60}[/]")

    table = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"], padding=(0, 1))
    table.add_column("Tool",        style=C["tool"],    min_width=16)
    table.add_column("Error",       style=C["error"],   min_width=22)
    table.add_column("Fix",         style=C["result"],  min_width=22)
    table.add_column("Explanation", style=C["muted"],   min_width=28)
    table.add_column("×",           style=C["muted"],   width=4)

    for c in corrections:
        table.add_row(
            c.get("tool", ""),
            str(c.get("bad_params", ""))[:40],
            str(c.get("fix_params", ""))[:40],
            c.get("explanation", "")[:50],
            str(c.get("times_applied", 1)),
        )
    console.print(table)
    console.print()
