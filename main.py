#!/usr/bin/env python3
"""
main.py — Coding Agent CLI entrypoint

Usage:
  python main.py                         # start without a project (create mode)
  python main.py --project /path/to/proj # work on an existing project
  python main.py -p /path/to/new/dir     # will offer to create the directory
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    STAGE_THINK, STAGE_ACT, STAGE_RESPOND, STAGE_COMPRESS,
    PROJECT_ROOT, parse_stage,
)
from agent import Agent
from ui import (
    console, C,
    render_banner,
    render_history,
    show_error,
    show_corrections,
)
from ui.logger import log_warning, log_info

load_dotenv()


# ── Startup model config display ──────────────────────────────────────────────

def _render_model_config() -> None:
    """Print a table showing which provider+model is assigned to each stage."""
    from rich.table import Table
    from rich import box
    from config import THINKING_ENABLED, THINKING_BUDGET, TOKENS_THINK, TOKENS_ACT, TOKENS_RESPOND, TOKENS_COMPRESS

    console.print(f"  [{C['muted']}]Model config[/]\n")
    table = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"], padding=(0, 2))
    table.add_column("Stage",    style="dim white",  min_width=10)
    table.add_column("Provider", style="magenta",    min_width=12)
    table.add_column("Model",    style="bold white", min_width=28)
    table.add_column("Tokens",   style="cyan",       min_width=8)

    stage_tokens = {
        "think":    TOKENS_THINK,
        "act":      TOKENS_ACT,
        "respond":  TOKENS_RESPOND,
        "compress": TOKENS_COMPRESS,
    }
    for stage_label, stage_str in [
        ("think",    STAGE_THINK),
        ("act",      STAGE_ACT),
        ("respond",  STAGE_RESPOND),
        ("compress", STAGE_COMPRESS),
    ]:
        provider, model, _, _ = parse_stage(stage_str)
        table.add_row(stage_label, provider, model, str(stage_tokens[stage_label]))

    console.print(table)

    think_status = (
        f"[bold yellow]ON[/]  budget={THINKING_BUDGET} tokens"
        if THINKING_ENABLED else
        "[dim]OFF[/]"
    )
    console.print(f"  [{C['muted']}]Thinking :[/]  {think_status}\n")


# ── Client factory ────────────────────────────────────────────────────────────

def _make_default_client() -> OpenAI:
    """Build the primary client (used for backward compat and direct skill calls)."""
    provider, model, base_url, api_key = parse_stage(STAGE_THINK)
    if not api_key:
        console.print(
            f"\n  [bold red]✖  No API key found for provider '{provider}'.[/]\n"
            f"  Add it to your [cyan].env[/] file, e.g.:\n\n"
            f"    [dim]NVIDIA_API_KEY=nvapi-…[/]    for nvidia\n"
            f"    [dim]OPENAI_API_KEY=sk-…[/]      for openai\n"
            f"    [dim]GEMINI_API_KEY=AIza…[/]     for gemini\n\n"
            f"  Then set the stage in .env:\n"
            f"    [dim]STAGE_THINK=gemini:gemini-2.0-flash-lite[/]\n"
        )
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


# ── Project root resolution ───────────────────────────────────────────────────

def _resolve_project_root(arg_project: str | None) -> str:
    """
    Determine the project root.

    Priority:
      1. --project CLI arg
      2. PROJECT_ROOT from .env (if it exists as a dir)
      3. Current working directory (fallback / no-project mode)

    If the path does not exist, offer to create it (for new project scaffolding).
    """
    candidate = arg_project or PROJECT_ROOT

    if os.path.isdir(candidate):
        return candidate

    # Path given but doesn't exist — offer to create for new projects
    if arg_project:
        console.print(
            f"\n  [yellow]⚠  Directory not found:[/] [white]{candidate}[/]"
        )
        try:
            answer = console.input(
                f"  [{C['accent']}]Create it as a new project directory? [y/N][/] "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print()
            sys.exit(0)

        if answer in ("y", "yes"):
            os.makedirs(candidate, exist_ok=True)
            console.print(f"  [green]✔  Created:[/] {candidate}\n")
            return candidate
        else:
            console.print(
                f"\n  [dim]Falling back to current directory: {os.getcwd()}[/]\n"
            )
            return os.getcwd()

    # No project arg and PROJECT_ROOT doesn't exist — use cwd (no-project mode)
    return os.getcwd()


# ── Commands ───────────────────────────────────────────────────────────────────

HELP_TEXT = """
  [bold cyan]Commands[/]
  [dim]──────────────────────────────────────────[/]
  [green]exit / quit[/]       — quit the agent
  [green]/clear[/]             — clear conversation memory
  [green]/tools[/]             — list available tools & skills
  [green]/config[/]            — show all LLM token budgets and thinking settings
  [green]/models[/]            — show per-stage model configuration
  [green]/corrections[/]       — show recorded tool corrections
  [green]/scan[/]              — scan project & rebuild structure.md
  [green]/project[/]           — show current project root
  [green]/help[/]              — show this message
  [dim]──────────────────────────────────────────[/]
  Tip: You can ask Senku to create a new project from scratch —
       e.g. "make me a Django webapp in ./myapp"
  Tip: structure.md is auto-injected into every planning prompt.
       Run /scan whenever you add or move files.
"""


def _cmd_tools(agent: Agent) -> None:
    import tools  as tr
    import skills as sr
    import mcp    as mr

    console.print("\n  [bold cyan]◆  Tools[/]")
    for t in tr.all_tools():
        console.print(f"  [yellow]·[/] [white]{t['name']}[/]  [dim]{t['description']}[/]")

    console.print("\n  [bold magenta]✦  Skills[/]")
    for s in sr.all_skills():
        tag = " [dim][learned][/]" if s.get("learned") else ""
        console.print(f"  [magenta]·[/] [white]{s['name']}[/]  [dim]{s['description']}[/]{tag}")

    enabled_mcp = mr.all_servers()
    if enabled_mcp:
        console.print("\n  [bold blue]⊕  MCP Servers[/]")
        for m in enabled_mcp:
            console.print(f"  [blue]·[/] [white]{m['name']}[/]  [dim]{m['description']}[/]")
    console.print()


def _cmd_config() -> None:
    from rich.table import Table
    from rich import box
    from config import (
        THINKING_ENABLED, THINKING_BUDGET,
        TOKENS_THINK, TOKENS_ACT, TOKENS_RESPOND, TOKENS_COMPRESS,
        TOKENS_CODE_WRITER, TOKENS_CODE_REVIEWER, TOKENS_PROJECT_SCANNER,
        TOKENS_TEST_RUNNER, TOKENS_DEPENDENCY_RESOLVER, TOKENS_MEMORY_COMPRESS,
        TEMPERATURE_THINK, TEMPERATURE_ACT, TEMPERATURE_RESPOND, TEMPERATURE_SKILLS,
    )

    console.print(f"\n  [bold cyan]◆  LLM Configuration[/]\n")

    # Thinking
    think_status = (
        f"[bold yellow]ENABLED[/]  budget = {THINKING_BUDGET} tokens"
        if THINKING_ENABLED else "[dim]DISABLED[/]"
    )
    console.print(f"  [dim]Thinking     :[/]  {think_status}\n")

    # Token budgets
    t = Table(box=box.SIMPLE, show_header=True, header_style=C["muted"], padding=(0, 2))
    t.add_column("Call site",           style="white",    min_width=26)
    t.add_column("Max tokens",          style="cyan",     min_width=12)
    t.add_column("Temperature",         style="magenta",  min_width=12)
    t.add_column("Env var to override", style="dim white",min_width=28)

    rows = [
        ("think stage",          TOKENS_THINK,              TEMPERATURE_THINK,   "TOKENS_THINK / TEMPERATURE_THINK"),
        ("act stage",            TOKENS_ACT,                TEMPERATURE_ACT,     "TOKENS_ACT / TEMPERATURE_ACT"),
        ("respond stage",        TOKENS_RESPOND,            TEMPERATURE_RESPOND, "TOKENS_RESPOND / TEMPERATURE_RESPOND"),
        ("compress stage",       TOKENS_COMPRESS,           TEMPERATURE_ACT,     "TOKENS_COMPRESS"),
        ("skill: code_writer",   TOKENS_CODE_WRITER,        TEMPERATURE_SKILLS,  "TOKENS_CODE_WRITER"),
        ("skill: code_reviewer", TOKENS_CODE_REVIEWER,      TEMPERATURE_SKILLS,  "TOKENS_CODE_REVIEWER"),
        ("skill: project_scan",  TOKENS_PROJECT_SCANNER,    TEMPERATURE_SKILLS,  "TOKENS_PROJECT_SCANNER"),
        ("skill: test_runner",   TOKENS_TEST_RUNNER,        TEMPERATURE_SKILLS,  "TOKENS_TEST_RUNNER"),
        ("skill: dep_resolver",  TOKENS_DEPENDENCY_RESOLVER,TEMPERATURE_SKILLS,  "TOKENS_DEPENDENCY_RESOLVER"),
        ("memory: compress",     TOKENS_MEMORY_COMPRESS,    TEMPERATURE_SKILLS,  "TOKENS_MEMORY_COMPRESS"),
    ]
    for name, tok, temp, env in rows:
        t.add_row(name, str(tok), str(temp), env)

    console.print(t)
    console.print(f"  [dim]Set any env var in your .env to override. Restart the agent to apply.[/]\n")


def _cmd_models() -> None:
    _render_model_config()


def _cmd_scan(agent: Agent) -> None:
    import skills as sr
    scanner = sr.get("project_scanner")
    if not scanner:
        console.print("  [red]✖  project_scanner skill not found.[/]\n")
        return

    console.print(f"  [dim]Scanning '{agent.project_root}'…[/]\n")
    result = scanner["fn"]({"root": agent.project_root}, agent.client)
    console.print(f"\n  [green]{result[:400]}[/]\n")
    agent._rebuild_think_system()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Senku Coding Agent")
    parser.add_argument(
        "--project", "-p", default=None,
        help=(
            "Project root directory (overrides PROJECT_ROOT env var). "
            "Omit to run without a project — you can ask Senku to create one."
        ),
    )
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project)

    client = _make_default_client()
    agent  = Agent(client, project_root=project_root)

    render_banner()

    # Determine display mode label
    is_existing = os.path.isdir(project_root) and bool(os.listdir(project_root))
    mode_label  = f"[bold white]{project_root}[/]"
    if not is_existing:
        mode_label += f"  [{C['muted']}](empty — ready to scaffold)[/]"

    console.print(f"  [{C['muted']}]Project :[/]  {mode_label}\n")
    _render_model_config()

    from pathlib import Path
    if not (Path(project_root) / "structure.md").exists():
        console.print(
            "  [yellow]⚠  No structure.md found.[/]  "
            "Run [bold]/scan[/] to generate one, or just start asking Senku to build something.\n"
        )

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = console.input("  [bold green]>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Goodbye.[/]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("\n  [dim]Goodbye.[/]\n")
            break

        if user_input.lower() == "/clear":
            agent.memory.clear()
            history.clear()
            render_banner()
            console.print("  [dim]Memory cleared.[/]\n")
            continue

        if user_input.lower() == "/tools":
            _cmd_tools(agent)
            continue

        if user_input.lower() == "/config":
            _cmd_config()
            continue

        if user_input.lower() == "/models":
            _cmd_models()
            continue

        if user_input.lower() == "/scan":
            _cmd_scan(agent)
            continue

        if user_input.lower() == "/project":
            console.print(f"\n  [cyan]Project root:[/] {agent.project_root}\n")
            continue

        if user_input.lower() == "/help":
            console.print(HELP_TEXT)
            continue

        if user_input.lower() == "/corrections":
            from agent.correction_tracker import all_corrections
            show_corrections(all_corrections())
            continue

        history.append(("user", user_input))
        console.print()

        try:
            result = agent.run(user_input)
        except KeyboardInterrupt:
            console.print("\n  [dim]Interrupted.[/]\n")
            continue
        except Exception as e:
            show_error(str(e))
            result = f"An error occurred: {e}"

        if result:
            history.append(("assistant", result))

        render_history(history)


if __name__ == "__main__":
    main()
