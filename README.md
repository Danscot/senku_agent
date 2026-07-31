# Senku — Autonomous Coding Agent

> *"10 billion percent, this will work."*

A multi-stage coding agent specialized in Django and HTML/CSS/JavaScript, powered by any OpenAI-compatible LLM provider.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env

# Run without a project — ask Senku to build one for you
python main.py

# Or point it at an existing project
python main.py --project /path/to/myapp

# Or let it create a new directory
python main.py --project /path/to/new-django-app
```

---

## No-Project Mode

You don't need an existing project to use Senku. Just run `python main.py` and ask:

- *"Make me a Django webapp called blog"*
- *"Create a new Django REST API project in ./api"*
- *"Scaffold a todo app with models, views, and templates"*

Senku will scaffold the project directly into the configured project root (or wherever you point it).

---

## Project Root & Path Resolution

All tool calls (shell commands, file reads/writes, code generation) automatically execute inside the **project root**:

- Shell commands run with `cwd` set to the project root
- File paths are resolved relative to the project root
- The LLM always knows which directory it's working in

Set the project root via:
- `--project /path` CLI flag (highest priority)
- `PROJECT_ROOT=/path` in `.env`
- Falls back to `cwd` if neither is set

---

## Environment Variables (`.env`)

```env
# API keys — at least one required
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...

# Per-stage model routing (provider:model-name)
STAGE_THINK=nvidia:moonshotai/kimi-k2-thinking
STAGE_ACT=nvidia:moonshotai/kimi-k2-thinking
STAGE_RESPOND=nvidia:moonshotai/kimi-k2-thinking
STAGE_COMPRESS=nvidia:moonshotai/kimi-k2-thinking

# Default project directory
PROJECT_ROOT=/path/to/your/project
```

---

## Commands

| Command         | Description                              |
|----------------|------------------------------------------|
| `/scan`         | Scan project and rebuild `structure.md`  |
| `/project`      | Show current project root                |
| `/tools`        | List available tools and skills          |
| `/models`       | Show per-stage model configuration       |
| `/corrections`  | Show recorded tool correction memory     |
| `/clear`        | Clear conversation memory                |
| `/help`         | Show help                                |
| `exit` / `quit` | Exit the agent                           |

---

## Architecture

```
main.py          — CLI entry, project root resolution
agent/core.py    — THINK → ACT → CHECK → RESPOND pipeline
tools/           — shell, file, git, django, search
skills/          — code_writer, code_reviewer, test_runner, project_scanner
memory/          — conversation buffer with compression
ui/              — Rich console display (Senku-branded)
mcp/             — MCP server integrations
```
