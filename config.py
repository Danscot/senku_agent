"""
config.py — Single source of truth for ALL agent settings.

Multi-provider model configuration:
  Each "stage" in the agent pipeline can use a different model from a different
  provider. Set the env vars below to override any default.

  Providers supported:
    nvidia    — NVIDIA NIM (OpenAI-compatible),  base: https://integrate.api.nvidia.com/v1
    openai    — OpenAI API,                       base: https://api.openai.com/v1
    gemini    — Google via OpenAI-compat shim,    base: https://generativelanguage.googleapis.com/v1beta/openai
    anthropic — Anthropic via OpenAI-compat shim, base: https://api.anthropic.com/v1
    openrouter— OpenRouter,                       base: https://openrouter.ai/api/v1
    custom    — any OpenAI-compatible endpoint,   base: CUSTOM_BASE_URL
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ════════════════════════════════════════════════════════════════════════════════
#  API KEYS
# ════════════════════════════════════════════════════════════════════════════════

NVIDIA_API_KEY:     str = os.getenv("NVIDIA_API_KEY",     "")
OPENAI_API_KEY:     str = os.getenv("OPENAI_API_KEY",     "")
GEMINI_API_KEY:     str = os.getenv("GEMINI_API_KEY",     "")
ANTHROPIC_API_KEY:  str = os.getenv("ANTHROPIC_API_KEY",  "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
CUSTOM_API_KEY:     str = os.getenv("CUSTOM_API_KEY",     "")
CUSTOM_BASE_URL:    str = os.getenv("CUSTOM_BASE_URL",    "")


# ════════════════════════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY
# ════════════════════════════════════════════════════════════════════════════════

PROVIDER_URLS: dict[str, str] = {
    "nvidia":      "https://integrate.api.nvidia.com/v1",
    "openai":      "https://api.openai.com/v1",
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "anthropic":   "https://api.anthropic.com/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "custom":      CUSTOM_BASE_URL,
}

PROVIDER_KEYS: dict[str, str] = {
    "nvidia":      NVIDIA_API_KEY,
    "openai":      OPENAI_API_KEY,
    "gemini":      GEMINI_API_KEY,
    "anthropic":   ANTHROPIC_API_KEY,
    "openrouter":  OPENROUTER_API_KEY,
    "custom":      CUSTOM_API_KEY,
}


# ════════════════════════════════════════════════════════════════════════════════
#  PER-STAGE MODEL ROUTING
#  Format: "provider:model-name"
#  Each stage can be independently routed to any provider.
#
#    STAGE_THINK    — planning & JSON reasoning   (needs strong instruction-following)
#    STAGE_ACT      — tool-result checking/fixes  (fast, cheap)
#    STAGE_RESPOND  — final streaming response    (best quality, user-visible)
#    STAGE_COMPRESS — memory compression          (summarisation, can be cheap)
# ════════════════════════════════════════════════════════════════════════════════

def _stage(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)

STAGE_THINK:    str = _stage("STAGE_THINK",    "nvidia:moonshotai/kimi-k2-thinking")
STAGE_ACT:      str = _stage("STAGE_ACT",      "nvidia:moonshotai/kimi-k2-thinking")
STAGE_RESPOND:  str = _stage("STAGE_RESPOND",  "nvidia:moonshotai/kimi-k2-thinking")
STAGE_COMPRESS: str = _stage("STAGE_COMPRESS", "nvidia:moonshotai/kimi-k2-thinking")

# Legacy single-model shim (backward compat: AGENT_MODEL overrides all stages)
_LEGACY_MODEL: str = os.getenv("AGENT_MODEL", "")
if _LEGACY_MODEL:
    STAGE_THINK    = f"nvidia:{_LEGACY_MODEL}"
    STAGE_ACT      = f"nvidia:{_LEGACY_MODEL}"
    STAGE_RESPOND  = f"nvidia:{_LEGACY_MODEL}"
    STAGE_COMPRESS = f"nvidia:{_LEGACY_MODEL}"

# Kept for any skill code that still imports MODEL directly
MODEL: str = _LEGACY_MODEL or STAGE_THINK.split(":", 1)[-1]


# ════════════════════════════════════════════════════════════════════════════════
#  THINKING / CHAIN-OF-THOUGHT
#
#  THINKING_ENABLED controls:
#    1. Token budgets — thinking models need larger budgets because reasoning
#       consumes tokens before the actual answer starts.
#    2. NVIDIA-only: sends extra_body.chat_template_kwargs to enable thinking
#       on models like QwQ-32B, DeepSeek-R1, Kimi-K2, GLM-5.2.
#
#  Gemini / Gemma-4 think by DEFAULT through the OpenAI-compat shim.
#  No extra param is sent — the shim rejects any extra_body thinking fields.
#  <thought> blocks are always stripped from output regardless of this flag.
#
#  Set THINKING_ENABLED=false when using non-thinking models (gemini-flash,
#  gpt-4o, llama-3) to get tighter token budgets.
# ════════════════════════════════════════════════════════════════════════════════

THINKING_ENABLED: bool = os.getenv("THINKING_ENABLED", "true").lower() in ("1", "true", "yes")
THINKING_BUDGET:  int  = int(os.getenv("THINKING_BUDGET", "8000"))


# ════════════════════════════════════════════════════════════════════════════════
#  LLM CALL PARAMETERS — centralized token budgets per stage / use-case
#
#  Rules:
#    • Thinking models need higher limits because the <thought> block itself
#      consumes tokens before the actual answer starts.
#    • THINK stage needs the most: it outputs a full JSON plan.
#    • ACT stage is cheap: it only outputs yes/no or a small correction JSON.
#    • RESPOND stage is user-visible: give it room to write complete answers.
#    • COMPRESS stage just needs a short summary.
#    • Skill calls (code_writer, reviewer, etc.) have their own budgets
#      because they write complete files which can be long.
# ════════════════════════════════════════════════════════════════════════════════

# ── Core pipeline stages ──────────────────────────────────────────────────────
TOKENS_THINK:    int = int(os.getenv("TOKENS_THINK",    "16000" if THINKING_ENABLED else "8000"))
TOKENS_ACT:      int = int(os.getenv("TOKENS_ACT",      "4000"  if THINKING_ENABLED else "2000"))
TOKENS_RESPOND:  int = int(os.getenv("TOKENS_RESPOND",  "12000"))
TOKENS_COMPRESS: int = int(os.getenv("TOKENS_COMPRESS", "2000"))

# ── Skill-specific budgets ────────────────────────────────────────────────────
TOKENS_CODE_WRITER:        int = int(os.getenv("TOKENS_CODE_WRITER",        "8000"))
TOKENS_CODE_REVIEWER:      int = int(os.getenv("TOKENS_CODE_REVIEWER",      "4000"))
TOKENS_PROJECT_SCANNER:    int = int(os.getenv("TOKENS_PROJECT_SCANNER",    "1000" if THINKING_ENABLED else "500"))
TOKENS_TEST_RUNNER:        int = int(os.getenv("TOKENS_TEST_RUNNER",        "4000"))
TOKENS_DEPENDENCY_RESOLVER:int = int(os.getenv("TOKENS_DEPENDENCY_RESOLVER","1200" if THINKING_ENABLED else "600"))
TOKENS_MEMORY_COMPRESS:    int = int(os.getenv("TOKENS_MEMORY_COMPRESS",    "800"  if THINKING_ENABLED else "400"))


# ════════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE & SAMPLING
#  Most tasks need determinism (temp=0). Respond stage can be slightly warmer.
# ════════════════════════════════════════════════════════════════════════════════

TEMPERATURE_THINK:   float = float(os.getenv("TEMPERATURE_THINK",   "0.0"))
TEMPERATURE_ACT:     float = float(os.getenv("TEMPERATURE_ACT",     "0.0"))
TEMPERATURE_RESPOND: float = float(os.getenv("TEMPERATURE_RESPOND", "0.3"))
TEMPERATURE_SKILLS:  float = float(os.getenv("TEMPERATURE_SKILLS",  "0.2"))


# ════════════════════════════════════════════════════════════════════════════════
#  PROJECT ROOT
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", os.getcwd())


# ════════════════════════════════════════════════════════════════════════════════
#  HELPER: parse "provider:model"
# ════════════════════════════════════════════════════════════════════════════════

def parse_stage(stage_str: str) -> tuple[str, str, str, str]:
    """
    Parse a stage string like "gemini:gemma-4-31b-it".
    Returns (provider, model, base_url, api_key).
    Falls back to nvidia provider if no prefix given.
    """
    if ":" in stage_str:
        provider, model = stage_str.split(":", 1)
    else:
        provider, model = "nvidia", stage_str

    provider = provider.lower()
    base_url  = PROVIDER_URLS.get(provider, PROVIDER_URLS["nvidia"])
    api_key   = PROVIDER_KEYS.get(provider, NVIDIA_API_KEY)

    return provider, model, base_url, api_key
