"""
config.py — Single source of truth for environment-driven settings.

Multi-provider model configuration:
  Each "stage" in the agent pipeline can use a different model from a different
  provider. Set the env vars below to override any default.

  Providers supported:
    nvidia   — NVIDIA NIM (OpenAI-compatible),  base: https://integrate.api.nvidia.com/v1
    openai   — OpenAI API,                       base: https://api.openai.com/v1
    gemini   — Google via OpenAI-compat shim,    base: https://generativelanguage.googleapis.com/v1beta/openai
    anthropic— Anthropic via OpenAI-compat shim, base: https://api.anthropic.com/v1
    openrouter— OpenRouter,                      base: https://openrouter.ai/api/v1
    custom   — any OpenAI-compatible endpoint,   base: CUSTOM_BASE_URL
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
NVIDIA_API_KEY:      str = os.getenv("NVIDIA_API_KEY",      "")
OPENAI_API_KEY:      str = os.getenv("OPENAI_API_KEY",      "")
GEMINI_API_KEY:      str = os.getenv("GEMINI_API_KEY",      "")
ANTHROPIC_API_KEY:   str = os.getenv("ANTHROPIC_API_KEY",   "")
OPENROUTER_API_KEY:  str = os.getenv("OPENROUTER_API_KEY",  "")
CUSTOM_API_KEY:      str = os.getenv("CUSTOM_API_KEY",      "")
CUSTOM_BASE_URL:     str = os.getenv("CUSTOM_BASE_URL",     "")

# ── Provider base URLs ────────────────────────────────────────────────────────
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

# ── Per-stage model config ────────────────────────────────────────────────────
# Format: "provider:model-name"
# Each stage can be independently routed to any provider.
#
#   STAGE_THINK    — planning & JSON reasoning   (needs strong instruction-following)
#   STAGE_ACT      — tool-result checking/fixes  (fast, cheap)
#   STAGE_RESPOND  — final streaming response    (best quality, user-visible)
#   STAGE_COMPRESS — memory compression          (summarisation, can be cheap)

def _stage(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)

STAGE_THINK:    str = _stage("STAGE_THINK",    "gemini:gemini-3.1-flash-lite")
STAGE_ACT:      str = _stage("STAGE_ACT",      "gemini:gemini-3.1-flash-lite")
STAGE_RESPOND:  str = _stage("STAGE_RESPOND",  "gemini:gemini-3.1-flash-lite")
STAGE_COMPRESS: str = _stage("STAGE_COMPRESS", "gemini:gemini-3.1-flash-lite")

# ── Legacy single-model shim (backward compat) ────────────────────────────────
# If AGENT_MODEL is set it overrides all stages (old behaviour preserved).
_LEGACY_MODEL: str = os.getenv("AGENT_MODEL", "")
if _LEGACY_MODEL:
    STAGE_THINK    = f"nvidia:{_LEGACY_MODEL}"
    STAGE_ACT      = f"nvidia:{_LEGACY_MODEL}"
    STAGE_RESPOND  = f"nvidia:{_LEGACY_MODEL}"
    STAGE_COMPRESS = f"nvidia:{_LEGACY_MODEL}"

# Kept for any code that still imports MODEL directly
MODEL: str = _LEGACY_MODEL or STAGE_THINK.split(":", 1)[-1]

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", os.getcwd())


# ── Helper: parse "provider:model" ───────────────────────────────────────────
def parse_stage(stage_str: str) -> tuple[str, str, str, str]:
    """
    Parse a stage string like "gemini:gemini-2.0-flash-lite".
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
