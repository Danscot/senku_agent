"""
agent/thinking.py — Per-provider thinking configuration.

Reality of thinking support through OpenAI-compat shims:

  NVIDIA  — extra_body: {chat_template_kwargs: {enable_thinking: True, clear_thinking: False}}
            Works for select models (QwQ, DeepSeek-R1, GLM-5.2, Kimi-K2).
            Reasoning tokens arrive on delta.reasoning_content (separate field).

  Gemini  — Thinks by DEFAULT. The OpenAI-compat shim rejects any extra_body
            thinking fields (thinking_config, thinking, etc.) with a 400.
            No params sent — we just strip <thought> blocks from the output.

  Anthropic — extra_body: {thinking: {...}} only works on their native API.
              Through the OpenAI shim it also causes a 400. Not used here.

  OpenAI  — o1/o3 reason internally. No param accepted or needed.

  Default — No extra_body sent. <thought>/<think> stripping always happens
            because some models emit these tags regardless.
"""

from __future__ import annotations
import re
from config import THINKING_ENABLED, THINKING_BUDGET


# ── Models known to support thinking per provider ─────────────────────────────
# Used to auto-detect whether to send thinking params even when THINKING_ENABLED=True.
# If THINKING_ENABLED=True but the model isn't in this list, we skip the param
# (avoids 400 errors on non-thinking models like gemini-flash).

_NVIDIA_THINKING_MODELS: set[str] = {
    "z-ai/glm-5.2",
    "qwen/qwq-32b",
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-r1-0528",
    "moonshotai/kimi-k2-thinking",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
}

_GEMINI_THINKING_MODELS: set[str] = {
    "gemma-4-31b-it",
    "gemma-4-27b-it",
    "gemma-4-9b-it",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash-thinking-exp",
}

# Prefix match — catches versioned names like "gemma-4-31b-it-v1"
_NVIDIA_THINKING_PREFIXES = ("qwen/qwq", "deepseek-ai/deepseek-r1", "moonshotai/kimi-k2")
_GEMINI_THINKING_PREFIXES = ("gemma-4", "gemini-2.5")


def _model_supports_thinking(provider: str, model: str) -> bool:
    """Return True if this provider+model combo is known to support thinking params."""
    m = model.lower()
    if provider == "nvidia":
        return (
            m in _NVIDIA_THINKING_MODELS
            or any(m.startswith(p) for p in _NVIDIA_THINKING_PREFIXES)
        )
    if provider == "gemini":
        return (
            m in _GEMINI_THINKING_MODELS
            or any(m.startswith(p) for p in _GEMINI_THINKING_PREFIXES)
        )
    return False


def build_thinking_kwargs(provider: str, model: str) -> dict:
    """
    Return extra kwargs to merge into client.chat.completions.create().

    Reality check per provider (OpenAI-compat shim):
      - NVIDIA:    supports extra_body.chat_template_kwargs for select models
      - Gemini:    thinks by default — the shim rejects ANY extra_body thinking field
      - Anthropic: extra_body.thinking works on their own API, not the shim
      - OpenAI:    o1/o3 reason internally, no param needed
      - Default:   nothing sent; stripping always happens regardless

    Returns {} for any provider that doesn't accept thinking params via the shim.
    """
    if not THINKING_ENABLED:
        return {}
    if not _model_supports_thinking(provider, model):
        return {}

    # Only NVIDIA is confirmed to accept thinking params via OpenAI-compat shim
    if provider == "nvidia":
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False,
                }
            },
            "seed": 42,
        }

    # Gemini, Anthropic shim, OpenRouter, custom — think by default or internally.
    # Sending any extra_body thinking field causes a 400. Return nothing.
    return {}


def extract_chunk(chunk) -> tuple[str, str]:
    """
    Extract (content, reasoning) from a streaming chunk.

    NVIDIA thinking models return reasoning on delta.reasoning_content.
    Gemini/others may also use this field or fold it into content with tags.

    Returns:
        content   — the actual response text (shown to user / parsed as JSON)
        reasoning — the chain-of-thought text (logged/displayed separately)
    """
    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
        return "", ""

    delta = chunk.choices[0].delta
    if delta is None:
        return "", ""

    content   = getattr(delta, "content",           None) or ""
    reasoning = getattr(delta, "reasoning_content", None) or ""

    return content, reasoning


# ── Post-processing: strip thought tags from accumulated text ─────────────────

def strip_thought_tags(text: str) -> str:
    """
    Remove <thought>...</thought> and <think>...</think> blocks.
    Used as a safety net for models that fold reasoning into content
    rather than using the reasoning_content field.
    """
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>",     "", text, flags=re.DOTALL)
    return text.strip()
