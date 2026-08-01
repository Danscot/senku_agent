"""
skills/_llm.py — Shared LLM call helper for all skills.

Centralises:
  - max_tokens from config
  - temperature from config
  - thinking budget injection (when THINKING_ENABLED=True)
  - <thought> / <think> block stripping from responses
"""

from __future__ import annotations
import re
from config import THINKING_ENABLED, THINKING_BUDGET, TEMPERATURE_SKILLS


def skill_llm_call(client, model: str, messages: list[dict], max_tokens: int) -> str:
    """
    Make a streaming LLM call for a skill, accumulate chunks, strip
    chain-of-thought blocks, and return the clean text.

    Args:
        client:     OpenAI-compatible client
        model:      model name string
        messages:   list of {role, content} dicts
        max_tokens: token budget (pass the relevant TOKENS_* constant)
    """
    call_kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE_SKILLS,
        stream=True,
    )
    if THINKING_ENABLED:
        call_kwargs["extra_body"] = {
            "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET}
        }

    stream = client.chat.completions.create(**call_kwargs)

    chunks: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)

    raw = "".join(chunks).strip()
    return _strip_thought(raw)


def _strip_thought(text: str) -> str:
    """Remove <thought>…</thought> and <think>…</think> blocks."""
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>",    "", text, flags=re.DOTALL)
    return text.strip()
