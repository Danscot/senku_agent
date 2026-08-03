"""
skills/_llm.py — Shared LLM call helper for all skills.

Centralises:
  - max_tokens and temperature from config
  - per-provider thinking param injection (NVIDIA, Gemini, Anthropic)
  - delta.reasoning_content handling (NVIDIA thinking models)
  - <thought>/<think> block stripping as a safety net
"""

from __future__ import annotations
from config import TEMPERATURE_SKILLS


def skill_llm_call(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int,
    provider: str = "",
) -> str:
    """
    Make a streaming LLM call for a skill and return the clean response text.

    Handles per-provider thinking params and reasoning_content automatically.
    Strips <thought>/<think> blocks from the final output as a safety net.

    Args:
        client:     OpenAI-compatible client
        model:      model name string
        messages:   list of {role, content} dicts
        max_tokens: token budget (use the relevant TOKENS_* constant from config)
        provider:   provider name string (e.g. "nvidia", "gemini") — used to
                    select the correct thinking param format. Optional; if omitted
                    no thinking params are sent.
    """
    from agent.thinking import build_thinking_kwargs, extract_chunk, strip_thought_tags

    thinking_kwargs = build_thinking_kwargs(provider, model) if provider else {}

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE_SKILLS,
        stream=True,
        **thinking_kwargs,
    )

    content_chunks:   list[str] = []
    reasoning_chunks: list[str] = []

    for chunk in stream:
        content, reasoning = extract_chunk(chunk)
        if content:
            content_chunks.append(content)
        if reasoning:
            reasoning_chunks.append(reasoning)

    raw = "".join(content_chunks).strip()
    return strip_thought_tags(raw)
