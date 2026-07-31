from __future__ import annotations

from config import MODEL
"""
memory/buffer.py
Sliding-window conversation buffer with LLM-based compression
when the context grows too large.
"""

from dataclasses import dataclass, field
from typing import List

# One turn in the conversation
@dataclass
class Turn:
    role:    str   # "user" | "assistant" | "tool"
    content: str
    label:   str = ""  # optional tag, e.g. tool name

def _safe_extract(resp, default=""):
    """Safely extract content from API response."""
    if not resp or not resp.choices:
        return default
    content = resp.choices[0].message.content
    if content is None:
        return default
    return content.strip() if isinstance(content, str) else str(content).strip()

@dataclass
class ConversationBuffer:
    max_turns:    int = 40          # turns before compression kicks in
    keep_recent:  int = 10          # always keep the last N turns verbatim
    _turns: List[Turn] = field(default_factory=list, repr=False)
    _summary: str = ""              # compressed older context

    # ── Public API ────────────────────────────────────────────────────────────
    def push(self, role: str, content: str, label: str = ""):
        self._turns.append(Turn(role=role, content=content, label=label))

    def compress(self, client) -> None:
        """Compress the oldest turns into a summary using the LLM."""
        if len(self._turns) <= self.keep_recent:
            return

        old_turns = self._turns[:-self.keep_recent]
        recent    = self._turns[-self.keep_recent:]

        history_text = "\n".join(
            f"{t.role.upper()}: {t.content[:400]}" for t in old_turns
        )

        prompt = f"""Compress the following conversation history into a concise factual summary.
Preserve all task outcomes, decisions made, and important facts.
History:
{history_text}
"""
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        new_summary = "".join(chunks).strip()

        if self._summary:
            self._summary = self._summary + "\n\n" + new_summary
        else:
            self._summary = new_summary

        self._turns = recent

    def should_compress(self) -> bool:
        return len(self._turns) >= self.max_turns

    def to_messages(self) -> list[dict]:
        """Convert to the messages list expected by the OpenAI-compatible API."""
        msgs = []
        if self._summary:
            msgs.append({
                "role":    "user",
                "content": f"[Context from earlier in the session]\n{self._summary}",
            })
            msgs.append({
                "role":    "assistant",
                "content": "Understood. I have the context from earlier.",
            })
        for t in self._turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def last_n(self, n: int = 20) -> list[tuple[str, str]]:
        """Return the last N turns as (role, content) pairs for the UI."""
        return [(t.role, t.content) for t in self._turns[-n:]]

    def clear(self):
        self._turns.clear()
        self._summary = ""