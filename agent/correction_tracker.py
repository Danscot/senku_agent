"""
agent/correction_tracker.py
Persistent memory for tool errors and their corrections.

Storage: corrections/tool_corrections.json
Schema:
  {
    "tool:shell": [
      {
        "error_pattern": "python: not found",
        "bad_params":    {"cmd": "python --version"},
        "fix_params":    {"cmd": "python3 --version"},
        "explanation":   "System uses python3, not python",
        "times_applied": 3,
        "last_seen":     "2024-01-15T10:30:00"
      },
      ...
    ],
    ...
  }
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

_STORE = Path(__file__).parent.parent / "corrections" / "tool_corrections.json"


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load() -> dict:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    if _STORE.exists():
        try:
            return json.loads(_STORE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(data: dict):
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Lookup ────────────────────────────────────────────────────────────────────

def find_correction(tool_key: str, params: dict, error_msg: str) -> Optional[dict]:
    """
    Check if we have a known fix for this tool + error combination.
    Returns the correction dict if found, else None.
    """
    data = _load()
    entries = data.get(tool_key, [])
    error_lower = error_msg.lower()

    for entry in entries:
        pattern = entry.get("error_pattern", "").lower()
        if pattern and pattern in error_lower:
            return entry

    # Fuzzy: check if any param value that failed matches a known bad param
    for entry in entries:
        bad = entry.get("bad_params", {})
        if any(str(v) == str(params.get(k)) for k, v in bad.items()):
            return entry

    return None


# ── Record ────────────────────────────────────────────────────────────────────

def record_correction(
    tool_key:    str,
    bad_params:  dict,
    fix_params:  dict,
    error_msg:   str,
    explanation: str,
):
    """
    Save a new correction or increment the counter of an existing one.
    """
    data    = _load()
    entries = data.setdefault(tool_key, [])
    ts      = datetime.now().isoformat(timespec="seconds")
    error_lower = error_msg.lower()

    # Update existing entry if pattern already known
    for entry in entries:
        pattern = entry.get("error_pattern", "").lower()
        if pattern and pattern in error_lower:
            entry["fix_params"]    = fix_params
            entry["explanation"]   = explanation
            entry["times_applied"] = entry.get("times_applied", 0) + 1
            entry["last_seen"]     = ts
            _save(data)
            return

    # New entry — derive a short error pattern (first meaningful line)
    pattern = _derive_pattern(error_msg)
    entries.append({
        "error_pattern": pattern,
        "bad_params":    bad_params,
        "fix_params":    fix_params,
        "explanation":   explanation,
        "times_applied": 1,
        "last_seen":     ts,
    })
    _save(data)


def _derive_pattern(error_msg: str) -> str:
    """Extract the most informative short phrase from an error message."""
    for line in error_msg.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Trim to 80 chars, lowercase
            return line[:80].lower()
    return error_msg[:80].lower()


# ── Describe (injected into system prompt) ────────────────────────────────────

def describe_corrections() -> str:
    """
    Return a compact summary of all known corrections for injection into
    the agent's system prompt so it avoids repeating past mistakes.
    """
    data = _load()
    if not data:
        return ""

    lines = ["KNOWN TOOL CORRECTIONS (avoid repeating these mistakes):"]
    for tool_key, entries in data.items():
        for e in entries:
            bad = e.get("bad_params", {})
            fix = e.get("fix_params", {})
            exp = e.get("explanation", "")
            lines.append(
                f"  ✖ {tool_key} bad={bad}  →  ✔ fix={fix}  ({exp})"
            )
    return "\n".join(lines)


# ── List all (for /corrections command) ──────────────────────────────────────

def all_corrections() -> list[dict]:
    data = _load()
    result = []
    for tool_key, entries in data.items():
        for e in entries:
            result.append({"tool": tool_key, **e})
    return result
