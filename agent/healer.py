"""
agent/healer.py
Self-healing: detects errors caused by missing system tools or Python packages,
attempts to install/fix them automatically, then signals the caller to retry.

Two classes of fixable errors:
  1. Missing Python package  → pip install <pkg>
  2. Missing system binary   → suggest apt/brew install, or pip if available via pip

Healer is called BEFORE the param-correction logic in _dispatch, because
"yt-dlp not found" is a fundamentally different problem from "wrong params".
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Optional

from ui import C, console


# ─────────────────────────────────────────────────────────────────────────────
# Known error → fix mappings
# ─────────────────────────────────────────────────────────────────────────────

# Pattern (case-insensitive substring) → fix recipe
# fix recipe: {"type": "pip"|"binary", "pkg": str, "binary": str, "hint": str}

_KNOWN_FIXES: list[tuple[str, dict]] = [
    # ── Python packages ───────────────────────────────────────────────────────
    ("no module named 'yt_dlp'",     {"type": "pip", "pkg": "yt-dlp"}),
    ("no module named 'yt-dlp'",     {"type": "pip", "pkg": "yt-dlp"}),
    ("no module named 'speech_recognition'",
                                      {"type": "pip", "pkg": "SpeechRecognition"}),
    ("no module named 'pydub'",      {"type": "pip", "pkg": "pydub"}),
    ("no module named 'telegram'",   {"type": "pip", "pkg": "python-telegram-bot"}),
    ("no module named 'bs4'",        {"type": "pip", "pkg": "beautifulsoup4"}),
    ("no module named 'requests'",   {"type": "pip", "pkg": "requests"}),
    ("no module named 'PIL'",        {"type": "pip", "pkg": "Pillow"}),
    ("no module named 'numpy'",      {"type": "pip", "pkg": "numpy"}),
    ("no module named 'pandas'",     {"type": "pip", "pkg": "pandas"}),
    ("no module named 'openai'",     {"type": "pip", "pkg": "openai"}),
    ("no module named 'dotenv'",     {"type": "pip", "pkg": "python-dotenv"}),
    ("no module named 'rich'",       {"type": "pip", "pkg": "rich"}),

    # ── System binaries installed via pip ─────────────────────────────────────
    ("yt-dlp: not found",            {"type": "pip", "pkg": "yt-dlp"}),
    ("yt-dlp is not installed",      {"type": "pip", "pkg": "yt-dlp"}),
    ("'yt-dlp' is not installed",    {"type": "pip", "pkg": "yt-dlp"}),
    ("error: yt-dlp is not",         {"type": "pip", "pkg": "yt-dlp"}),

    # ── System binaries (no pip equivalent) ───────────────────────────────────
    ("ffmpeg: not found",            {"type": "binary", "binary": "ffmpeg",
                                      "hint": "sudo apt install ffmpeg  OR  brew install ffmpeg"}),
    ("ffmpeg is not installed",      {"type": "binary", "binary": "ffmpeg",
                                      "hint": "sudo apt install ffmpeg  OR  brew install ffmpeg"}),
    ("command not found: git",       {"type": "binary", "binary": "git",
                                      "hint": "sudo apt install git  OR  brew install git"}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def detect_fixable(error_msg: str) -> Optional[dict]:
    """
    Return a fix recipe if the error is a known missing-dependency problem,
    or None if we don't know how to fix it automatically.
    """
    lower = error_msg.lower()
    for pattern, recipe in _KNOWN_FIXES:
        if pattern in lower:
            return recipe

    # Generic ModuleNotFoundError / ImportError pattern
    m = re.search(
        r"(?:no module named|modulenotfounderror:.*?)'([a-z0-9_\-]+)'",
        lower,
    )
    if m:
        pkg = m.group(1).replace("_", "-")
        return {"type": "pip", "pkg": pkg}

    # Generic "X: not found" / "X is not installed" for known pip-installable CLIs
    m2 = re.search(r"'?([a-z][a-z0-9_\-]+)'?\s+(?:is not installed|: not found)", lower)
    if m2:
        candidate = m2.group(1)
        # Only auto-pip things that are known CLI tools
        _pip_cli = {"yt-dlp", "ffmpeg-python", "mutagen", "httpx"}
        if candidate in _pip_cli:
            return {"type": "pip", "pkg": candidate}

    return None


def attempt_fix(recipe: dict) -> tuple[bool, str]:
    """
    Try to apply the fix. Returns (success: bool, message: str).
    """
    fix_type = recipe.get("type")

    # ── pip install ───────────────────────────────────────────────────────────
    if fix_type == "pip":
        pkg = recipe["pkg"]
        console.print(
            f"\n  [{C['step']}]⟳  Healer:[/]  "
            f"[{C['muted']}]pip install {pkg}[/]"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print(
                    f"  [{C['result']}]✔  Installed {pkg} successfully[/]"
                )
                return True, f"Installed {pkg} via pip"
            else:
                err = (result.stderr or result.stdout).strip()[-200:]
                return False, f"pip install {pkg} failed: {err}"
        except subprocess.TimeoutExpired:
            return False, f"pip install {pkg} timed out"
        except Exception as e:
            return False, f"pip install {pkg} raised {e}"

    # ── system binary ─────────────────────────────────────────────────────────
    if fix_type == "binary":
        binary = recipe["binary"]
        hint   = recipe.get("hint", f"Install {binary} using your system package manager")

        # Check if it somehow already exists
        if shutil.which(binary):
            return True, f"{binary} is already installed"

        console.print(
            f"\n  [{C['error']}]⟳  Healer:[/]  "
            f"[{C['muted']}]{binary} requires manual installation[/]\n"
            f"  [{C['step']}]Run:[/]  [{C['accent']}]{hint}[/]\n"
        )
        return False, f"Cannot auto-install system binary '{binary}'. Run: {hint}"

    return False, f"Unknown fix type: {fix_type}"


def try_heal(error_msg: str) -> tuple[bool, str]:
    """
    Top-level entry point.
    Returns (fixed: bool, message: str).
    If fixed=True, the caller should retry the original operation.
    """
    recipe = detect_fixable(error_msg)
    if recipe is None:
        return False, "No automatic fix available for this error."

    return attempt_fix(recipe)
