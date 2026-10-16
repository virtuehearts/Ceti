"""colour — retro ASCII BBS presentation for the terminal.

Phosphor-green-on-black by default, with cyan/amber/red accents.
Pure ASCII art (no block characters — a real BBS never needed them).

Colour is TTY-only: piped output, tests, and NO_COLOR get plain text,
so logs and assertions never see escape codes. CETI_COLOR=0 forces
plain text; CETI_COLOR=1 forces colour even off-tty (screenshots).
"""
from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
AMBER = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

BANNER = r"""
   ____  _____  _____  ___
  / ___|| ____||_   _||_ _|
 | |    |  _|    | |   | |
 | |___ | |___   | |   | |
  \____||_____|  |_|  |___|
  -- CETI // sovereign local brain --
  your iron. your data. your mind.
"""


def enabled() -> bool:
    """Colour on? TTY + not NO_COLOR + not forced off. CETI_COLOR=1
    forces it on (screenshots, demos)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CETI_COLOR") == "0":
        return False
    if os.environ.get("CETI_COLOR") == "1":
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def paint(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes when colour is enabled, else plain."""
    if not enabled() or not codes:
        return text
    return "".join(codes) + text + RESET


def clear_code() -> str:
    """ANSI clear-screen + home. Empty when colour is off, so captured
    output and tests never see a screen-clear."""
    return "\033[2J\033[H" if enabled() else ""


def clear_screen() -> None:
    code = clear_code()
    if code:
        sys.stdout.write(code)
        sys.stdout.flush()
