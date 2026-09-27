"""Tiny terminal toolkit: colours, boxes, the banner.

Deliberately dependency-free. Colour is disabled when stdout is not a TTY or when
``NO_COLOR`` is set, and forced on with ``FORCE_COLOR``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap

_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "grey": "90",
}


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force not in ("", "0", "false", "no")
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def paint(text: str, *styles: str) -> str:
    if not styles or not color_enabled():
        return text
    codes = ";".join(_CODES[s] for s in styles)
    return f"\033[{codes}m{text}\033[0m"


def bold(t: str) -> str:
    return paint(t, "bold")


def dim(t: str) -> str:
    return paint(t, "dim")


def red(t: str) -> str:
    return paint(t, "red")


def green(t: str) -> str:
    return paint(t, "green")


def yellow(t: str) -> str:
    return paint(t, "yellow")


def cyan(t: str) -> str:
    return paint(t, "cyan")


def magenta(t: str) -> str:
    return paint(t, "magenta")


def grey(t: str) -> str:
    return paint(t, "grey")


def width(default: int = 80, cap: int = 100) -> int:
    return min(shutil.get_terminal_size((default, 24)).columns, cap)


def hr(char: str = "─", w: int | None = None) -> str:
    return dim(char * (w or width()))


def title(text: str) -> str:
    w = width()
    bar = "═" * max(0, (w - len(text) - 2) // 2)
    return paint(f"{bar} {text} {bar}", "bold", "magenta")


def wrap(text: str, indent: str = "  ", w: int | None = None) -> str:
    w = (w or width()) - len(indent)
    paragraphs = text.strip().split("\n\n")
    out = []
    for para in paragraphs:
        out.append(textwrap.fill(" ".join(para.split()), width=w, initial_indent=indent, subsequent_indent=indent))
    return "\n\n".join(out)


_ANSI = re.compile(r"\033\[[0-9;]*m")


def visible_len(s: str) -> int:
    """Length of ``s`` as the terminal shows it (ANSI colour codes excluded)."""
    return len(_ANSI.sub("", s))


def pad(s: str, n: int) -> str:
    """Left-justify ``s`` to ``n`` visible columns, ignoring ANSI codes."""
    return s + " " * max(0, n - visible_len(s))


def box(lines: list[str], header: str | None = None, pad: int = 1) -> str:
    """Draw a single-line box around ``lines`` (ANSI codes are ignored for width)."""
    visible = visible_len
    inner = max([visible(s) for s in lines] + [visible(header or "")]) + pad * 2
    top = "┌" + "─" * inner + "┐"
    if header:
        head = f" {header} "
        top = "┌" + "─" * pad + head + "─" * max(0, inner - pad - visible(head)) + "┐"
    body = []
    for s in lines:
        fill = inner - visible(s) - pad
        body.append("│" + " " * pad + s + " " * max(0, fill) + "│")
    bottom = "└" + "─" * inner + "┘"
    return "\n".join([top, *body, bottom])


BANNER = r"""
 _   _                      _   ____
| \ | | ___ _   _ _ __ __ _| | |  _ \ _   _ _ __   __ _  ___  ___  _ __
|  \| |/ _ \ | | | '__/ _` | | | | | | | | | '_ \ / _` |/ _ \/ _ \| '_ \
| |\  |  __/ |_| | | | (_| | | | |_| | |_| | | | | (_| |  __/ (_) | | | |
|_| \_|\___|\__,_|_|  \__,_|_| |____/ \__,_|_| |_|\__, |\___|\___/|_| |_|
                                                   |___/
""".strip("\n")


def banner() -> str:
    return paint(BANNER, "magenta")


# Glyphs used in the map and status lines. Kept in one place so a plain-ASCII
# fallback is easy if a terminal cannot draw them.
GLYPHS = {
    "cleared": "★",
    "boss_defeated": "☠",
    "boss_alive": "☠",
    "secret_found": "◆",
    "secret_hidden": "◇",
    "room_done": "■",
    "room_open": "□",
    "arrow": "→",
    "surface": "☀",
    "locked": "🔒",
}

if os.environ.get("DUNGEON_ASCII"):
    GLYPHS.update(
        cleared="*",
        boss_defeated="X",
        boss_alive="X",
        secret_found="#",
        secret_hidden="+",
        room_done="#",
        room_open="-",
        arrow="->",
        surface="O",
        locked="[locked]",
    )


def glyph(name: str) -> str:
    return GLYPHS[name]
