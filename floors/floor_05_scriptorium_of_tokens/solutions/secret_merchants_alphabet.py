"""SECRET - THE MERCHANT'S ALPHABET  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

GPT-2 stores its byte-level vocabulary as *text*, which means every byte value
needs a printable, non-whitespace character to stand for it. The 188 bytes that
are already printable keep their own character; the other 68 (control bytes,
space, DEL, the C1 range and a few Latin-1 oddities) are pushed up into unused
code points starting at U+0100. The result is a bijection, so it costs nothing.
"""

from __future__ import annotations

import re


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> printable character bijection."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {b: chr(b) for b in printable}
    shift = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shift)  # U+0100 onward: printable, unused by Latin-1
            shift += 1
    return table


def unicode_to_bytes() -> dict[str, int]:
    """The inverse of :func:`bytes_to_unicode`."""
    return {ch: b for b, ch in bytes_to_unicode().items()}


def bytes_to_visible(data: bytes) -> str:
    """Render raw bytes as a string of stand-in characters, one per byte."""
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)


def visible_to_bytes(text: str) -> bytes:
    """Undo :func:`bytes_to_visible`."""
    table = unicode_to_bytes()
    return bytes(table[ch] for ch in text)


# GPT-2's real pattern uses \p{L} and \p{N} from the third-party ``regex`` module.
# The stdlib ``re`` has no \p{...} classes, so letters are written as "word
# characters that are not digits or underscore", and underscore is folded into
# the punctuation branch so that every character matches *some* branch.
PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[str]:
    """Split into words, numbers, punctuation runs and whitespace, GPT-2 style."""
    return PRETOKENIZE_PATTERN.findall(text)
