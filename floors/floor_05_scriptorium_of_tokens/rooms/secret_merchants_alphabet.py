"""SECRET - THE MERCHANT'S ALPHABET   (optional)

    A ledger slides under the door of the Golem's vault. A travelling merchant
    once needed to write byte-level vocabularies down on paper, and paper
    cannot hold a newline or a NUL. So the merchant invented an alphabet: 256
    printable characters, one for every byte value, and a rule to go back.

GPT-2 stores its vocabulary as text (vocab.json). That only works if every byte
value has a printable, non-whitespace stand-in. The trick:

    * bytes that are already printable in Latin-1 keep their own character:
      '!'..'~' (33..126), '¡'..'¬' (161..172), '®'..'ÿ' (174..255) - 188 of them;
    * the remaining 68 (controls, space, DEL, 128..160, '\xad') are mapped to
      U+0100, U+0101, ... in increasing byte order.

The result is a bijection between 256 byte values and 256 characters, so a byte
string becomes a same-length character string and back with no loss. When you
see "Ġthe" in a GPT-2 vocabulary, that "Ġ" is byte 0x20, a space, in this alphabet.

Pre-tokenization: before BPE ever runs, GPT-2 splits text with a regular
expression into words, numbers, punctuation runs and whitespace, each with an
optional leading space glued on. Merges never cross those boundaries. The real
pattern is

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

and it needs the third-party ``regex`` module for \\p{L} (any letter) and \\p{N}
(any number). The stdlib ``re`` has no \\p{...} classes. Write the closest
stdlib version: letters are ``[^\\W\\d_]`` (word characters that are not digits or
underscore), numbers are ``\\d``, and everything that is neither whitespace nor
a letter nor a digit is punctuation. Make sure EVERY character matches some
branch (underscore is the one people forget), so the pieces concatenate back to
the original text.
"""

from __future__ import annotations

import re

# Write the stdlib approximation of GPT-2's pattern here (a compiled re.Pattern).
PRETOKENIZE_PATTERN: re.Pattern | None = None


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> character bijection, all 256 entries, values printable and never whitespace."""
    raise NotImplementedError("bytes_to_unicode() is unwritten")


def unicode_to_bytes() -> dict[str, int]:
    """The inverse mapping."""
    raise NotImplementedError("unicode_to_bytes() is unwritten")


def bytes_to_visible(data: bytes) -> str:
    """Raw bytes -> a string with one stand-in character per byte (same length as ``data``)."""
    raise NotImplementedError("bytes_to_visible() is unwritten")


def visible_to_bytes(text: str) -> bytes:
    """Undo ``bytes_to_visible``."""
    raise NotImplementedError("visible_to_bytes() is unwritten")


def pretokenize(text: str) -> list[str]:
    """``PRETOKENIZE_PATTERN.findall(text)``: pieces that concatenate back to ``text``."""
    raise NotImplementedError("pretokenize() is unwritten")
