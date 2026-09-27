"""SECRET - THE MERCHANT'S ALPHABET

256 bytes, 256 printable stand-ins, and a regular expression that cuts text
into pieces without losing a single character.
"""

import re

import pytest

from dungeon.trials import load_room

ledger = load_room(__file__, "secret_merchants_alphabet")

pytestmark = pytest.mark.secret

SAMPLES = [
    "Hello world's 42!",
    "the scribe   copies  the chronicles",
    "snake_case_name and __dunder__",
    "C++11, C#, F#: 3.14 vs 2,718",
    "don't you'll we've I'm they're he'd it's",
    "Ὀδυσσεύς ἐπὶ τῆς θαλάσσης",
    "नमस्ते दुनिया",
    "東京の図書館で本を読む。",
    "مرحبا بالعالم",
    "👩‍💻 👨‍👩‍👧‍👦 🇯🇵",
    "é café",
    "tabs\tand\nnewlines\r\n  trailing   ",
    "٣.١٤ १२३ ４２",
    "",
    " ",
    "\x00\x01\x7f\x80\xa0\xad\xff",
]


def _table():
    table = ledger.bytes_to_unicode()
    if not isinstance(table, dict):
        raise NotImplementedError("bytes_to_unicode() is unwritten")
    return table


def test_every_byte_has_exactly_one_printable_letter():
    table = _table()
    assert set(table) == set(range(256)), f"The alphabet needs all 256 byte values as keys; you have {len(table)}."
    values = list(table.values())
    assert len(set(values)) == 256, "The mapping must be a bijection: 256 distinct characters."
    for b, ch in table.items():
        assert isinstance(ch, str) and len(ch) == 1, f"Byte {b} maps to {ch!r}; each value is a single character."
        assert ch.isprintable() and not ch.isspace(), (
            f"Byte {b:#04x} maps to {ch!r}, which is not printable or is whitespace. The whole point is a vocabulary you can write down."
        )


def test_printable_bytes_keep_their_own_letter():
    table = _table()
    for b in list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)):
        assert table[b] == chr(b), f"Byte {b:#04x} is already printable in Latin-1 and must map to itself, not {table[b]!r}."


def test_the_awkward_bytes_climb_to_u0100_in_order():
    table = _table()
    assert table[0] == "Ā", f"Byte 0 is the first non-printable byte and maps to U+0100 'Ā'; got {table[0]!r}."
    assert table[32] == "Ġ", f"Space (0x20) is the 33rd awkward byte and maps to U+0120 'Ġ'; got {table[32]!r}. Yes, THAT Ġ."
    assert table[127] == "ġ" and table[128] == "Ģ", "DEL and 0x80 follow space at U+0121 and U+0122."
    assert table[173] == "Ń", f"Soft hyphen 0xAD is the last awkward byte and maps to U+0143; got {table[173]!r}."
    awkward = sorted(b for b in range(256) if table[b] != chr(b))
    assert len(awkward) == 68, f"Exactly 68 bytes need a stand-in; you moved {len(awkward)}."
    assert [table[b] for b in awkward] == [chr(256 + i) for i in range(68)], "Stand-ins are assigned in increasing byte order."


def test_the_inverse_undoes_the_alphabet():
    table = _table()
    inverse = ledger.unicode_to_bytes()
    assert inverse == {ch: b for b, ch in table.items()}, "unicode_to_bytes must be the exact inverse of bytes_to_unicode."
    assert all(inverse[table[b]] == b for b in range(256))


@pytest.mark.parametrize("text", SAMPLES, ids=[f"sample{i}" for i in range(len(SAMPLES))])
def test_bytes_become_visible_and_back(text):
    data = text.encode("utf-8", errors="surrogatepass")
    visible = ledger.bytes_to_visible(data)
    assert len(visible) == len(data), f"One stand-in per byte: {len(data)} bytes should give {len(data)} characters, got {len(visible)}."
    assert visible.isprintable() or visible == "", f"The visible form must be printable end to end; got {visible!r}."
    assert ledger.visible_to_bytes(visible) == data, f"visible_to_bytes(bytes_to_visible(x)) must be x; failed for {text!r}."


def test_a_space_before_a_word_shows_as_g_with_a_dot():
    assert ledger.bytes_to_visible(b" the") == "Ġthe", "b' the' renders as 'Ġthe', the prefix you see all over GPT-2's vocab.json."


def _pattern():
    pattern = ledger.PRETOKENIZE_PATTERN
    if pattern is None:
        raise NotImplementedError("PRETOKENIZE_PATTERN is unwritten")
    return re.compile(pattern)


@pytest.mark.parametrize("text", SAMPLES, ids=[f"sample{i}" for i in range(len(SAMPLES))])
def test_pieces_concatenate_back_to_the_text(text):
    pieces = ledger.pretokenize(text)
    assert isinstance(pieces, list)
    assert "".join(pieces) == text, (
        f"Pre-tokenization dropped or duplicated characters: {pieces!r} does not rebuild {text!r}. "
        "Every character must match some branch of the pattern (underscore is the usual culprit)."
    )
    assert all(pieces), "No empty pieces."


def test_words_numbers_punctuation_and_spaces_are_cut_apart():
    assert ledger.pretokenize("Hello world's 42!") == ["Hello", " world", "'s", " 42", "!"], (
        "GPT-2 style: a leading space sticks to the word, contractions split off, digits are their own piece."
    )
    assert ledger.pretokenize("a  b") == ["a", " ", " b"], (
        "Of two spaces, the second sticks to 'b' and the first stands alone: that is what \\s+(?!\\S) does."
    )
    assert ledger.pretokenize("x_y") == ["x", "_", "y"], "Underscore is not a letter here; it must still match something."
    assert ledger.pretokenize("東京の図書館") == ["東京の図書館"], "CJK characters are word characters to re, so they stay together."
    assert ledger.pretokenize("٣.١٤") == ["٣", ".", "١٤"], "\\d matches Unicode digits, so Arabic-Indic digits are numbers."


def test_the_pattern_is_a_stdlib_regex_with_the_five_gpt2_contractions():
    pattern = _pattern()
    assert isinstance(pattern, re.Pattern)
    text = pattern.pattern
    for contraction in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d"):
        assert contraction in text, f"GPT-2's pattern splits off {contraction!r}; yours does not mention it."
    assert "\\p{" not in text, "\\p{...} classes need the third-party regex module; this room is stdlib re only."
