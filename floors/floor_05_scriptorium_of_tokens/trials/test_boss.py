"""BOSS FIGHT - THE BABEL GOLEM

Phase 1: the base vocabulary and the mechanics of merging bytes.
Phase 2: a battery of strings from every script comes back byte for byte.
Phase 3: the Golem shows the Character Codex a letter it has never seen.
Phase 4: merges must actually compress the Chronicles.
"""

import pytest

from dungeon.artifacts.tiny_gpt import read_corpus
from dungeon.trials import load_room

golem = load_room(__file__, "boss_babel_golem")
codex = load_room(__file__, "room_1_character_codex")

pytestmark = pytest.mark.boss

TRAIN_VOCAB = 512
COMPRESSION_BAR = 2.0  # reference solution: ~2.4 bytes/token on the training page at vocab 512

BATTERY = {
    "plain": "The scribe copies the Chronicles by candlelight.",
    "greek": "Ὀδυσσεύς ἐπὶ τῆς θαλάσσης",
    "devanagari": "नमस्ते दुनिया, क्या हाल है?",
    "cjk": "東京の図書館で本を読む。",
    "hangul": "도서관에서 책을 읽다",
    "arabic": "مرحبا بالعالم، كيف حالك؟",
    "hebrew": "שלום עולם",
    "zwj_emoji": "👩‍💻 👨‍👩‍👧‍👦 🏳️‍🌈",
    "flags": "🇯🇵🇧🇷🇰🇪",
    "combining": "é ä Z̵̢a̶l̸g̕o̵",
    "astral": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𐐀𐐁",
    "digits": "٣.١٤١٥ १२३ ４２",
    "whitespace": "  two  spaces\tand\ta tab\nand\r\nline ends  ",
    "long_repeat": "a" * 3000,
    "long_repeat_multibyte": "ñ" * 1500,
    "unseen_words": "quokka xylophone zeitgeist syzygy",
    "empty": "",
    "single_space": " ",
    "mixed": "café naïve ☕ 日本語 — résumé 2026",
}


@pytest.fixture(scope="module")
def page():
    return read_corpus()[:16000]


@pytest.fixture(scope="module")
def trained(page):
    tok = golem.ByteLevelBPE()
    tok.train(page, TRAIN_VOCAB)
    return tok


# ================================================================== phase 1
def test_phase_1_pieces_concatenate_back_to_the_text():
    for text in BATTERY.values():
        pieces = golem.pretokenize(text)
        assert "".join(pieces) == text, f"Pre-tokenization must be lossless; pieces {pieces!r} do not rebuild {text!r}."
    assert golem.pretokenize("the  scribe ") == ["the", "  scribe", " "], (
        "Leading whitespace sticks to the following word; a trailing run stands alone."
    )


def test_phase_1_merge_ids_binds_bytes_left_to_right():
    assert golem.merge_ids([1, 2, 1, 2, 3], (1, 2), 300) == [300, 300, 3]
    assert golem.merge_ids([7, 7, 7], (7, 7), 256) == [256, 7], "Non-overlapping, left to right: (7 7 7) -> (256 7)."
    assert golem.merge_ids([1, 2, 3], (9, 9), 256) == [1, 2, 3], "An absent pair changes nothing."
    assert golem.merge_ids([], (1, 2), 256) == []


def test_phase_1_an_untrained_golem_speaks_in_bytes():
    tok = golem.ByteLevelBPE()
    assert tok.vocab_size == 256, f"Before training the vocabulary is exactly the 256 byte values, got {tok.vocab_size}."
    assert tok.vocab[0x41] == b"A" and tok.vocab[0xFF] == b"\xff", "vocab[i] must be the single byte bytes([i])."
    assert tok.encode("Hi") == [72, 105], f"With no merges, encode() is just the UTF-8 bytes: 'Hi' -> [72, 105], got {tok.encode('Hi')}."
    assert tok.encode("α") == [0xCE, 0xB1], f"'α' is two UTF-8 bytes 0xCE 0xB1; got {tok.encode('α')}."
    assert tok.decode([72, 105]) == "Hi"
    assert tok.unk_id is None, "A byte-level tokenizer has no unknown token: unk_id must be None."
    assert tok.merges == [] and tok.ranks == {}


def test_phase_1_training_fills_the_vocabulary_with_byte_strings(trained):
    assert trained.vocab_size == TRAIN_VOCAB, f"train(text, {TRAIN_VOCAB}) must leave exactly {TRAIN_VOCAB} ids, got {trained.vocab_size}."
    assert len(trained.merges) == TRAIN_VOCAB - 256, f"{TRAIN_VOCAB - 256} merges expected, got {len(trained.merges)}."
    for k, (a, b) in enumerate(trained.merges):
        assert trained.vocab[256 + k] == trained.vocab[a] + trained.vocab[b], (
            f"Merge {k} = {(a, b)} must define vocab[{256 + k}] as vocab[{a}] + vocab[{b}]."
        )
        assert trained.ranks[(a, b)] == k, "ranks maps each merged pair to its position in merges."
    assert any(b"th" in v for v in trained.vocab.values()), "Trained on English, some token should contain b'th'."
    assert all(len(v) >= 2 for i, v in trained.vocab.items() if i >= 256), "Every merged id stands for at least two bytes."


def test_phase_1_merges_are_learned_most_frequent_first():
    tok = golem.ByteLevelBPE()
    tok.train("aaaa bbb", 257)
    assert tok.merges == [(97, 97)], (
        f"'aaaa' holds three adjacent (a, a) pairs and ' bbb' two (b, b) pairs; expected [(97, 97)], got {tok.merges}."
    )
    tok.train("ab ab ab cd", 258)
    assert tok.merges[0] == (97, 98), f"(a, b) appears 3 times and must be merged first; got {tok.merges[0]}."
    assert tok.merges[1] == (32, 256), (
        f"After 'ab' became id 256, the pair (space, 256) appears twice and beats (c, d) at once; got {tok.merges[1]}. "
        "Whitespace is part of the pieces, so ' ab' becomes a token of its own."
    )
    tok.train("cba", 257)
    assert tok.merges == [(98, 97)], (
        f"'cba' holds (c, b) and (b, a) once each. The tie goes to the smallest pair, (98, 97) = (b, a), not to the "
        f"first one counted; got {tok.merges}."
    )


def test_phase_1_training_is_deterministic_and_starts_fresh(page, trained):
    again = golem.ByteLevelBPE()
    again.train(page, TRAIN_VOCAB)
    assert again.merges == trained.merges, "Two trainings on the same page must learn the same ledger."
    again.train(page[:2000], 300)
    assert again.vocab_size == 300, "train() must start from the 256 base ids, not stack onto the previous vocabulary."
    with pytest.raises(ValueError):
        again.train(page, 100)


def test_phase_1_encoding_merges_by_rank(trained, page):
    ids = trained.encode(page[:3000])
    assert all(0 <= i < trained.vocab_size for i in ids), "Every id must be inside the vocabulary."
    assert len(ids) < len(page[:3000].encode("utf-8")), "Merges must reduce the token count below the byte count."
    # Whatever the merge order, re-encoding a decoded token must be the same single token.
    for token_id in (300, 400, 511):
        piece = trained.vocab[token_id]
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert trained.encode(text) == [token_id], (
            f"vocab[{token_id}] = {piece!r} decodes to {text!r}, which should encode straight back to [{token_id}]. "
            "Merge the LOWEST-ranked present pair first, repeatedly, until nothing applies."
        )


# ================================================================== phase 2
@pytest.mark.parametrize("name", list(BATTERY), ids=list(BATTERY))
def test_phase_2_every_script_comes_back_byte_for_byte(trained, name):
    text = BATTERY[name]
    ids = trained.encode(text)
    back = trained.decode(ids)
    assert back == text, (
        f"Round trip failed for {name!r}: {text!r} -> {len(ids)} ids -> {back!r}. "
        "Bytes in, bytes out: no character can be unknown, so nothing may be dropped or replaced."
    )
    assert len(ids) <= len(text.encode("utf-8")), "A byte-level tokenizer never needs more ids than there are bytes."
    assert all(type(i) is int for i in ids), "encode() returns plain Python ints."


def test_phase_2_the_golem_never_shatters_on_the_unseen(trained):
    unseen = " ".join(BATTERY[k] for k in ("greek", "devanagari", "cjk", "arabic", "zwj_emoji"))
    ids = trained.encode(unseen)
    assert trained.decode(ids) == unseen
    assert max(ids) < trained.vocab_size and min(ids) >= 0


def test_phase_2_a_lone_fragment_of_a_character_decodes_to_the_replacement_mark(trained):
    assert trained.decode([0xCE]) == "�", (
        f"0xCE is the first byte of 'α' with nothing after it. decode() must return U+FFFD, got {trained.decode([0xCE])!r}. "
        "Use errors='replace'."
    )
    assert trained.decode([0xCE, 0xB1]) == "α", "The two bytes together are a complete 'α'."
    assert "�" in trained.decode([0xE2, 0x82]), "An incomplete 3-byte sequence also decodes to the replacement mark, not an exception."
    for byte in range(128):
        assert trained.decode([byte]) == chr(byte), "Every ASCII byte decodes alone to its own character."
    for byte in range(128, 256):
        assert trained.decode([byte]) == "�", f"Byte {byte:#x} alone is never valid UTF-8 and must decode to U+FFFD."
    for token_id, piece in trained.vocab.items():
        decoded = trained.decode([token_id])
        assert isinstance(decoded, str), "decode() must return a str for every single id, complete sequence or not."
        if "�" not in decoded:
            assert decoded.encode("utf-8") == piece, "A cleanly decoded token must re-encode to exactly its bytes."


# ================================================================== phase 3
def test_phase_3_the_codex_shatters_and_the_golem_does_not(trained, page):
    scribe = codex.CharTokenizer.from_text(page)
    for name in ("greek", "devanagari", "cjk", "arabic", "zwj_emoji", "flags"):
        text = BATTERY[name]
        rate_codex = golem.unknown_rate(scribe, text)
        rate_golem = golem.unknown_rate(trained, text)
        assert rate_codex > 0.0, (
            f"The Character Codex was trained on English; {name!r} must produce <unk> ids, but unknown_rate says {rate_codex}."
        )
        assert rate_golem == 0.0, f"Nothing is unknown to a byte-level tokenizer, yet unknown_rate on {name!r} is {rate_golem}."
    assert golem.unknown_rate(scribe, page[:500]) == 0.0, "On its own training text the Codex knows every character."


def test_phase_3_unknown_rate_is_a_fraction_of_ids():
    class Half:
        unk_id = 1

        def encode(self, text):
            return [1, 5, 1, 7]

    assert golem.unknown_rate(Half(), "anything") == pytest.approx(0.5), "2 of 4 ids are <unk>: the rate is 0.5."

    class NoUnk:
        def encode(self, text):
            return [1, 2, 3]

    assert golem.unknown_rate(NoUnk(), "anything") == 0.0, "A tokenizer without an unk_id attribute has an unknown rate of 0.0."

    class Empty:
        unk_id = 1

        def encode(self, text):
            return []

    assert golem.unknown_rate(Empty(), "") == 0.0, "No ids at all: 0.0, not a ZeroDivisionError."


# ================================================================== phase 4
def test_phase_4_merges_buy_compression(trained, page):
    untrained = golem.ByteLevelBPE()
    assert golem.compression_ratio(untrained, page) == pytest.approx(1.0), (
        "One id per byte: an untrained byte tokenizer scores exactly 1.0 bytes per token."
    )
    ratio = golem.compression_ratio(trained, page)
    assert ratio > COMPRESSION_BAR, (
        f"At vocab {TRAIN_VOCAB} the Golem should fit more than {COMPRESSION_BAR} bytes into each token on its training page; "
        f"yours manages {ratio:.3f}. Are the merges applied when encoding? Is the most frequent pair merged first?"
    )
    scribe = codex.CharTokenizer.from_text(page)
    assert ratio > golem.compression_ratio(scribe, page), "One-token-per-character cannot beat merged byte pairs on bytes per token."


def test_phase_4_compression_generalizes_to_an_unseen_page(trained):
    other = read_corpus()[100000:116000]
    ratio = golem.compression_ratio(trained, other)
    assert ratio > 1.6, f"On a page it never saw the Golem should still beat 1.6 bytes/token; got {ratio:.3f}."
    assert golem.compression_ratio(trained, "") == 0.0, "No tokens, no ratio: return 0.0 for empty text."
