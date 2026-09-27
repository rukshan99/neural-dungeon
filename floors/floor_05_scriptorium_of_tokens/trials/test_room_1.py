"""TRIAL 5.1 - THE CHARACTER CODEX

The scribe checks the numbering of the codex, that a page copied into numbers
comes back as the same page, and that a batch of ragged lines becomes a
rectangle with an honest mask.
"""

import pytest

from dungeon.artifacts.tiny_gpt import read_corpus
from dungeon.trials import load_room

room = load_room(__file__, "room_1_character_codex")

TRAINING_TEXT = "the scribe copies the chronicles by candlelight, page after page."


def _codex():
    return room.CharTokenizer.from_text(TRAINING_TEXT)


def test_the_red_ink_lines_come_first():
    tok = _codex()
    for tid, name in enumerate(["<pad>", "<unk>", "<bos>", "<eos>"]):
        assert tok.itos[tid] == name, (
            f"id {tid} should be {name}, yours is {tok.itos.get(tid)!r}. The four specials occupy ids 0..3."
        )
        assert tok.stoi[name] == tid
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3), (
        "pad_id, unk_id, bos_id, eos_id must be 0, 1, 2, 3."
    )


def test_the_ordinary_characters_follow_in_sorted_order():
    tok = _codex()
    expected_chars = sorted(set(TRAINING_TEXT))
    assert tok.chars == expected_chars, "chars must be the sorted distinct characters of the text."
    assert tok.vocab_size == 4 + len(expected_chars), (
        f"vocab_size should be 4 specials + {len(expected_chars)} characters, got {tok.vocab_size}."
    )
    for offset, ch in enumerate(expected_chars):
        assert tok.stoi[ch] == 4 + offset, (
            f"{ch!r} should have id {4 + offset} (4 + its sorted position), got {tok.stoi[ch]}."
        )
    assert tok.itos[4] == expected_chars[0], "itos must be the exact inverse of stoi."


def test_the_same_page_in_any_order_yields_the_same_codex():
    a = room.CharTokenizer.from_text("gamma beta alpha")
    b = room.CharTokenizer.from_text("alpha beta gamma")
    assert a.stoi == b.stoi, "The codex depends on the SET of characters, sorted; not on where they appear."


def test_encode_wraps_the_page_in_bos_and_eos():
    tok = _codex()
    ids = tok.encode("page")
    assert ids[0] == 2 and ids[-1] == 3, f"encode() should start with <bos>=2 and end with <eos>=3, got {ids}."
    assert len(ids) == len("page") + 2, f"'page' is 4 characters + 2 specials = 6 ids, got {len(ids)}."
    assert all(type(i) is int for i in ids), "encode() returns plain Python ints."
    bare = tok.encode("page", add_special=False)
    assert bare == ids[1:-1], "add_special=False should give exactly the inner ids, no <bos>/<eos>."
    assert bare == [tok.stoi[c] for c in "page"]


def test_a_copied_page_reads_back_identically():
    tok = _codex()
    assert tok.decode(tok.encode(TRAINING_TEXT)) == TRAINING_TEXT, (
        "Round trip failed on the training text. Every character in it has an id; decode must invert encode."
    )
    chronicles = read_corpus()
    scribe = room.CharTokenizer.from_text(chronicles)
    page = chronicles[2000:2600]
    assert scribe.decode(scribe.encode(page)) == page, "Round trip on a page of the Chronicles failed."
    assert scribe.vocab_size == 4 + len(set(chronicles))


def test_a_letter_the_scribe_never_saw_becomes_unk():
    tok = _codex()
    ids = tok.encode("paΩge", add_special=False)  # Omega is not in the training text
    assert ids[2] == 1, f"An unknown character must map to <unk>=1, you produced {ids[2]} for 'Ω'."
    assert ids[:2] == [tok.stoi["p"], tok.stoi["a"]], "Known characters around the unknown one must be unaffected."
    for ch in "ॐ字🙂":
        assert tok.encode(ch, add_special=False) == [1], f"{ch!r} is unknown to this codex and must encode to [1]."


def test_decode_strips_the_red_ink_unless_asked_not_to():
    tok = _codex()
    ids = [2, tok.stoi["h"], tok.stoi["i"], 1, 3, 0, 0]
    assert tok.decode(ids) == "hi", (
        f"decode() with skip_special=True drops ids 0..3 (pad, unk, bos, eos). Got {tok.decode(ids)!r}."
    )
    assert tok.decode(ids, skip_special=False) == "<bos>hi<unk><eos><pad><pad>", (
        "With skip_special=False the specials render as their strings."
    )
    assert tok.decode([]) == "", "Decoding nothing is the empty string."


def test_batch_encode_pads_on_the_right_to_the_longest_line():
    tok = _codex()
    texts = ["a", "abc", "ab"]
    batch = tok.batch_encode(texts)
    assert set(batch) == {"input_ids", "attention_mask"}, f"Keys must be input_ids and attention_mask, got {set(batch)}."
    ids, mask = batch["input_ids"], batch["attention_mask"]
    longest = len("abc") + 2
    assert [len(r) for r in ids] == [longest] * 3, (
        f"Every row must be padded to the longest sequence ({longest} incl. specials); rows are {[len(r) for r in ids]}."
    )
    assert [len(r) for r in mask] == [longest] * 3, "attention_mask must have the same shape as input_ids."
    assert ids[0] == [2, tok.stoi["a"], 3, 0, 0], f"Row 0 should be <bos> a <eos> <pad> <pad>, got {ids[0]}."
    assert mask[0] == [1, 1, 1, 0, 0], f"Mask row 0 should be 1 over the 3 real ids then 0s, got {mask[0]}."
    assert mask[1] == [1] * longest, "The longest line needs no padding: its mask is all ones."
    for row_ids, row_mask, text in zip(ids, mask, texts):
        assert sum(row_mask) == len(text) + 2, "sum(mask) must equal the number of real ids (specials included)."
        assert all(i == 0 for i, m in zip(row_ids, row_mask) if m == 0), "Every masked-out position must hold <pad>=0."
        assert row_mask == sorted(row_mask, reverse=True), "Padding goes on the RIGHT: the mask is 1s then 0s."


def test_batch_encode_truncates_but_keeps_the_eos():
    tok = _codex()
    batch = tok.batch_encode(["candlelight", "page"], max_len=6)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert len(ids[0]) == 6, f"max_len=6 means no row longer than 6, got {len(ids[0])}."
    assert ids[0][0] == 2 and ids[0][-1] == 3, (
        f"A truncated sequence must still begin with <bos> and END with <eos>, got {ids[0]}."
    )
    assert ids[0][1:5] == [tok.stoi[c] for c in "cand"], "Keep the first max_len - 2 characters, then <eos>."
    assert mask[0] == [1] * 6
    assert ids[1] == [2, *[tok.stoi[c] for c in "page"], 3], "'page' fits in 6 and must be untouched."


def test_batch_encode_can_pad_to_a_fixed_width():
    tok = _codex()
    batch = tok.batch_encode(["a", "ab"], max_len=8, pad_to_longest=False)
    assert [len(r) for r in batch["input_ids"]] == [8, 8], (
        "pad_to_longest=False pads every row to exactly max_len, so that all batches share one shape."
    )
    assert batch["attention_mask"][1] == [1, 1, 1, 1, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        tok.batch_encode(["a"], pad_to_longest=False)
    assert tok.batch_encode([]) == {"input_ids": [], "attention_mask": []}


def test_to_tensors_makes_a_long_rectangle():
    torch = pytest.importorskip(
        "torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    tok = _codex()
    tensors = room.to_tensors(tok.batch_encode(["a", "abc"]))
    assert set(tensors) == {"input_ids", "attention_mask"}
    for key, t in tensors.items():
        assert isinstance(t, torch.Tensor), f"{key} should be a torch.Tensor, got {type(t).__name__}."
        assert t.dtype == torch.long, f"{key} must be int64 (torch.long) so it can index an embedding; got {t.dtype}."
        assert tuple(t.shape) == (2, 5), f"{key} should be (B, T) = (2, 5), got {tuple(t.shape)}."
    assert int(tensors["attention_mask"].sum()) == 3 + 5
    try:
        decoded = tok.decode(tensors["input_ids"][1])
    except (KeyError, TypeError) as exc:
        pytest.fail(
            f"decode() choked on a row of the tensor it just produced ({type(exc).__name__}: {exc}). "
            "Torch scalars are not dict keys; call int(i) on every id first."
        )
    assert decoded == "abc", f"decode() of the tensor row for 'abc' gave {decoded!r}: skip the specials and the padding."
