"""TRIAL 9.3 - THE SCRIBE'S CHUNKS

Every card must trace back to its page: text[start:end] == chunk.text, always.
Then overlap exactness, sentence boundaries, duplicates and corpus-wide ids.
"""

import re

import pytest

from dungeon.trials import load_room
from floors.floor_09_library_of_echoes.assets.corpus import PASSAGES, passage_by_id

room = load_room(__file__, "room_3_scribes_chunks")

SCROLL = (
    "The Basilisk waits at the exit. Its gaze petrifies helpers! Do you know the four rules? "
    "Then walk across. A sign on the far side reads: no loops. The bridge sways when you disobey."
)
SENTENCE = re.compile(r"[^.!?]*[.!?]+(?=\s|$)")


# ------------------------------------------------------------------ chunk_text
@pytest.mark.parametrize("size,overlap", [(40, 10), (25, 0), (30, 29), (7, 3)], ids=["40/10", "25/0", "30/29", "7/3"])
def test_every_card_traces_back_to_its_page(size, overlap):
    chunks = room.chunk_text(SCROLL, size, overlap, doc_id="scroll")
    assert chunks, "A non-empty text must produce at least one chunk."
    for i, c in enumerate(chunks):
        assert isinstance(c, room.Chunk), f"chunk_text must return Chunk objects, got {type(c).__name__}."
        assert c.doc_id == "scroll" and c.index == i, f"Chunk {i} has doc_id={c.doc_id!r} index={c.index}; expected 'scroll' and {i}."
        assert SCROLL[c.start : c.end] == c.text, (
            f"Chunk {i}: text[{c.start}:{c.end}] is {SCROLL[c.start:c.end]!r} but chunk.text is {c.text!r}. Offsets must be exact."
        )
        assert len(c.text) <= size, f"Chunk {i} has {len(c.text)} characters, more than size={size}."


@pytest.mark.parametrize("size,overlap", [(40, 10), (30, 29), (7, 3)], ids=["40/10", "30/29", "7/3"])
def test_consecutive_cards_overlap_by_exactly_the_overlap(size, overlap):
    chunks = room.chunk_text(SCROLL, size, overlap)
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end - cur.start == overlap, (
            f"Chunks {prev.index} and {cur.index} overlap by {prev.end - cur.start} characters, not {overlap}. "
            f"Each chunk should start size - overlap = {size - overlap} after the previous one."
        )
        if overlap:
            assert cur.text[:overlap] == prev.text[-overlap:], "The shared characters must be identical on both cards."


@pytest.mark.parametrize("size,overlap", [(40, 10), (25, 0), (7, 3)], ids=["40/10", "25/0", "7/3"])
def test_the_non_overlapping_parts_reassemble_the_page(size, overlap):
    chunks = room.chunk_text(SCROLL, size, overlap)
    rebuilt = chunks[0].text + "".join(c.text[overlap:] for c in chunks[1:])
    assert rebuilt == SCROLL, (
        "Taking the first chunk whole and dropping the first `overlap` characters of every later chunk must give back "
        f"the exact text. Got {len(rebuilt)} characters vs {len(SCROLL)}; are you skipping or repeating characters?"
    )


def test_the_last_card_may_be_short_but_never_redundant():
    chunks = room.chunk_text(SCROLL, 40, 10)
    assert chunks[-1].end == len(SCROLL), f"The last chunk must reach the end of the text ({len(SCROLL)}), it ends at {chunks[-1].end}."
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.end > prev.end, f"Chunk {cur.index} ends at {cur.end}, inside chunk {prev.index} (ends {prev.end}). Stop once a chunk reaches the end."
    exact_fit = "a" * 60
    fit = room.chunk_text(exact_fit, 30, 0)
    assert len(fit) == 2, f"60 characters in windows of 30 with no overlap is 2 chunks, not {len(fit)}. Do not emit an empty trailing chunk."


def test_empty_pages_and_impossible_overlaps():
    assert room.chunk_text("", 10, 2) == [], "Empty text, no chunks."
    with pytest.raises(ValueError):
        room.chunk_text(SCROLL, 10, 10)
    with pytest.raises(ValueError):
        room.chunk_text(SCROLL, 10, 12)
    with pytest.raises(ValueError):
        room.chunk_text(SCROLL, 0, 0)


def test_card_ids_are_unique_within_a_page():
    chunks = room.chunk_text(SCROLL, 20, 5, doc_id="scroll")
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids), f"Duplicate chunk ids: {ids}"
    assert ids[0] == "scroll#0", f"Chunk.id is '<doc_id>#<index>'; the first should be 'scroll#0', got {ids[0]!r}."


# ---------------------------------------------------------- chunk_by_sentences
@pytest.mark.parametrize("max_chars", [60, 90, 200, 10], ids=["60", "90", "200", "10"])
def test_sentence_cards_never_cut_a_thought_in_half(max_chars):
    chunks = room.chunk_by_sentences(SCROLL, max_chars, doc_id="scroll")
    assert chunks, "A non-empty text must produce at least one chunk."
    for c in chunks:
        assert SCROLL[c.start : c.end] == c.text, f"text[{c.start}:{c.end}] != chunk.text: {c.text!r}"
        assert c.text[-1] in ".!?", f"Chunk {c.index} ends mid-sentence: {c.text!r}"
        assert c.text == c.text.strip(), f"Chunk {c.index} carries leading/trailing whitespace: {c.text!r}"
    for i, c in enumerate(chunks):
        assert c.index == i and c.doc_id == "scroll"


@pytest.mark.parametrize("max_chars", [60, 90, 200], ids=["60", "90", "200"])
def test_sentence_cards_respect_max_chars_unless_one_sentence_is_simply_too_long(max_chars):
    for c in room.chunk_by_sentences(SCROLL, max_chars):
        n_sentences = len(SENTENCE.findall(c.text))
        assert len(c.text) <= max_chars or n_sentences == 1, (
            f"Chunk {c.index} has {len(c.text)} characters (> {max_chars}) and {n_sentences} sentences. "
            "Only a single over-long sentence may exceed max_chars."
        )


def test_an_over_long_sentence_stands_alone():
    chunks = room.chunk_by_sentences(SCROLL, 10)
    sentences = [s.strip() for s in SENTENCE.findall(SCROLL)]
    assert [c.text for c in chunks] == sentences, (
        "With max_chars smaller than every sentence, each sentence becomes its own chunk, whole, in order."
    )


@pytest.mark.parametrize("max_chars", [60, 90, 120], ids=["60", "90", "120"])
def test_sentence_cards_are_packed_greedily(max_chars):
    chunks = room.chunk_by_sentences(SCROLL, max_chars)
    for cur, nxt in zip(chunks, chunks[1:]):
        first_next_sentence = SENTENCE.search(SCROLL, nxt.start)
        would_be = first_next_sentence.end() - cur.start
        assert would_be > max_chars, (
            f"Chunk {cur.index} ({len(cur.text)} chars) could have taken the next sentence too ({would_be} <= {max_chars}). "
            "Pack greedily: add sentences while the chunk still fits."
        )


def test_sentence_cards_cover_every_sentence_exactly_once():
    chunks = room.chunk_by_sentences(SCROLL, 70)
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(SCROLL.split()), (
        "Concatenating the chunks (ignoring whitespace) must give back the whole text: nothing dropped, nothing repeated."
    )
    assert room.chunk_by_sentences("", 50) == []


# --------------------------------------------------------------------- dedupe
def _chunk(doc, idx, text):
    return room.Chunk(doc, idx, 0, len(text), text)


def test_dedupe_drops_exact_and_whitespace_only_duplicates():
    chunks = [
        _chunk("a", 0, "The Basilisk waits."),
        _chunk("b", 0, "The  Basilisk waits. "),
        _chunk("c", 0, "the basilisk waits."),
        _chunk("d", 0, "The Basilisk sleeps."),
        _chunk("a", 1, "The Basilisk waits."),
    ]
    out = room.dedupe(chunks)
    assert [c.id for c in out] == ["a#0", "d#0"], (
        f"Expected to keep a#0 (first of the duplicates) and d#0 (different text); kept {[c.id for c in out]}. "
        "Normalise as lowercase with whitespace collapsed before comparing."
    )
    assert out[0] is chunks[0], "Keep the first occurrence, unchanged."


def test_dedupe_keeps_the_order_of_first_appearance():
    chunks = [_chunk("x", i, t) for i, t in enumerate(["c", "a", "b", "a", "c", "d"])]
    assert [c.text for c in room.dedupe(chunks)] == ["c", "a", "b", "d"]


# --------------------------------------------------------------- chunk_corpus
def test_every_card_in_the_library_has_a_unique_id():
    chunks = room.chunk_corpus(PASSAGES, size=120, overlap=20)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids), f"{len(ids) - len(set(ids))} duplicate chunk ids across the corpus. Use the passage id as doc_id."
    assert {c.doc_id for c in chunks} == {p.id for p in PASSAGES}, "Every passage should contribute chunks, under its own id."
    assert len(chunks) > len(PASSAGES), "Passages longer than 120 characters must produce more than one chunk each."


def test_every_card_in_the_library_traces_back_to_its_passage():
    chunks = room.chunk_corpus(PASSAGES, size=120, overlap=20)
    for c in chunks:
        source = passage_by_id(c.doc_id).text
        assert source[c.start : c.end] == c.text, f"{c.id}: offsets do not point at the chunk's text inside passage {c.doc_id}."
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for pid, cs in by_doc.items():
        cs.sort(key=lambda c: c.index)
        rebuilt = cs[0].text + "".join(c.text[20:] for c in cs[1:])
        assert rebuilt == passage_by_id(pid).text, f"Chunks of {pid} do not reassemble the passage."
