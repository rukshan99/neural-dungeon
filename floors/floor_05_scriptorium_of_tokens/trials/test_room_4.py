"""TRIAL 5.4 - TOKEN ECONOMICS

The bursar audits your counting, your pricing, your cuts and your windows.
Then reads your prophecy about the price of a few short strings.
"""

import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_4_token_economics")
codex = load_room(__file__, "room_1_character_codex")


class WordTokenizer:
    """A stand-in tokenizer: one id per whitespace-separated word, no specials."""

    def encode(self, text):
        return [hash(w) % 1000 for w in text.split()]


def by_words(text):
    return len(text.split())


# --------------------------------------------------------------------- counting
def test_counting_defers_to_the_tokenizer():
    assert room.count_tokens("one two three", WordTokenizer()) == 3
    assert room.count_tokens("", WordTokenizer()) == 0
    tok = codex.CharTokenizer.from_text("the scribe")
    assert room.count_tokens("the", tok) == 5, (
        "The Character Codex adds <bos> and <eos>: 'the' costs 3 + 2 = 5 ids, and the specials DO occupy context."
    )


def test_the_price_of_a_call():
    cost = room.estimate_cost(1_000_000, 0, 3.0, 15.0)
    assert cost == pytest.approx(3.0), f"A million input tokens at 3.0 per million cost 3.0, you charged {cost}."
    cost = room.estimate_cost(0, 500_000, 3.0, 15.0)
    assert cost == pytest.approx(7.5), f"Half a million output tokens at 15.0 per million cost 7.5, you charged {cost}."
    cost = room.estimate_cost(1200, 300, 3.0, 15.0)
    assert cost == pytest.approx(0.0036 + 0.0045), (
        f"1200 in at 3.0/M is 0.0036, 300 out at 15.0/M is 0.0045: total 0.0081, you charged {cost}."
    )
    assert room.estimate_cost(0, 0, 3.0, 15.0) == 0.0
    assert isinstance(room.estimate_cost(1, 1, 1.0, 1.0), float)


# ------------------------------------------------------------------- truncation
CONVERSATION = [
    {"role": "system", "content": "you are the bursar"},  # 4 words
    {"role": "user", "content": "how much for one page"},  # 5 words
    {"role": "assistant", "content": "a coin a token"},  # 4 words
    {"role": "user", "content": "and two pages"},  # 3 words
    {"role": "assistant", "content": "twice that"},  # 2 words
]


def test_a_conversation_that_fits_is_untouched():
    out = room.truncate_messages(CONVERSATION, budget=18, count_fn=by_words)
    assert out == CONVERSATION, "18 words of budget for 18 words of conversation: nothing should be dropped."
    assert out is not CONVERSATION, "Return a new list rather than the input object."


def test_the_oldest_exchange_is_forgotten_first():
    out = room.truncate_messages(CONVERSATION, budget=13, count_fn=by_words)
    assert out == [CONVERSATION[0], *CONVERSATION[2:]], (
        f"Dropping the oldest non-system message (5 words) brings 18 down to 13. Got {[m['content'] for m in out]}."
    )
    out = room.truncate_messages(CONVERSATION, budget=9, count_fn=by_words)
    assert out == [CONVERSATION[0], *CONVERSATION[3:]], (
        f"With budget 9, the two oldest user/assistant messages go and the system message stays. Got {[m['content'] for m in out]}."
    )


def test_the_system_message_survives_even_when_alone():
    out = room.truncate_messages(CONVERSATION, budget=4, count_fn=by_words)
    assert out == [CONVERSATION[0]], "Only the system message fits in 4 words, so only it remains."
    out = room.truncate_messages(CONVERSATION, budget=2, count_fn=by_words)
    assert out == [CONVERSATION[0]], (
        "The system message alone exceeds the budget; return it anyway rather than splitting or dropping it."
    )


def test_the_system_message_is_ordinary_when_asked():
    out = room.truncate_messages(CONVERSATION, budget=9, count_fn=by_words, keep_system=False)
    assert out == CONVERSATION[2:], (
        f"With keep_system=False the system message is just the oldest message and goes first. Got {[m['content'] for m in out]}."
    )


def test_truncation_never_splits_and_never_reorders():
    original = [dict(m) for m in CONVERSATION]
    out = room.truncate_messages(CONVERSATION, budget=10, count_fn=by_words)
    assert CONVERSATION == original, "Do not mutate the messages you were given."
    assert all(m in CONVERSATION for m in out), "Every surviving message must be an original message, unedited."
    positions = [CONVERSATION.index(m) for m in out]
    assert positions == sorted(positions), "Survivors keep their original order."
    assert sum(by_words(m["content"]) for m in out) <= 10
    assert len(out) == 3, (
        "Dropping the two oldest turns (5 then 4 words) brings 18 down to 9, which fits 10. "
        f"Do not drop a third. You kept {len(out)} messages."
    )


def test_a_late_system_message_is_also_protected():
    convo = [
        {"role": "user", "content": "a b c"},
        {"role": "system", "content": "s s"},
        {"role": "user", "content": "d"},
    ]
    out = room.truncate_messages(convo, budget=3, count_fn=by_words)
    assert out == convo[1:], "Protection is by role, not by position: the system message in the middle stays."


# --------------------------------------------------------------------- chunking
def _check_windows(ids, chunks, max_tokens, overlap):
    assert chunks, "At least one window for a non-empty input."
    assert all(len(c) == max_tokens for c in chunks[:-1]), (
        f"Every window but the last must have exactly {max_tokens} ids; lengths are {[len(c) for c in chunks]}."
    )
    assert 0 < len(chunks[-1]) <= max_tokens, "The last window may be shorter but never empty."
    for a, b in zip(chunks, chunks[1:]):
        assert a[len(a) - overlap :] == b[:overlap] if overlap else True, (
            f"Consecutive windows must share exactly {overlap} ids: {a} then {b}."
        )
    flattened = list(chunks[0])
    for c in chunks[1:]:
        flattened.extend(c[overlap:])
    assert flattened == list(ids), (
        "Stitching the windows back together (skipping each overlap) must reproduce the input exactly: "
        f"got {flattened} for {list(ids)}."
    )


def test_windows_cover_the_scroll_with_exact_overlap():
    ids = list(range(10))
    chunks = room.chunk_by_tokens(ids, max_tokens=4, overlap=1)
    assert chunks == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]], (
        f"Windows of 4 stepping by 3 over 0..9 are [0-3], [3-6], [6-9]; got {chunks}."
    )
    _check_windows(ids, chunks, 4, 1)


def test_the_last_window_may_be_short_but_still_new():
    ids = list(range(11))
    chunks = room.chunk_by_tokens(ids, max_tokens=4, overlap=1)
    assert chunks[-1] == [9, 10], f"The final window holds the leftover ids [9, 10]; got {chunks[-1]}."
    _check_windows(ids, chunks, 4, 1)
    chunks = room.chunk_by_tokens(list(range(7)), max_tokens=3, overlap=2)
    assert chunks == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], (
        f"overlap=2 with max_tokens=3 steps by one id at a time; got {chunks}."
    )
    chunks = room.chunk_by_tokens(list(range(6)), max_tokens=3, overlap=0)
    assert chunks == [[0, 1, 2], [3, 4, 5]], "With zero overlap and an exact fit, no empty trailing window."


@pytest.mark.parametrize("n,max_tokens,overlap", [(1, 5, 2), (5, 5, 2), (6, 5, 2), (37, 8, 3), (100, 7, 6), (23, 1, 0)])
def test_windows_hold_for_awkward_lengths(n, max_tokens, overlap):
    ids = [i * 3 for i in range(n)]
    _check_windows(ids, room.chunk_by_tokens(ids, max_tokens, overlap), max_tokens, overlap)


def test_a_short_scroll_is_one_window_and_an_empty_one_is_none():
    assert room.chunk_by_tokens([1, 2, 3], max_tokens=10, overlap=2) == [[1, 2, 3]]
    assert room.chunk_by_tokens([], max_tokens=10, overlap=2) == []


def test_impossible_windows_are_refused():
    with pytest.raises(ValueError):
        room.chunk_by_tokens([1, 2, 3], max_tokens=4, overlap=4)
    with pytest.raises(ValueError):
        room.chunk_by_tokens([1, 2, 3], max_tokens=0, overlap=0)
    with pytest.raises(ValueError):
        room.chunk_by_tokens([1, 2, 3], max_tokens=4, overlap=-1)


# --------------------------------------------------------------------- prophecy
LEDGER_TRUTH = {
    ("the thing", "char"): 11,
    ("the thing", "bpe"): 3,
    ("then", "bpe"): 3,
    ("singing", "bpe"): 3,
    ("thing thing thing", "char"): 19,
    ("thing thing thing", "bpe"): 6,
}


def test_the_ledger_prophecy_is_complete():
    assert set(room.TOKEN_PROPHECY) == set(LEDGER_TRUTH), "Do not add or remove entries from TOKEN_PROPHECY."
    assert room.TOY_MERGES == [("t", "h"), ("th", "e"), ("the", "</w>"), ("i", "n"), ("in", "g"), ("ing", "</w>")], (
        "Leave TOY_MERGES as it is."
    )


@pytest.mark.parametrize("entry", list(LEDGER_TRUTH), ids=[f"{s}-{k}" for s, k in LEDGER_TRUTH])
def test_the_ledger_prophecy_holds(entry):
    prediction = room.TOKEN_PROPHECY.get(entry)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied the price of {entry}. Fill in TOKEN_PROPHECY.")
    text, kind = entry
    truth = LEDGER_TRUTH[entry]
    hint = (
        "the Codex charges one id per character plus <bos> and <eos>"
        if kind == "char"
        else "replay TOY_MERGES in order on each word; (the, </w>) only fires when </w> directly follows 'the'"
    )
    assert prediction == truth, f"{text!r} under {kind} costs {truth} tokens, you predicted {prediction}: {hint}."
