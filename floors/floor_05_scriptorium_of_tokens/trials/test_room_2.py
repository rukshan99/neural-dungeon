"""TRIAL 5.2 - BYTE-PAIR BINDING

The ledger is checked against the textbook, against the Chronicles, and
against your own prophecy about a seven-word corpus.
"""

import pytest

from dungeon.artifacts.tiny_gpt import read_corpus
from dungeon.trials import load_room

room = load_room(__file__, "room_2_byte_pair_binding")

EOW = "</w>"

# The example from the BPE paper (Sennrich et al., 2016): low x5, lower x2, newest x6, widest x3.
TEXTBOOK = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
TEXTBOOK_MERGES = [("e", "s"), ("es", "t"), ("est", EOW), ("l", "o"), ("lo", "w")]


@pytest.fixture(scope="module")
def chronicle_page():
    return read_corpus()[:8000]


@pytest.fixture(scope="module")
def chronicle_merges(chronicle_page):
    return room.train_bpe(chronicle_page, 80)


# ------------------------------------------------------------------ mechanics
def test_words_are_counted_with_their_end_of_word_marks():
    counts = room.word_counts("low low lower\nnewest")
    assert counts == {
        ("l", "o", "w", EOW): 2,
        ("l", "o", "w", "e", "r", EOW): 1,
        ("n", "e", "w", "e", "s", "t", EOW): 1,
    }, f"Each word is a tuple of characters ending in {EOW!r}, mapped to its count. Got {counts}."


def test_pairs_are_counted_weighted_by_word_frequency():
    pairs = room.pair_counts(room.word_counts(TEXTBOOK))
    assert pairs[("e", "s")] == 9, f"(e, s) appears in newest x6 and widest x3 = 9, you counted {pairs.get(('e', 's'))}."
    assert pairs[("l", "o")] == 7, f"(l, o) is in low x5 and lower x2 = 7, you counted {pairs.get(('l', 'o'))}."
    assert pairs[("w", EOW)] == 5, "(w, </w>) closes only 'low' (x5)."
    assert pairs[("w", "e")] == 8, "(w, e) is in lower x2 and newest x6 = 8."
    assert ("s", "l") not in pairs, "Pairs never cross word boundaries: there is no (s, l) anywhere."


def test_merge_pair_binds_left_to_right_without_overlap():
    assert room.merge_pair(("a", "a", "a", EOW), ("a", "a")) == ("aa", "a", EOW), (
        "Scan left to right and never reuse a symbol: (a a a) with (a, a) is (aa, a)."
    )
    assert room.merge_pair(("l", "o", "w", EOW), ("l", "o")) == ("lo", "w", EOW)
    assert room.merge_pair(("l", "o", "w", EOW), ("x", "y")) == ("l", "o", "w", EOW), "A pair that is absent changes nothing."
    assert room.merge_pair(("lo", "w", "lo", "w"), ("lo", "w")) == ("low", "low")


# ------------------------------------------------------------------- textbook
def test_the_textbook_ledger_opens_with_es_est_and_low():
    merges = room.train_bpe(TEXTBOOK, 5)
    assert merges == TEXTBOOK_MERGES, (
        f"Expected the first five merges {TEXTBOOK_MERGES}, got {merges}. "
        "Most frequent pair first; on ties take the lexicographically smallest pair."
    )


def test_ties_are_broken_by_the_smallest_pair():
    # (a, b) and (b, c) both occur exactly once. "a" < "b", so (a, b) must win.
    assert room.train_bpe("abc", 1) == [("a", "b")], "Both pairs count 1; the tie goes to the lexicographically smallest."


def test_the_ledger_stops_when_nothing_is_left_to_bind():
    merges = room.train_bpe("a b", 10)
    assert merges == [("a", EOW), ("b", EOW)], (
        f"After two merges every word is a single symbol. train_bpe must return 2 merges, not pretend to do 10: {merges}"
    )


# ------------------------------------------------------------------- encoding
def test_encoding_replays_the_ledger_in_rank_order():
    merges = [("b", "c"), ("a", "b")]
    assert room.encode_bpe("abc", merges) == ["a", "bc", EOW], (
        "Merges apply in the order they were learned: (b, c) first eats the b, so (a, b) never fires."
    )
    assert room.encode_bpe("abc", [("a", "b"), ("ab", "c"), ("abc", EOW)]) == ["abc" + EOW]


def test_encoding_the_textbook_gives_whole_words():
    merges = room.train_bpe(TEXTBOOK, 5)
    assert room.encode_bpe("low lowest", merges) == ["low", EOW, "low", "est" + EOW], (
        "After 5 merges 'low' is [low, </w>] (the (w, </w>) pair was never merged) and 'lowest' is [low, est</w>]."
    )


def test_unseen_characters_stay_single_tokens_not_errors():
    merges = room.train_bpe(TEXTBOOK, 5)
    assert room.encode_bpe("zq", merges) == ["z", "q", EOW], (
        "Characters absent from training just stay as themselves. (Remember this when you meet the Golem.)"
    )


def test_more_merges_never_mean_more_tokens(chronicle_page, chronicle_merges):
    counts = [len(room.encode_bpe(chronicle_page, chronicle_merges[:k])) for k in (0, 20, 40, 80)]
    assert all(a >= b for a, b in zip(counts, counts[1:])), (
        f"Token counts for 0/20/40/80 merges were {counts}. Each extra merge can only bind, never split."
    )
    assert counts[-1] < counts[0] * 0.75, (
        f"80 merges should shrink the page noticeably: {counts[0]} -> {counts[-1]} tokens is not much of a binding."
    )


def test_a_bound_page_reads_back_word_for_word(chronicle_page, chronicle_merges):
    tokens = room.encode_bpe(chronicle_page, chronicle_merges)
    assert room.decode_bpe(tokens) == " ".join(chronicle_page.split()), (
        "decode(encode(text)) must give back the words joined by single spaces (whitespace is normalized, words are not)."
    )
    assert room.decode_bpe([]) == ""


def test_the_scribe_is_deterministic(chronicle_page, chronicle_merges):
    again = room.train_bpe(chronicle_page, 80)
    assert again == chronicle_merges, "Training twice on the same text must produce the same ledger."
    assert room.encode_bpe(chronicle_page, chronicle_merges) == room.encode_bpe(chronicle_page, chronicle_merges)


# ------------------------------------------------------------------- prophecy
PROPHECY_TRUTH = {
    "first merge": ("g", EOW),
    "second merge": ("u", "g" + EOW),
    "third merge": ("h", "ug" + EOW),
    "tokens of 'hugs' after three merges": 5,
    "tokens of 'hug pun' after three merges": 5,
}


def test_the_prophecy_is_complete():
    assert set(room.BPE_PROPHECY) == set(PROPHECY_TRUTH), "Do not rename or remove the prophecy's questions."
    assert room.TINY_CORPUS == "hug hug hug pug pun pun bun", "Leave TINY_CORPUS as it is."


@pytest.mark.parametrize("question", list(PROPHECY_TRUTH), ids=list(PROPHECY_TRUTH))
def test_the_binding_prophecy_holds(question):
    prediction = room.BPE_PROPHECY.get(question)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {question!r}. Fill in BPE_PROPHECY.")
    truth = PROPHECY_TRUTH[question]
    if isinstance(truth, tuple):
        assert tuple(prediction) == truth, (
            f"For {question!r} the ledger says {truth}, you predicted {tuple(prediction)}. "
            "Count pairs weighted by word frequency; on a tie take the lexicographically smallest pair."
        )
    else:
        assert prediction == truth, (
            f"For {question!r} the answer is {truth} tokens, you predicted {prediction}. "
            "Remember the end-of-word marker: 'hug</w>' cannot appear inside 'hugs'."
        )
