"""ROOM 5.2 - BYTE-PAIR BINDING

    The second desk is covered in strips of vellum. The scribe here refuses to
    copy one letter at a time. "Look how often 'th' appears," she says, and
    binds the two strips into one. Then 'the'. Then 'the' with the little
    end-of-word mark. Every binding is written in a ledger, in order. The
    ledger IS the tokenizer.

Byte-pair encoding (BPE) learns a vocabulary from data. The algorithm, in full:

    1. Split the text into words on whitespace. Turn each word into a tuple
       of characters with an END_OF_WORD marker at the end: "low" -> (l, o, w, </w>).
       Count how many times each word occurs.
    2. Count every adjacent pair of symbols, weighted by word frequency.
    3. Take the most frequent pair. Ties: the lexicographically smallest pair
       (plain Python tuple comparison). Record it in the merge list.
    4. Rewrite every word, replacing that pair with the concatenated symbol.
    5. Repeat from 2, num_merges times, or until no pairs remain.

Encoding new text replays the ledger: split into words, start from characters,
apply the merges IN THE ORDER THEY WERE LEARNED. Order matters: a later merge
like (es, t) can only fire after the earlier (e, s) has created "es".

The end-of-word marker is what makes "hug" and "hugs" different: the token
"hug</w>" can never appear inside "hugs". Decoding turns every "</w>" back into
a space, so the round trip preserves words but normalizes whitespace: the
result equals " ".join(text.split()).

Subword tokenizers are the middle ground between characters (tiny vocabulary,
very long sequences) and words (huge vocabulary, unknown words everywhere).
GPT-2, Llama and most models you will meet use BPE of some flavour; the boss
of this floor uses it over bytes instead of characters.
"""

from __future__ import annotations

from collections.abc import Sequence

END_OF_WORD = "</w>"

Word = tuple[str, ...]  # a word as a tuple of symbols, e.g. ("l", "o", "w", "</w>")
Pair = tuple[str, str]


def word_counts(text: str) -> dict[Word, int]:
    """Split ``text`` on whitespace; map each word-as-symbols to its frequency.

    word_counts("low low lower") == {("l","o","w","</w>"): 2, ("l","o","w","e","r","</w>"): 1}
    """
    raise NotImplementedError("word_counts() is unwritten")


def pair_counts(words: dict[Word, int]) -> dict[Pair, int]:
    """Count adjacent symbol pairs over all words, each occurrence weighted by the word's count.

    ("l","o","w","</w>") with count 5 contributes 5 to (l,o), (o,w) and (w,</w>).
    """
    raise NotImplementedError("pair_counts() is unwritten")


def merge_pair(symbols: Word, pair: Pair) -> Word:
    """Replace every non-overlapping occurrence of ``pair`` in one word with ``pair[0] + pair[1]``.

    Scan left to right. merge_pair(("a","a","a","</w>"), ("a","a")) == ("aa","a","</w>").
    """
    raise NotImplementedError("merge_pair() is unwritten")


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """Learn up to ``num_merges`` merges from ``text``. Most frequent pair first.

    Ties are broken by choosing the lexicographically smallest pair. Stop early
    (returning fewer merges) when no adjacent pairs remain.
    """
    raise NotImplementedError("train_bpe() is unwritten")


def encode_bpe(text: str, merges: Sequence[Pair]) -> list[str]:
    """Tokenize ``text`` with a learned merge list. Returns token strings, not ids.

    For each whitespace-separated word: start from (*word, END_OF_WORD), apply
    every merge in rank order with ``merge_pair``, then append the resulting
    symbols. Characters never seen in training simply stay single-character
    tokens: there is no <unk> in this scheme.
    """
    raise NotImplementedError("encode_bpe() is unwritten")


def decode_bpe(tokens: Sequence[str]) -> str:
    """Concatenate the tokens; every END_OF_WORD becomes a space; no trailing space."""
    raise NotImplementedError("decode_bpe() is unwritten")


# ---------------------------------------------------------------------------
# THE BINDING PROPHECY
# The corpus below has 7 words. Work the algorithm by hand (rules 1-5 above,
# tie-break included) and fill in the answers BEFORE running the trial.
# Merges are (str, str) pairs; token counts are ints.
# ---------------------------------------------------------------------------
TINY_CORPUS = "hug hug hug pug pun pun bun"

BPE_PROPHECY: dict[str, Pair | int | None] = {
    "first merge": None,
    "second merge": None,
    "third merge": None,
    "tokens of 'hugs' after three merges": None,
    "tokens of 'hug pun' after three merges": None,
}
