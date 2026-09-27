"""ROOM 5.2 - BYTE-PAIR BINDING  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Classic BPE (Sennrich, Haddow and Birch, 2016): words are split into characters
with an end-of-word marker, then the most frequent adjacent pair is merged into
one symbol, num_merges times. Ties are broken by the lexicographically smallest
pair so that training is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

END_OF_WORD = "</w>"

Word = tuple[str, ...]
Pair = tuple[str, str]


def word_counts(text: str) -> dict[Word, int]:
    """Split on whitespace; each word becomes a tuple of characters + END_OF_WORD."""
    counts: dict[Word, int] = {}
    for word in text.split():
        symbols = (*word, END_OF_WORD)
        counts[symbols] = counts.get(symbols, 0) + 1
    return counts


def pair_counts(words: dict[Word, int]) -> dict[Pair, int]:
    """Count adjacent symbol pairs across all words, weighted by word frequency."""
    counts: dict[Pair, int] = {}
    for symbols, n in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + n
    return counts


def merge_pair(symbols: Word, pair: Pair) -> Word:
    """Replace every non-overlapping occurrence of ``pair`` in one word, left to right."""
    a, b = pair
    merged = a + b
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """Learn up to ``num_merges`` merges, most frequent pair first."""
    words = word_counts(text)
    merges: list[Pair] = []
    for _ in range(num_merges):
        pairs = pair_counts(words)
        if not pairs:
            break  # every word is a single symbol; nothing left to bind
        best = min(pairs, key=lambda p: (-pairs[p], p))  # max count, then smallest pair
        merges.append(best)
        rewritten: dict[Word, int] = {}
        for symbols, n in words.items():
            new = merge_pair(symbols, best)
            rewritten[new] = rewritten.get(new, 0) + n
        words = rewritten
    return merges


def encode_bpe(text: str, merges: Sequence[Pair]) -> list[str]:
    """Split into words, then apply the merges to each word in rank order."""
    tokens: list[str] = []
    cache: dict[str, Word] = {}
    for word in text.split():
        if word not in cache:
            symbols: Word = (*word, END_OF_WORD)
            for pair in merges:
                if len(symbols) == 1:
                    break
                symbols = merge_pair(symbols, pair)
            cache[word] = symbols
        tokens.extend(cache[word])
    return tokens


def decode_bpe(tokens: Sequence[str]) -> str:
    """Glue the tokens back together; every END_OF_WORD becomes a space."""
    return "".join(tokens).replace(END_OF_WORD, " ").rstrip(" ")


# ---------------------------------------------------------------------------
# THE BINDING PROPHECY  (answers)
# ---------------------------------------------------------------------------
TINY_CORPUS = "hug hug hug pug pun pun bun"

BPE_PROPHECY: dict[str, Pair | int | None] = {
    "first merge": ("g", END_OF_WORD),  # (u,g) and (g,</w>) both count 4; "g" < "u"
    "second merge": ("u", "g" + END_OF_WORD),
    "third merge": ("h", "ug" + END_OF_WORD),  # three pairs at 3; "h" is the smallest
    "tokens of 'hugs' after three merges": 5,  # h u g s </w>: no merge fires without </w> after g
    "tokens of 'hug pun' after three merges": 5,  # hug</w> (1) + p u n </w> (4)
}
