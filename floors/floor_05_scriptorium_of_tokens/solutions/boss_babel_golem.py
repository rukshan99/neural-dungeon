"""BOSS - THE BABEL GOLEM  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Byte-level BPE. The base vocabulary is every byte value 0..255, so any string
that UTF-8 can encode (which is every string) can be tokenized. Merges create
new ids 256, 257, ... whose value is the concatenation of two existing byte
strings. Decoding joins the byte strings and decodes UTF-8 with errors="replace",
so an id whose bytes are only part of a multi-byte character shows up as U+FFFD
instead of crashing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# A piece is optional leading whitespace glued to a run of non-whitespace, or a
# trailing whitespace run. The pieces concatenate back to the original text, so
# nothing is lost, and no merge ever crosses a word boundary.
PIECE_PATTERN = re.compile(r"\s*\S+|\s+")

BytePair = tuple[int, int]


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into pieces whose concatenation is exactly ``text``."""
    return PIECE_PATTERN.findall(text)


def merge_ids(ids: Sequence[int], pair: BytePair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    a, b = pair
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteLevelBPE:
    """BPE over UTF-8 bytes. Nothing is ever unknown."""

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[BytePair] = []
        self.ranks: dict[BytePair, int] = {}
        self.unk_id: int | None = None  # there is no such thing here
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256 (one id per byte), got {vocab_size}")
        self.__init__()  # start from the 256 base tokens every time
        pieces: dict[tuple[int, ...], int] = {}
        for piece in pretokenize(text):
            key = tuple(piece.encode("utf-8"))
            pieces[key] = pieces.get(key, 0) + 1

        while len(self.vocab) < vocab_size:
            counts: dict[BytePair, int] = {}
            for seq, n in pieces.items():
                for pair in zip(seq, seq[1:]):
                    counts[pair] = counts.get(pair, 0) + n
            if not counts:
                break  # every piece is already a single token
            best = min(counts, key=lambda p: (-counts[p], p))  # max count, then smallest pair
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.ranks[best] = len(self.merges)
            self.merges.append(best)
            rewritten: dict[tuple[int, ...], int] = {}
            for seq, n in pieces.items():
                new_seq = tuple(merge_ids(seq, best, new_id))
                rewritten[new_seq] = rewritten.get(new_seq, 0) + n
            pieces = rewritten
        self._cache.clear()

    def _encode_piece(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) > 1:
            # Merge the pair with the LOWEST rank (learned earliest) first.
            best: BytePair | None = None
            best_rank = len(self.merges)
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = merge_ids(ids, best, 256 + best_rank)
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in pretokenize(text):
            data = piece.encode("utf-8")
            if data not in self._cache:
                self._cache[data] = self._encode_piece(data)
            out.extend(self._cache[data])
        return out

    def decode(self, ids: Sequence[int]) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")


def unknown_rate(tokenizer, text: str) -> float:
    """Fraction of ids that are the tokenizer's unknown token. 0.0 when it has none."""
    ids = tokenizer.encode(text)
    unk = getattr(tokenizer, "unk_id", None)
    if unk is None or not ids:
        return 0.0
    return sum(1 for i in ids if i == unk) / len(ids)


def compression_ratio(tokenizer, text: str) -> float:
    """UTF-8 bytes per token. 1.0 for an untrained byte tokenizer; higher is better."""
    n_tokens = len(tokenizer.encode(text))
    if n_tokens == 0:
        return 0.0
    return len(text.encode("utf-8")) / n_tokens
