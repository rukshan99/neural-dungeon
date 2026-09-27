"""ROOM 5.1 - THE CHARACTER CODEX  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The vocabulary is four special tokens followed by the sorted characters of the
training text. Fixing the specials at ids 0..3 means <pad> is 0 no matter what
corpus the codex was copied from, which every later floor relies on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class CharTokenizer:
    """One token per character, plus four special tokens in front."""

    def __init__(self, chars: Iterable[str]):
        self.chars = list(chars)
        self.vocab = SPECIAL_TOKENS + self.chars
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, UNK_ID) for ch in text]
        if add_special:
            ids = [BOS_ID, *ids, EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        pieces = []
        for i in ids:
            i = int(i)
            if skip_special and i < len(SPECIAL_TOKENS):
                continue
            pieces.append(self.itos[i])
        return "".join(pieces)

    def batch_encode(
        self,
        texts: Sequence[str],
        max_len: int | None = None,
        pad_to_longest: bool = True,
    ) -> dict[str, list[list[int]]]:
        if not pad_to_longest and max_len is None:
            raise ValueError("pad_to_longest=False needs a max_len to pad to")
        seqs = [self.encode(t) for t in texts]
        if max_len is not None:
            # Truncate the middle, never the <eos>: HF-style truncation keeps specials.
            seqs = [s if len(s) <= max_len else s[: max_len - 1] + [EOS_ID] for s in seqs]
        if not seqs:
            return {"input_ids": [], "attention_mask": []}
        target = max(len(s) for s in seqs) if pad_to_longest else max_len
        input_ids = [s + [PAD_ID] * (target - len(s)) for s in seqs]
        attention_mask = [[1] * len(s) + [0] * (target - len(s)) for s in seqs]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def to_tensors(batch: dict[str, list[list[int]]]) -> dict:
    """Lists of lists -> int64 tensors of shape (B, T). Imports torch lazily."""
    import torch

    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
