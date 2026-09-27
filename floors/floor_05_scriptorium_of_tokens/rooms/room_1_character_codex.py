"""ROOM 5.1 - THE CHARACTER CODEX

    The first desk in the scriptorium. A scribe has copied every distinct
    character of the Chronicles into a codex, one per line, numbered. Four
    lines at the front are reserved and written in red ink: <pad>, <unk>,
    <bos>, <eos>. "Every model downstairs reads numbers," the scribe says.
    "This book is how letters become numbers, and back."

A tokenizer turns text into a list of integer ids and back. The character
tokenizer is the simplest one that works: every distinct character is a
token. Its vocabulary is small (69 characters for the Chronicles) but its
sequences are long (one id per character), and a character it has never seen
has no number at all. That last problem is what the boss of this floor is made of.

Conventions this room fixes, and the rest of the dungeon relies on:

    id 0  <pad>   filler so that a batch of ragged sequences becomes a rectangle
    id 1  <unk>   the id of every character that is not in the codex
    id 2  <bos>   beginning of sequence, prepended by encode()
    id 3  <eos>   end of sequence, appended by encode()
    4..   the ordinary characters, in sorted order

Putting the specials first (not last) means <pad> is 0 whatever the corpus,
so a padded batch is zeros on the right no matter which codex produced it.

Padding and attention masks: models take rectangular (B, T) tensors, but
sentences have different lengths. batch_encode() right-pads every sequence
with <pad> to a common length and returns an *attention mask* of the same
shape with 1 over real tokens (specials included) and 0 over padding. The
mask is how attention and the loss learn to ignore the filler; the pad id
alone is not enough, because a model would otherwise happily attend to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class CharTokenizer:
    """One token per character, plus the four special tokens in front.

    Attributes the trial (and later floors) expect after ``__init__``:

        self.chars      list[str]         the ordinary characters, in the given order
        self.vocab      list[str]         SPECIAL_TOKENS + self.chars
        self.stoi       dict[str, int]    token string -> id
        self.itos       dict[int, str]    id -> token string
        self.pad_id, self.unk_id, self.bos_id, self.eos_id     0, 1, 2, 3
    """

    def __init__(self, chars: Iterable[str]):
        """``chars``: the ordinary characters of the codex, already deduplicated and ordered.

        Build ``vocab`` as the four specials followed by ``chars``, then the two
        lookup dicts. Do not sort here; ``from_text`` passes the characters sorted.
        """
        raise NotImplementedError("CharTokenizer.__init__() is unwritten")

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """The codex of a training text: its distinct characters in sorted order."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Specials plus characters. For the Chronicles that is 4 + 69 = 73."""
        return len(self.vocab)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Characters -> ids. Unknown characters become UNK_ID (never raise).

        With ``add_special`` the result is ``[BOS_ID, *ids, EOS_ID]``. Return a plain
        list of Python ints; ``to_tensors`` does the torch conversion later.
        """
        raise NotImplementedError("CharTokenizer.encode() is unwritten")

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """ids -> text. Accept any sequence of ints (including numpy/torch scalars: use int()).

        With ``skip_special`` every id below 4 is dropped, so <pad>, <unk>, <bos> and
        <eos> vanish and an unknown character is simply gone (there is nothing to
        put back). With ``skip_special=False`` they render as their strings, e.g.
        "<bos>Hi<eos><pad>".
        """
        raise NotImplementedError("CharTokenizer.decode() is unwritten")

    def batch_encode(
        self,
        texts: Sequence[str],
        max_len: int | None = None,
        pad_to_longest: bool = True,
    ) -> dict[str, list[list[int]]]:
        """Encode several texts into a rectangle. Returns lists of lists, not tensors.

        Contract:
          * every text is encoded with specials (<bos> ... <eos>);
          * if ``max_len`` is given, a sequence longer than ``max_len`` is cut to
            ``max_len`` ids while KEEPING <eos> as its last id: ``ids[:max_len - 1] + [EOS_ID]``;
          * the common length L is the longest sequence in the batch (after
            truncation) when ``pad_to_longest`` is True, otherwise exactly ``max_len``
            (raise ValueError if ``pad_to_longest`` is False and ``max_len`` is None);
          * pad on the RIGHT with PAD_ID up to L;
          * return {"input_ids": ..., "attention_mask": ...} where the mask is 1 over
            real ids (specials included) and 0 over padding;
          * an empty ``texts`` gives {"input_ids": [], "attention_mask": []}.
        """
        raise NotImplementedError("CharTokenizer.batch_encode() is unwritten")


def to_tensors(batch: dict[str, list[list[int]]]) -> dict:
    """Turn the output of ``batch_encode`` into int64 torch tensors of shape (B, T).

    Keep the same keys. Import torch *inside* the function so that the rest of this
    room works without it.
    """
    raise NotImplementedError("to_tensors() is unwritten")
