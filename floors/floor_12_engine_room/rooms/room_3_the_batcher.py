"""ROOM 12.3 - THE BATCHER

    A ferry crosses the flooded lower hall, and it takes eight petitioners at
    a time whether they are eight or one. The petitions are of different
    lengths. The ferryman lines them up so that every last word is at the
    same edge of the boat, and fills the empty seats with sandbags.

Prompts of different lengths cannot share a (B, T) tensor without padding. This
room pads on the LEFT so every row's newest token is in the last column: the
argmax of ``logits[:, -1]`` is the next token for every row and decoding appends
one column to the batch. Two things make the sandbags harmless:

    position ids     0, 1, 2, ... starting at each row's FIRST REAL token, so a
                     padded row is embedded exactly as it would be alone
    attention mask   causal (query i sees keys <= i) AND key-not-a-pad; padded
                     keys get -inf before the softmax and so weight exactly 0

One trap: a padded QUERY at the start of a row sees only padded keys. Every
score is -inf, the softmax is NaN, and NaN spreads through every matmul that
follows. Those rows are never read, but they must be finite: ``torch.nan_to_num``
after the softmax, or a large finite negative instead of -inf, or let every
query see itself. Pick one and say which.

The reference GPT has no hooks for masks or positions, so you write the forward
yourself around its weights (``wte``, ``wpe``, ``blocks[i].ln_1/attn/ln_2/mlp``,
``ln_f``, ``lm_head``), the way room 1 did. No cache in this room: the whole padded
batch is re-run each step. The boss will make you combine the two.
"""

from __future__ import annotations

import torch


def left_pad(sequences: list[list[int]], pad_id: int):
    """Pad a ragged list of token lists on the LEFT to a rectangle.

    Returns ``(idx, attention_mask, position_ids)``:
      idx             (B, T) int64; ``pad_id`` in the padded slots, T = longest sequence
      attention_mask  (B, T) bool; True on real tokens, False on pads
      position_ids    (B, T) int64; 0, 1, 2, ... from each row's first real token; 0 on pads
    Raise ValueError if any sequence is empty. Hint for the positions: a cumulative
    sum of the mask, minus one, clamped at 0.
    """
    raise NotImplementedError("left_pad() is unwritten")


def masked_attention(block, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """One block's self-attention with an explicit boolean mask.

    x: (B, T, C) after ``block.ln_1``. allowed: bool, broadcastable to (B, H, T, T),
    True where query i may look at key j. Returns (B, T, C) after ``c_proj``.
    Scores that are not allowed get -inf before the softmax. Fully masked rows
    must come out finite (see the module docstring).
    """
    raise NotImplementedError("masked_attention() is unwritten")


def forward_masked(model, idx: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """The GPT forward with explicit positions and a causal + key-padding mask.

    idx (B, T), attention_mask (B, T) bool, position_ids (B, T). Returns logits (B, T, V).
    Embed with ``model.wte(idx) + model.wpe(position_ids)``. Build
    ``allowed = tril(T, T)[None, None] & attention_mask[:, None, None, :]`` -> (B, 1, T, T),
    run every block with ``masked_attention``, then ``ln_f`` and ``lm_head``.
    Every logit, including those at padded positions, must be finite.
    """
    raise NotImplementedError("forward_masked() is unwritten")


def batched_greedy_generate(model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None = None) -> list[list[int]]:
    """Greedy-decode all prompts together. Returns ONLY the generated tokens, one list per prompt.

    Left-pad, then loop: ``forward_masked`` on the whole batch, take the argmax of the
    last column for every row, append a column to idx / mask (True) / position_ids
    (previous + 1). A row that produces ``eos_id`` is finished: its list stops before
    the eos and later tokens for that row are discarded; the batch stops when every
    row is finished or ``max_new_tokens`` is reached. Never exceed block_size (the
    batch is as long as its longest prompt plus the new tokens).

    Each row's output must equal that prompt's individual greedy generation.
    Decorate with ``@torch.no_grad()``.
    """
    raise NotImplementedError("batched_greedy_generate() is unwritten")
