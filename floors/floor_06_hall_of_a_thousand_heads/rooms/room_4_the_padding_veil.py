"""ROOM 6.4 - THE PADDING VEIL

    Scrolls of different lengths arrive at the hall together. The short ones
    are padded with blank parchment so that all of them fit the same rack.
    The heads must not hear the blanks. A voice that is not there should
    weigh exactly nothing.

Batching sequences of unequal length means padding them to a common T. The
padded positions hold arbitrary numbers (usually zeros, but after a Linear
they are not zero any more) and must be *masked out of the keys*: no query,
real or padded, may put weight on a padded key. This is a mask along the KEY
axis (the last axis of the scores), and it differs per batch element:

    lengths = [3, 5, 2],  T = 5
    key_padding_mask = [[1, 1, 1, 0, 0],       (B, T)  True = real token
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0]]

Combined with the causal veil (T, T) it becomes (B, 1, T, T): the 1 is the
head axis, left as a 1 so the same mask broadcasts over every head:

    combined[b, 0, i, j] = causal[i, j] and padding[b, j]

Two traps:

* Apply the padding mask along the keys (last axis), never the queries. A
  padded *query* row produces garbage, but garbage that nobody reads: the
  loss is masked at those positions too. A padded *key* that is not masked
  changes the output at every real position. The boss has an oracle that
  gets this wrong.
* A row whose every key is masked (an empty sequence, or a bug) is a row of
  all -inf, and softmax of all -inf is NaN, which then spreads through the
  whole batch via the gradient. ``safe_softmax`` returns zeros for such rows
  instead. Silence, not poison.

The test that matters: run a padded batch through ``masked_mha``, then run
each sequence *alone*, unpadded, through the same module. The real positions
must agree to 1e-5. If padding changes what a real token computes, the model
you serve will behave differently from the model you trained.
"""

from __future__ import annotations

import torch

from .room_2_veil_of_causality import causal_mask  # noqa: F401 - use it in masked_mha
from .room_3_the_thousand_heads import MultiHeadAttention


def key_padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """lengths: (B,) integer tensor -> (B, T) bool, True where position t < lengths[b].

    ``torch.arange(T)`` compared against ``lengths[:, None]`` does it in one
    broadcast. dtype must be bool.
    """
    raise NotImplementedError("key_padding_mask() is unwritten")


def combine_masks(causal: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """causal: (T, T) bool, padding: (B, T) bool -> (B, 1, T, T) bool.

    out[b, 0, i, j] = causal[i, j] & padding[b, j]. The padding enters along
    the KEY axis (the last one). The size-1 axis is the head axis.
    """
    raise NotImplementedError("combine_masks() is unwritten")


def safe_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis with a mask, and no NaN.

    scores: (..., Tq, Tk) float.  mask: bool, broadcastable to scores, True = keep.
    Returns weights of scores' shape where:
      * masked entries are exactly 0,
      * every row with at least one True sums to 1,
      * every row with NO True is all zeros (not NaN).
    Fill -inf, softmax, then blank the dead rows: ``mask.any(dim=-1, keepdim=True)``
    tells you which rows are alive.
    """
    raise NotImplementedError("safe_softmax() is unwritten")


def masked_mha(mha: MultiHeadAttention, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Causal + key-padding multi-head self-attention, assembled from ``mha``'s pieces.

    x: (B, T, d_model)   lengths: (B,) ints   ->   out: (B, T, d_model), finite everywhere.

    Use ``mha.qkv``, ``mha.split_heads``, ``mha.merge_heads``, ``mha.proj``,
    ``mha.head_dim`` and your own ``safe_softmax`` rather than ``mha.forward``:
    the forward's plain softmax turns a fully masked row into NaN. Scores are
    q @ k^T / sqrt(head_dim); the mask is combine_masks(causal_mask(T),
    key_padding_mask(lengths, T)). Padded query positions may hold anything
    finite; real positions must equal what the unpadded sequence would give.
    """
    raise NotImplementedError("masked_mha() is unwritten")
