"""ROOM 6.1 - THE SINGLE GAZE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Attention is two matmuls with a softmax between them. Everything here touches
only the last two axes, so the same code serves (B, T, d) in this room and
(B, H, T, head_dim) in Room 6.3.
"""

from __future__ import annotations

import math

import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q @ k^T / sqrt(d) with d = q.shape[-1]. (B, Tq, d), (B, Tk, d) -> (B, Tq, Tk)."""
    d = q.shape[-1]
    return q @ k.transpose(-2, -1) / math.sqrt(d)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights @ v, weights). mask: bool, True = may attend, applied BEFORE softmax."""
    scores = attention_scores(q, k)
    if mask is not None:
        # -inf before the softmax: exp(-inf) == 0 exactly, and the denominator
        # only ever sees the allowed keys, so the surviving weights sum to 1.
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


# The Prophecy of the Peaks. With q, k ~ N(0, 1) in d dimensions, q . k has
# variance d, so unscaled scores have standard deviation sqrt(d): 2 at d=4
# (already lopsided), 8 at d=64 (one voice dominates), 32 at d=1024 (a
# one-hot). Dividing by sqrt(d) pins the standard deviation to 1 at every d,
# so the scaled softmax stays equally spread no matter how wide the head is.
PEAKINESS_PROPHECY: dict[str, str] = {
    "unscaled, d=4": "spread",
    "unscaled, d=64": "sharp",
    "unscaled, d=1024": "one_hot",
    "scaled, d=4": "spread",
    "scaled, d=64": "spread",
    "scaled, d=1024": "spread",
}
