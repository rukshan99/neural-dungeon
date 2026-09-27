"""ROOM 6.1 - THE SINGLE GAZE

    The first carved head in the hall turns to face you. It asks every voice
    in the room the same question: how much of you should I hear? Then it
    listens to all of them at once, each in proportion to its answer.

Attention is a weighted average. One *query* vector q asks; every *key*
vector k_j answers with a relevance score; the matching *value* vectors v_j
are blended by those scores:

    score_j  = (q . k_j) / sqrt(d)        how relevant is key j to this query
    weight_j = softmax(score)_j           weights >= 0 and they sum to 1
    out      = sum_j weight_j * v_j       the blend of values

Batched, that is two matrix products with a softmax between them:

    scores  = q @ k^T / sqrt(d)      (B, Tq, d) x (B, d, Tk) -> (B, Tq, Tk)
    weights = softmax(scores, -1)    one row per query, each row sums to 1
    out     = weights @ v            (B, Tq, Tk) x (B, Tk, dv) -> (B, Tq, dv)

Three things the trial checks that are easy to get wrong:

* The softmax runs over the KEY axis (the last one). A row of weights belongs
  to one query and says how that query splits its hearing across the keys.
* The division is by sqrt(d) where d is the last dim of q and k. Not d_model,
  not T. Without it, dot products of random d-vectors have standard deviation
  sqrt(d), and for large d the softmax collapses onto one key. You will
  measure this yourself in the Prophecy at the bottom.
* Masking happens BEFORE the softmax, by setting forbidden scores to -inf.
  exp(-inf) is exactly 0.0 and the surviving weights still sum to 1. Zeroing
  weights *after* the softmax leaves rows that do not sum to 1 and lets the
  masked keys pollute the denominator. Room 6.2 builds a detector for that.

Write both functions so they touch only the last two axes (transpose(-2, -1),
softmax(dim=-1)). Then they work unchanged on (B, H, T, hd) in Room 6.3.
"""

from __future__ import annotations

import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product scores: ``q @ k^T / sqrt(d)``.

    q: (B, Tq, d)   k: (B, Tk, d)   ->   (B, Tq, Tk), same dtype as q.
    scores[b, i, j] is how strongly query i wants to hear key j, before the
    softmax. ``d`` is ``q.shape[-1]``. Extra leading axes (heads) must pass
    through untouched: use ``transpose(-2, -1)``, not ``.T``.
    """
    raise NotImplementedError("attention_scores() is unwritten")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One head's gaze. Returns ``(out, weights)``.

    q: (B, Tq, d)   k: (B, Tk, d)   v: (B, Tk, dv)
    mask: optional bool tensor broadcastable to (B, Tq, Tk). True = may attend.
          (A (Tq, Tk) mask is fine; it broadcasts over the batch.)
    out: (B, Tq, dv)   weights: (B, Tq, Tk), rows sum to 1, masked entries exactly 0.

    Steps: scores = attention_scores(q, k); where mask is False set the score
    to float("-inf") (``masked_fill`` with ``~mask``); softmax over dim=-1;
    out = weights @ v. The mask goes BEFORE the softmax.
    """
    raise NotImplementedError("scaled_dot_product_attention() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY OF THE PEAKS
# The trial draws q and k with entries ~ N(0, 1), T = 16 keys per query, and
# measures the MEAN over all rows of the LARGEST softmax weight in each row,
# with and without the 1/sqrt(d) scaling. Predict the bucket for each case:
#
#     "spread"    largest weight below 0.5     (the head hears many voices)
#     "sharp"     between 0.5 and 0.9          (mostly one voice)
#     "one_hot"   above 0.9                    (one voice, the rest silenced)
#
# Think about the standard deviation of q . k when q and k have d unit-variance
# entries. Commit before you run anything.
# ---------------------------------------------------------------------------
PEAKINESS_PROPHECY: dict[str, str | None] = {
    "unscaled, d=4": None,
    "unscaled, d=64": None,
    "unscaled, d=1024": None,
    "scaled, d=4": None,
    "scaled, d=64": None,
    "scaled, d=1024": None,
}
