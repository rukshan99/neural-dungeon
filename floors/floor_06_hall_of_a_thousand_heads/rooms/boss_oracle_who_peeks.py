"""BOSS - THE ORACLE WHO PEEKS

                    .-~~~~-.
                   /  _  _  \\           "Ask me the next word. I have never
                  |  (o)(o)  |            been wrong. Not once. Curious, no?"
                  |    __    |
                   \\  (__)  /            The Oracle sits behind four veils and
                  .-`------'-.           swears that each is causal. Its loss
                 /  |  ||  |  \\          curve is magnificent. Its generations
                /   |  ||  |   \\         are gibberish. Somewhere, a head is
               |____|__||__|____|        hearing the future.

WEAKNESS: causal-mask correctness, verified *empirically*. The Oracle's code
is a fixture you may read, but the detectors you write must not: they perturb
the future and watch the past. That way they also work on the next oracle,
whose code you will not have.

The four oracles below are complete and deliberately flawed (or not). Do NOT
fix them; they are the monsters. Same signature as Room 6.2's attn_fn for the
first three; the padding pair also takes ``lengths`` and reads the whole
scroll (no causal veil) but must ignore the blank parchment.

  Phase 1  leak_pairs / detect_leaks - which (query, key) pairs leak, and
           therefore which query positions are compromised. One future
           position perturbed at a time, so the report is exact.
  Phase 2  detect_padding_leak - do real positions change when only the
           padding changes?
  Phase 3  FixedOracle - your own causal + padding multi-head attention that
           passes both detectors and matches torch to 1e-5.
  Phase 4  ORACLE_PROPHECY - before you run anything, say which oracles leak.

Run:  dungeon fight 6
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .room_3_the_thousand_heads import MultiHeadAttention
from .room_4_the_padding_veil import masked_mha  # noqa: F401 - use it in FixedOracle.forward

AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PaddedAttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# THE ORACLES (fixtures: complete, some cursed; leave them exactly as they are)
# ---------------------------------------------------------------------------
def _scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


def oracle_honest(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention done right: the veil is applied as -inf before the softmax."""
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))
    weights = _scores(q, k).masked_fill(~veil, float("-inf")).softmax(dim=-1)
    return weights @ v


def oracle_mask_after_softmax(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Softmax first, veil second. The future is gone from the numerator, not the denominator."""
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool))
    weights = _scores(q, k).softmax(dim=-1) * veil
    return weights @ v


def oracle_off_by_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The veil hangs one seat too far down the hall: position t hears t+1."""
    T = q.shape[1]
    veil = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    weights = _scores(q, k).masked_fill(~veil, float("-inf")).softmax(dim=-1)
    return weights @ v


def oracle_padding_on_keys(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Bidirectional attention that ignores padded KEYS. Honest."""
    T = q.shape[1]
    real = torch.arange(T)[None, :] < torch.as_tensor(lengths)[:, None]  # (B, T)
    weights = _scores(q, k).masked_fill(~real[:, None, :], float("-inf")).softmax(dim=-1)
    return weights @ v


def oracle_padding_on_queries(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Bidirectional attention that veils padded QUERIES instead. Real positions still hear the blanks."""
    T = q.shape[1]
    real = torch.arange(T)[None, :] < torch.as_tensor(lengths)[:, None]  # (B, T)
    scores = _scores(q, k).masked_fill(~real[:, :, None], float("-inf"))
    weights = scores.softmax(dim=-1).masked_fill(~real[:, :, None], 0.0)  # padded rows: silence, not NaN
    return weights @ v


ORACLES: dict[str, Callable] = {
    "oracle_honest": oracle_honest,
    "oracle_mask_after_softmax": oracle_mask_after_softmax,
    "oracle_off_by_one": oracle_off_by_one,
    "oracle_padding_on_keys": oracle_padding_on_keys,
    "oracle_padding_on_queries": oracle_padding_on_queries,
}


# ---------------------------------------------------------------------------
# PHASE 1: WHO HEARS THE FUTURE, AND FROM WHOM
# ---------------------------------------------------------------------------
def leak_pairs(attn_fn: AttnFn, T: int, d: int, seed: int) -> list[tuple[int, int]]:
    """Every (t, j) with j > t such that perturbing position j changes out[:, t].

    Use a batch of 1. Draw q, k, v (1, T, d) from ``torch.Generator().manual_seed(seed)``
    and ``base = attn_fn(q, k, v)``. Then for each j in 1..T-1, ONE AT A TIME:
    clone q, k, v; overwrite row j of all three with fresh randn from the same
    generator; run attn_fn; for every t < j, if out[:, t] is not within 1e-6
    of base[:, t] (``torch.allclose(atol=1e-6, rtol=0)``), record (t, j).
    Return the pairs sorted, as plain Python ints.
    """
    raise NotImplementedError("leak_pairs() is unwritten")


def detect_leaks(attn_fn: AttnFn, T: int, d: int, seed: int) -> list[int]:
    """Every query position t whose output changes when SOME strictly-future key is perturbed.

    The sorted, duplicate-free list of ``t`` from ``leak_pairs``. An honest
    veil returns ``[]``.
    """
    raise NotImplementedError("detect_leaks() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 2: WHO HEARS THE BLANK PARCHMENT
# ---------------------------------------------------------------------------
def detect_padding_leak(attn_fn: PaddedAttnFn, lengths, d: int = 8, seed: int = 0) -> bool:
    """True if any REAL position's output changes when only the PADDED positions change.

    ``attn_fn(q, k, v, lengths) -> out`` with q, k, v of shape (B, T, d) and
    ``lengths`` a list or 1-D tensor of B ints. Take T = max(lengths) (pad to
    the longest scroll). Draw q, k, v from a seeded generator; ``base = attn_fn(...)``.
    Overwrite every padded position (t >= lengths[b]) of q, k and v with fresh
    randn; run again; compare ``out[b, :lengths[b]]`` with ``base[b, :lengths[b]]``
    for every b at atol=1e-6. Return a Python bool.
    """
    raise NotImplementedError("detect_padding_leak() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 3: AN ORACLE THAT CANNOT PEEK
# ---------------------------------------------------------------------------
class FixedOracle(MultiHeadAttention):
    """Room 6.3's heads behind Room 6.4's veils. Same layers (qkv, proj), same layout.

    Inherits ``__init__(d_model, n_heads)``, ``split_heads`` and ``merge_heads``.
    """

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        """x: (B, T, d_model), lengths: (B,) ints or None (= every position is real).

        Causal + key-padding attention, finite everywhere (including padded
        query rows and length-0 scrolls), matching torch's
        ``scaled_dot_product_attention`` with the combined boolean mask to 1e-5
        at every real position. ``masked_mha(self, x, lengths)`` is right there.
        """
        raise NotImplementedError("FixedOracle.forward() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 4: THE PROPHECY
# For each oracle, write "leaks" or "honest" BEFORE running the detectors.
# The trial runs its own detectors and compares.
# ---------------------------------------------------------------------------
ORACLE_PROPHECY: dict[str, str | None] = {
    "oracle_honest": None,
    "oracle_mask_after_softmax": None,
    "oracle_off_by_one": None,
    "oracle_padding_on_keys": None,
    "oracle_padding_on_queries": None,
}
