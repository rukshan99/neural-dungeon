"""BOSS - THE ORACLE WHO PEEKS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The oracles are fixtures and stay exactly as cursed as they are in rooms/.
The detectors never read code: perturb one position, watch the others.

Truths the detectors uncover (T = 8):
  oracle_honest              no leaks.
  oracle_mask_after_softmax  every (t, j) with j > t leaks: the denominator of
                             the softmax still contains every future key, so
                             every position but the last is compromised.
  oracle_off_by_one          exactly (t, t+1) for t in 0..T-2: one seat ahead.
  oracle_padding_on_keys     honest.
  oracle_padding_on_queries  leaks: real queries still hear the padded keys.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn  # noqa: F401 - FixedOracle is an nn.Module through MultiHeadAttention

from .room_3_the_thousand_heads import MultiHeadAttention
from .room_4_the_padding_veil import masked_mha

AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PaddedAttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# THE ORACLES (fixtures: identical to rooms/, cursed on purpose)
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
# PHASE 1
# ---------------------------------------------------------------------------
def leak_pairs(attn_fn: AttnFn, T: int, d: int, seed: int) -> list[tuple[int, int]]:
    """Every (t, j), j > t, such that perturbing position j alone changes out[:, t]."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, T, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v)
    pairs: list[tuple[int, int]] = []
    for j in range(1, T):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for tensor in (q2, k2, v2):
            tensor[:, j] = torch.randn(1, d, generator=g)
        out = attn_fn(q2, k2, v2)
        for t in range(j):
            if not torch.allclose(out[:, t], base[:, t], atol=1e-6, rtol=0.0):
                pairs.append((t, j))
    return sorted(pairs)


def detect_leaks(attn_fn: AttnFn, T: int, d: int, seed: int) -> list[int]:
    """The query positions that hear at least one future key."""
    return sorted({t for t, _ in leak_pairs(attn_fn, T, d, seed)})


# ---------------------------------------------------------------------------
# PHASE 2
# ---------------------------------------------------------------------------
def detect_padding_leak(attn_fn: PaddedAttnFn, lengths, d: int = 8, seed: int = 0) -> bool:
    """True if changing only the padded positions changes any real position's output."""
    lengths = torch.as_tensor(lengths)
    B, T = int(lengths.shape[0]), int(lengths.max())
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, T, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v, lengths)
    real = torch.arange(T)[None, :] < lengths[:, None]  # (B, T)
    padded = ~real
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for tensor in (q2, k2, v2):
        tensor[padded] = torch.randn(int(padded.sum()), d, generator=g)
    out = attn_fn(q2, k2, v2, lengths)
    return not torch.allclose(out[real], base[real], atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# PHASE 3
# ---------------------------------------------------------------------------
class FixedOracle(MultiHeadAttention):
    """Room 6.3's heads behind Room 6.4's veils. Same layers (qkv, proj), same layout."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        B, T, _ = x.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        return masked_mha(self, x, torch.as_tensor(lengths))


# ---------------------------------------------------------------------------
# PHASE 4
# ---------------------------------------------------------------------------
ORACLE_PROPHECY: dict[str, str] = {
    "oracle_honest": "honest",
    "oracle_mask_after_softmax": "leaks",
    "oracle_off_by_one": "leaks",
    "oracle_padding_on_keys": "honest",
    "oracle_padding_on_queries": "leaks",
}
