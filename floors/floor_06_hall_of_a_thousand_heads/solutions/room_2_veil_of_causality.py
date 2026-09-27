"""ROOM 6.2 - THE VEIL OF CAUSALITY  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The veil is torch.tril. The interesting function is leaks_future, which never
reads code: it perturbs the future and checks whether the past noticed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .room_1_the_single_gaze import scaled_dot_product_attention


def causal_mask(T: int) -> torch.Tensor:
    """(T, T) bool with mask[i, j] == (j <= i)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-attention behind the veil. The (T, T) mask broadcasts over the batch."""
    return scaled_dot_product_attention(q, k, v, mask=causal_mask(q.shape[-2]))


def leaks_future(
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    B: int,
    T: int,
    d: int,
    seed: int,
) -> bool:
    """True if perturbing positions > t ever changes the output at positions <= t."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, T, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v)
    for t in range(T - 1):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for tensor in (q2, k2, v2):
            tensor[:, t + 1 :] = torch.randn(B, T - t - 1, d, generator=g)
        out = attn_fn(q2, k2, v2)
        # A correct veil passes this bitwise: masked weights are exactly 0.0 and
        # 0.0 * (anything finite) is 0.0. The tolerance is for float noise only.
        if not torch.allclose(out[:, : t + 1], base[:, : t + 1], atol=1e-6, rtol=0.0):
            return True
    return False
