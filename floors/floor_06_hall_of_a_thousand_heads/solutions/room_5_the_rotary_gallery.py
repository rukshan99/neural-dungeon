"""ROOM 6.5 - THE ROTARY GALLERY  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

RoPE turns each consecutive pair (x[2i], x[2i+1]) of a query or key by the
angle m * theta_i, where m is the position and theta_i = base^(-2i/head_dim).
Because 2-D rotations compose by adding angles, R(m t)^T R(n t) = R((n-m) t),
so the score between a query at m and a key at n depends only on n - m. The
values are never rotated. GQA shrinks the key and value projections to
n_kv_heads heads and repeats each one across a group of n_heads / n_kv_heads
query heads, which divides the KV cache by the same factor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .room_1_the_single_gaze import scaled_dot_product_attention
from .room_2_veil_of_causality import causal_mask
from .room_3_the_thousand_heads import MultiHeadAttention


# ---------------------------------------------------------------------------
# ROTARY POSITION EMBEDDINGS
# ---------------------------------------------------------------------------
def rope_frequencies(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """(head_dim // 2,) float32: theta_i = base ** (-2i / head_dim). theta_0 == 1, the rest shrink geometrically."""
    if head_dim % 2:
        raise ValueError(f"head_dim={head_dim} must be even: RoPE rotates pairs of coordinates")
    two_i = torch.arange(0, head_dim, 2, dtype=torch.float32)  # 0, 2, 4, ... = 2i
    return base ** (-two_i / head_dim)


def rope_angles(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """(T,) positions -> (T, head_dim // 2) float32 angles, angles[m, i] = m * theta_i."""
    positions = torch.as_tensor(positions).to(torch.float32)
    return positions[:, None] * rope_frequencies(head_dim, base)[None, :]


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate pair (x[..., 2i], x[..., 2i+1]) at position m by m * theta_i. x: (..., T, head_dim)."""
    angles = rope_angles(positions, x.shape[-1], base)  # (T, hd/2); broadcasts over the leading axes
    cos, sin = angles.cos(), angles.sin()
    x_even, x_odd = x[..., 0::2], x[..., 1::2]  # the two coordinates of every pair, (..., T, hd/2)
    out_even = x_even * cos - x_odd * sin  # the usual 2-D rotation, once per pair
    out_odd = x_even * sin + x_odd * cos
    # stack puts the pair back side by side: (..., T, hd/2, 2) -> (..., T, hd), interleaved again
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class RoPEAttention(MultiHeadAttention):
    """Room 6.3's heads with the position rotated into q and k. Same layers, same layout."""

    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, causal: bool = True) -> None:
        super().__init__(d_model, n_heads)
        self.base = base
        self.causal = causal

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T)
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)  # (B, H, T, hd)
        q = apply_rope(q, positions, self.base)  # rotate the questions and the answers...
        k = apply_rope(k, positions, self.base)
        mask = causal_mask(T) if self.causal else None  # ...but never what is handed over (v)
        heads, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.proj(self.merge_heads(heads))


# ---------------------------------------------------------------------------
# GROUPED-QUERY ATTENTION
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """n_heads query heads sharing n_kv_heads key/value heads, groups = n_heads // n_kv_heads."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} is not divisible by n_kv_heads={n_kv_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.groups = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model)  # every query head keeps its own carving
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)  # only n_kv_heads of these
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """(B, T, n_heads * head_dim) -> (B, n_heads, T, head_dim). Room 6.3's split with a head count."""
        B, T, _ = x.shape
        return x.view(B, T, n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, d_model)."""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """(B, n_kv_heads, T, hd) -> (B, n_heads, T, hd): query head h reads kv head h // groups."""
        return kv.repeat_interleave(self.groups, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.split_heads(self.q_proj(x), self.n_heads)  # (B, H, T, hd)
        k = self.expand_kv(self.split_heads(self.k_proj(x), self.n_kv_heads))  # (B, n_kv, T, hd) -> (B, H, T, hd)
        v = self.expand_kv(self.split_heads(self.v_proj(x), self.n_kv_heads))
        heads, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.proj(self.merge_heads(heads))


def kv_cache_bytes(n_layer: int, T: int, n_kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    """Bytes of cached keys AND values for one sequence of T tokens: 2 * n_layer * T * n_kv_heads * head_dim * bytes_per."""
    return 2 * n_layer * T * n_kv_heads * head_dim * bytes_per


# The Prophecy of the Gallery. theta_0 = base^0 = 1 is the largest frequency,
# so the first pair turns fastest (one radian per seat) and the last pair
# slowest (about 1/base radians per seat). Shifting every position by the same
# offset rotates every q and every k by the same extra angle, and
# R(a)^T R(b) = R(b - a) cancels it: the scores are unchanged. A larger base
# makes every theta_i with i >= 1 smaller, so far positions are turned less
# (Llama 3 raised base to 500000 for exactly this reason). The KV cache holds
# n_kv_heads heads instead of n_heads: 32 / 8 = 4 times smaller.
GALLERY_PROPHECY: dict[str, str | int] = {
    "the pair that turns fastest": "first",
    "shift every position by 7, the scores are": "unchanged",
    "raise the base from 1e4 to 1e6, far positions turn": "less",
    "kv-cache saving for 32 heads and 8 kv heads": 4,
}
