"""ROOM 6.5 - THE ROTARY GALLERY

    A side gallery off the hall, lined with heads mounted on brass rings.
    Each ring is turned a little further than the one before it, so that
    two heads facing each other can tell, from the angle between them alone,
    how many seats apart they sit. And at the far end, a curious economy:
    four heads ask questions, but only one head answers for all of them.

Everything you built so far has no idea where anything is. Permute the tokens
of x and Room 6.3's MultiHeadAttention permutes its output the same way and
changes nothing else. A model that reads "the cat sat" and "sat the cat"
identically needs positions from somewhere. GPT-2 adds a learned vector per
position before the first layer (Floor 7's ``wpe``). Almost every model since
uses ROTARY POSITION EMBEDDINGS (RoPE) instead, and pairs them with
GROUPED-QUERY ATTENTION (GQA). This room builds both.

RoPE. Split a query's head_dim coordinates into head_dim/2 consecutive pairs
(x[0], x[1]), (x[2], x[3]), ... Each pair is a point in the plane. At
position m, rotate pair i by the angle m * theta_i:

    theta_i = base ** (-2i / head_dim)        i = 0 .. head_dim/2 - 1, base = 10000
    x[2i]   <- x[2i] * cos(m theta_i) - x[2i+1] * sin(m theta_i)
    x[2i+1] <- x[2i] * sin(m theta_i) + x[2i+1] * cos(m theta_i)

Do this to q and to k, NOT to v, before the scores. Why it works: writing
R(a) for the 2-D rotation by angle a, R(a)^T R(b) = R(b - a), so

    (R(m theta) q) . (R(n theta) k) = q^T R((n - m) theta) k

The score between a query at m and a key at n depends only on n - m. Shift
every position by the same offset and no score moves; that is what lets a
model trained at positions 0..T decode token 5000 against a KV cache. Pair 0
turns one radian per seat; the last pair turns about 1/base radians per seat
(for head_dim=64 a full circle takes ~47,000 seats). Fast pairs resolve
neighbours, slow pairs tell 10 from 1000. Raising the base slows every pair
but the first.

Convention: rotating consecutive pairs (x[2i], x[2i+1]) is the *interleaved*
layout of the original paper and of this room. Llama and most Hugging Face
code pair x[i] with x[i + head_dim/2] instead ("rotate_half"). Same maths,
different column order; weights trained under one layout must be permuted to
run under the other.

GQA. In multi-head attention every head carves its own k and v, and at decode
time all of them are cached for every past token. Each new token then reads
that whole cache from memory: decode is bound by KV-cache bandwidth, not by
arithmetic. GQA keeps n_heads query heads but only n_kv_heads key/value heads;
the n_heads / n_kv_heads query heads of a group share one k and one v
(``repeat_interleave`` along the head axis). n_kv_heads == n_heads is plain
MHA; n_kv_heads == 1 is multi-query attention. Llama 3 uses 32 heads and 8 kv
heads: a 4x smaller cache, four times as many sequences served at once, and
quality within noise of MHA.

The Prophecy at the bottom asks four things the trial then measures.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .room_1_the_single_gaze import scaled_dot_product_attention  # noqa: F401 - use it below
from .room_2_veil_of_causality import causal_mask  # noqa: F401 - use it in RoPEAttention.forward
from .room_3_the_thousand_heads import MultiHeadAttention


# ---------------------------------------------------------------------------
# ROTARY POSITION EMBEDDINGS
# ---------------------------------------------------------------------------
def rope_frequencies(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """The frequency ladder: a (head_dim // 2,) float32 tensor with theta_i = base ** (-2i / head_dim).

    theta_0 is exactly 1.0 and the entries shrink geometrically towards about
    1 / base. ``torch.arange(0, head_dim, 2)`` gives you 2i for every pair.
    Raise ``ValueError`` if head_dim is odd (pairs need an even count).
    """
    raise NotImplementedError("rope_frequencies() is unwritten")


def rope_angles(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """positions: (T,) ints -> (T, head_dim // 2) float32, angles[m, i] = positions[m] * theta_i.

    An outer product: ``positions[:, None] * theta[None, :]``. Convert the
    positions to float32 first. Row 0 (position 0) is all zeros.
    """
    raise NotImplementedError("rope_angles() is unwritten")


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate every consecutive pair of x by its position's angle. Shape in == shape out.

    x: (B, H, T, head_dim)  (any leading axes; only the last two matter)
    positions: (T,) ints, one per row of the T axis.

    angles = rope_angles(positions, head_dim, base) is (T, head_dim/2) and
    broadcasts against ``x[..., 0::2]`` and ``x[..., 1::2]``, which are the
    first and second coordinates of every pair, each (B, H, T, head_dim/2).
    Rotate: even' = even*cos - odd*sin, odd' = even*sin + odd*cos. Put the
    pairs back interleaved: ``torch.stack((even', odd'), dim=-1).flatten(-2)``.
    Position 0 must leave x untouched and every row's norm must be preserved.
    """
    raise NotImplementedError("apply_rope() is unwritten")


class RoPEAttention(MultiHeadAttention):
    """Room 6.3's heads, with the position rotated into q and k. Same qkv and proj layers.

    Inherits ``qkv``, ``proj``, ``split_heads``, ``merge_heads`` and the
    ``4d^2 + 4d`` parameter count. Adds no parameters: positions cost nothing
    in RoPE, which is one of the reasons it replaced learned tables.
    """

    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, causal: bool = True) -> None:
        """``super().__init__(d_model, n_heads)``, then store ``self.base`` and ``self.causal``.

        causal=True hangs Room 6.2's veil in forward. The trial lifts it once
        (causal=False) to compare against position-free attention.
        """
        raise NotImplementedError("RoPEAttention.__init__() is unwritten")

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        """x: (B, T, d_model), positions: (T,) ints or None (= torch.arange(T)) -> (B, T, d_model).

        1. ``self.qkv(x).split(self.d_model, dim=-1)``, split_heads each -> (B, H, T, hd).
        2. ``apply_rope`` on q and on k with the same positions and ``self.base``. NOT on v.
        3. Room 6.1's attention with ``causal_mask(T)`` if self.causal else no mask.
        4. merge_heads, proj.
        Positions need not start at 0: a query at 100 hearing a key at 97
        must give exactly what 3 hearing 0 gives.
        """
        raise NotImplementedError("forward() is unwritten")


# ---------------------------------------------------------------------------
# GROUPED-QUERY ATTENTION
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """n_heads query heads that share n_kv_heads key/value heads."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        """Build the layers. Contract (the trial reads these by name):

            self.d_model    = d_model
            self.n_heads    = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim   = d_model // n_heads
            self.groups     = n_heads // n_kv_heads          query heads per kv head
            self.q_proj     = nn.Linear(d_model, d_model)
            self.k_proj     = nn.Linear(d_model, n_kv_heads * head_dim)
            self.v_proj     = nn.Linear(d_model, n_kv_heads * head_dim)
            self.proj       = nn.Linear(d_model, d_model)

        Raise ``ValueError`` if d_model % n_heads or n_heads % n_kv_heads is
        not 0. Parameters: 2 d^2 + 2 d for q_proj and proj, plus
        2 * n_kv_heads * head_dim * (d + 1) for k_proj and v_proj together.
        """
        raise NotImplementedError("GroupedQueryAttention.__init__() is unwritten")

    def split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """(B, T, n_heads * head_dim) -> (B, n_heads, T, head_dim).

        Room 6.3's split with the head count as an argument, because q has
        n_heads heads and k, v have n_kv_heads: ``view(B, T, n_heads, head_dim).transpose(1, 2)``.
        """
        raise NotImplementedError("split_heads() is unwritten")

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, d_model). Exactly Room 6.3's merge."""
        raise NotImplementedError("merge_heads() is unwritten")

    def expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """(B, n_kv_heads, T, head_dim) -> (B, n_heads, T, head_dim).

        Query head h must read kv head ``h // groups``: heads 0..groups-1 share
        kv head 0, the next ``groups`` share kv head 1, and so on. That is
        ``kv.repeat_interleave(self.groups, dim=1)`` (``repeat`` would give
        0, 1, 0, 1 instead of 0, 0, 1, 1 and torch's ``enable_gqa`` would
        disagree with you). If RoPE is in play, rotate k *before* expanding:
        the rotation only touches the last axis, so it costs n_kv_heads heads
        instead of n_heads.
        """
        raise NotImplementedError("expand_kv() is unwritten")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). mask: bool, broadcastable to (B, H, T, T), True = may attend.

        q = split_heads(q_proj(x), n_heads); k, v = expand_kv(split_heads(k_proj(x), n_kv_heads))
        and likewise for v; Room 6.1's attention; merge; proj. With
        n_kv_heads == n_heads this is Room 6.3's module with its qkv weight
        cut into three; with n_kv_heads == 1 it is multi-query attention.
        """
        raise NotImplementedError("forward() is unwritten")


def kv_cache_bytes(n_layer: int, T: int, n_kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    """Bytes the KV cache holds for ONE sequence of T tokens. Return a Python int.

    Every layer caches a key and a value (that is the 2) for every token, for
    every kv head, head_dim numbers each, bytes_per bytes per number (2 for
    bf16 / fp16). Nothing here depends on n_heads: that is the whole point of
    GQA. The saving over MHA is the factor n_heads / n_kv_heads.
    """
    raise NotImplementedError("kv_cache_bytes() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY OF THE GALLERY
# Answer before you run anything; the trial measures each one.
#
#   "the pair that turns fastest"                          "first" or "last"
#       theta_i = base^(-2i/head_dim). Which pair has the largest angle per seat?
#   "shift every position by 7, the scores are"            "unchanged" or "changed"
#       q at m and k at n become q at m+7 and k at n+7. Does q . k move?
#   "raise the base from 1e4 to 1e6, far positions turn"   "less" or "more"
#       Compare the angles at position 1024 for both bases.
#   "kv-cache saving for 32 heads and 8 kv heads"          an int
#       By what factor does GQA shrink the cache for a Llama-3-like layer?
# ---------------------------------------------------------------------------
GALLERY_PROPHECY: dict[str, str | int | None] = {
    "the pair that turns fastest": None,
    "shift every position by 7, the scores are": None,
    "raise the base from 1e4 to 1e6, far positions turn": None,
    "kv-cache saving for 32 heads and 8 kv heads": None,
}
