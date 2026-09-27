"""ROOM 6.3 - THE THOUSAND HEADS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The layout here (fused qkv Linear, contiguous head chunks, output projection)
is the one torch.nn.MultiheadAttention uses internally, which is why the trial
can copy the weights across:

    twin.in_proj_weight  <- qkv.weight     (3*d_model, d_model): rows [0:d] q, [d:2d] k, [2d:3d] v
    twin.in_proj_bias    <- qkv.bias
    twin.out_proj.weight <- proj.weight
    twin.out_proj.bias   <- proj.bias

and it is the layout of the Chronicler's c_attn / c_proj on Floor 7.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .room_1_the_single_gaze import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """H heads of scaled dot-product self-attention with fused qkv and an output projection."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # one matmul produces q, k and v
        self.proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, head_dim). Head h owns columns h*hd:(h+1)*hd."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, d_model). Exact inverse of split_heads."""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        # Room 6.1's function, now with (B, H) as batch axes. Its sqrt(d) is
        # sqrt(head_dim) because head_dim is the last axis.
        heads, weights = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, H, T, hd), (B, H, T, T)
        out = self.proj(self.merge_heads(heads))
        return (out, weights) if return_weights else out
