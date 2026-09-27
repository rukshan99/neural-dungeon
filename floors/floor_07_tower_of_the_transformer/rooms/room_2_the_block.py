"""ROOM 7.2 - THE BLOCK

    One floor of the tower, complete. Every floor above it is this floor
    again, with different weights. Down the middle runs the stairwell: the
    residual stream, which nothing is allowed to replace, only to add to.

CAUSAL SELF-ATTENTION, shape by shape, for a stream x of (B, T, D) with H
heads and head size hd = D / H:

    qkv = c_attn(x)                        (B, T, 3D)   one Linear, D -> 3D
    q, k, v = qkv.split(D, dim=2)          three of (B, T, D), in that order
    q = q.view(B, T, H, hd).transpose(1, 2)              (B, H, T, hd), same for k, v
    att = q @ k.transpose(-2, -1) / sqrt(hd)             (B, H, T, T)  query x key
    att = att.masked_fill(~mask[:, :, :T, :T], -inf)     no looking at later keys
    att = softmax(att, dim=-1)                           each query's weights sum to 1
    y = att @ v                                          (B, H, T, hd)
    y = y.transpose(1, 2).contiguous().view(B, T, D)     merge the heads back
    out = c_proj(y)                                      (B, T, D)

The mask is ``torch.tril(torch.ones(block_size, block_size, dtype=bool))``
shaped (1, 1, block_size, block_size). Register it with
``self.register_buffer("mask", ..., persistent=False)``: it is derived, not
learned, and the Chronicler's checkpoint does not contain it. The scale is
1/sqrt(hd), the HEAD dimension, so the dot products stay O(1) whatever D is.

THE BLOCK is pre-norm:

    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))

Each sub-layer reads a normalised copy of the stream and adds its answer
back onto the raw stream. The stream itself is never normalised in place.
Parameter count: attention 4D^2 + 4D, MLP 8D^2 + 5D, two norms 4D, total
12D^2 + 13D. The names ln_1, attn, ln_2, mlp, c_attn, c_proj are the
checkpoint's names.

Use your own LayerNorm and MLP from room 1 (imported below).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Build the block from your own bricks.
from .room_1_norm_and_nonlinearity import MLP, LayerNorm  # noqa: F401


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Attributes the trial looks for: ``c_attn`` (Linear D -> 3D), ``c_proj``
    (Linear D -> D), ``n_head``, and a non-persistent ``mask`` buffer.
    Dropout modules (``attn_dropout`` on the weights, ``resid_dropout`` on
    the output) are identity at p=0. Raise ValueError (or assert) if
    n_embd is not divisible by n_head.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool = True, dropout: float = 0.0):
        raise NotImplementedError("CausalSelfAttention() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D) for any T <= block_size."""
        raise NotImplementedError("CausalSelfAttention.forward() is unwritten")


class Block(nn.Module):
    """x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).

    Sub-modules, with these names: ``ln_1`` (LayerNorm), ``attn``
    (CausalSelfAttention), ``ln_2`` (LayerNorm), ``mlp`` (MLP).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool = True, dropout: float = 0.0):
        raise NotImplementedError("Block() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)."""
        raise NotImplementedError("Block.forward() is unwritten")
