"""ROOM 7.2 - THE BLOCK  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

One floor of the tower: causal self-attention and an MLP, each wrapped in a
pre-norm residual connection. Every floor above it is a copy of this one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .room_1_norm_and_nonlinearity import MLP, LayerNorm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused q/k/v projection.

    c_attn: (B, T, D) -> (B, T, 3D), split into q, k, v of (B, T, D) each.
    Heads:  (B, T, D) -> (B, H, T, hd) with hd = D / H.
    Scores: q @ k^T / sqrt(hd) -> (B, H, T, T), future positions set to -inf,
            softmax over the last dim (the keys).
    Mix:    scores @ v -> (B, H, T, hd) -> merge heads -> (B, T, D) -> c_proj.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # The causal mask is derived, not learned, so it is not saved in the
        # checkpoint: persistent=False keeps it out of the state_dict.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge the heads
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    """A pre-norm transformer block.

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    The residual stream ``x`` is never normalised in place: each sub-layer
    reads a normalised *copy* and adds its output back to the raw stream.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias=bias, dropout=dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
