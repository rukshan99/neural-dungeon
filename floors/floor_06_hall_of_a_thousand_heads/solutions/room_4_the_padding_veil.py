"""ROOM 6.4 - THE PADDING VEIL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Padding is masked along the KEY axis and differs per batch element, so the
combined mask is (B, 1, T, T): batch, a size-1 head axis to broadcast over,
queries, keys. safe_softmax turns the all -inf rows (which softmax would make
NaN) into rows of zeros.
"""

from __future__ import annotations

import math

import torch

from .room_2_veil_of_causality import causal_mask
from .room_3_the_thousand_heads import MultiHeadAttention


def key_padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(B,) lengths -> (B, T) bool, True at real positions (t < lengths[b])."""
    lengths = torch.as_tensor(lengths)
    return torch.arange(T)[None, :] < lengths[:, None]


def combine_masks(causal: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """(T, T) & (B, T) along the keys -> (B, 1, T, T)."""
    return causal[None, None, :, :] & padding[:, None, None, :]


def safe_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked softmax over the last axis; fully masked rows become zeros instead of NaN."""
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    alive = mask.any(dim=-1, keepdim=True)  # (..., Tq, 1): does this row hear anyone at all?
    return weights.masked_fill(~alive, 0.0)  # masked_fill overwrites the NaNs too


def masked_mha(mha: MultiHeadAttention, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Causal + key-padding multi-head self-attention built from mha's pieces. Always finite."""
    B, T, _ = x.shape
    q, k, v = mha.qkv(x).split(mha.d_model, dim=-1)
    q, k, v = mha.split_heads(q), mha.split_heads(k), mha.split_heads(v)  # (B, H, T, hd)
    scores = q @ k.transpose(-2, -1) / math.sqrt(mha.head_dim)  # (B, H, T, T)
    mask = combine_masks(causal_mask(T), key_padding_mask(lengths, T))  # (B, 1, T, T)
    weights = safe_softmax(scores, mask)  # (B, H, T, T)
    return mha.proj(mha.merge_heads(weights @ v))
