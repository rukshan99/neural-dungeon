"""ROOM 5.3 - THE EMBEDDING WELL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

An embedding is a lookup table. Indexing a row and multiplying a one-hot vector
by the table are the same operation (same output, same gradient); the lookup is
just the version that does not build a V-wide vector of zeros first.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def embedding_lookup(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(V, D) table, (...) int64 ids -> (..., D). Plain advanced indexing."""
    return weight[ids]


def embedding_as_matmul(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The same lookup written as one_hot(ids) @ weight."""
    one_hot = nn.functional.one_hot(ids, num_classes=weight.shape[0]).to(weight.dtype)
    return one_hot @ weight


def token_embedding(vocab_size: int, d_model: int, padding_idx: int = 0) -> nn.Embedding:
    """An nn.Embedding whose padding row starts at zero and never receives gradient."""
    return nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) float32 table with

        PE[pos, 2i]     = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i + 1] = cos(pos / 10000^(2i / d_model))
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even so that sin/cos pairs line up, got {d_model}")
    position = torch.arange(max_len, dtype=torch.float32)[:, None]  # (max_len, 1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model / 2,)
    angle_rate = torch.exp(-math.log(10000.0) * two_i / d_model)  # 1 / 10000^(2i/d)
    angles = position * angle_rate  # (max_len, d_model / 2)
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class LearnedPositionalEmbedding(nn.Module):
    """A second lookup table, indexed by position instead of token id."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) token embeddings -> (B, T, D) with position added."""
        _, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}: no row to look up")
        positions = torch.arange(T, device=x.device)
        return x + self.pos(positions)  # (T, D) broadcasts over the batch axis
