"""ROOM 5.3 - THE EMBEDDING WELL

    Behind the desks, a well. Drop a token id in and a vector comes back up,
    always the same vector for the same id. The well is not magic: it is a
    table with one row per token, and the id is a row number.

An embedding is a lookup table W of shape (V, D): V rows (one per vocabulary
id), D columns (the model width). Looking up ids of shape (B, T) gives (B, T, D).

Two ways to write the same thing:

    W[ids]                          advanced indexing: pick rows
    one_hot(ids, V) @ W             a matrix product with a vector that is all
                                    zeros except a single 1

They produce identical outputs and identical gradients. The gradient of the
table is a *scatter-add*: the upstream gradient of each position is added into
the row of the id at that position, so a row used twice receives two
contributions and a row never used receives zeros. The indexing version is
simply the one that does not materialize a (B, T, V) tensor of zeros.

``nn.Embedding(V, D, padding_idx=0)`` freezes one row: it is initialized to
zeros and its gradient is always zero, so padding never learns anything and
never leaks into the loss through its embedding.

Positions: attention treats its input as a *set*; without extra information a
model cannot tell "dog bites man" from "man bites dog". Two fixes, both tables
of shape (max_len, D) added to the token embeddings:

    sinusoidal   fixed, no parameters:
                 PE[pos, 2i]   = sin(pos / 10000^(2i / D))
                 PE[pos, 2i+1] = cos(pos / 10000^(2i / D))
                 A useful property: PE[p] . PE[p + k] depends only on k, not on p
                 (sin a sin b + cos a cos b = cos(a - b)), so "how far apart" is
                 easy to read off with a dot product.
    learned      an nn.Embedding indexed by position. More flexible, but a
                 position beyond max_len has no row at all.

Keep the torch here tiny. The Chronicler in dungeon/artifacts/tiny_gpt.py uses
exactly these two tables (wte and wpe).
"""

from __future__ import annotations

import torch
from torch import nn


def embedding_lookup(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(V, D) table, integer ``ids`` of any shape -> (*ids.shape, D). Index, do not multiply."""
    raise NotImplementedError("embedding_lookup() is unwritten")


def embedding_as_matmul(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The same lookup as ``one_hot(ids, V).to(weight.dtype) @ weight``.

    Must use a matrix product (``@``, ``matmul``, ``mm`` or ``einsum``). The point is
    to see that it is the same function with the same gradient.
    """
    raise NotImplementedError("embedding_as_matmul() is unwritten")


def token_embedding(vocab_size: int, d_model: int, padding_idx: int = 0) -> nn.Embedding:
    """An ``nn.Embedding`` whose ``padding_idx`` row is zero and receives no gradient."""
    raise NotImplementedError("token_embedding() is unwritten")


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) float32 table, exactly per the formula in the module docstring.

    Even columns are sines, odd columns are cosines, column pair i uses the
    angular rate 1 / 10000^(2i / d_model). Raise ValueError if ``d_model`` is odd.
    Build it with tensor ops (``torch.arange``, ``torch.exp``/``pow``, slicing ``[:, 0::2]``),
    not a double loop.
    """
    raise NotImplementedError("sinusoidal_positional_encoding() is unwritten")


class LearnedPositionalEmbedding(nn.Module):
    """A trainable (max_len, d_model) table added to token embeddings by position."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        raise NotImplementedError("LearnedPositionalEmbedding.__init__() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) token embeddings -> (B, T, D) with the position vectors added.

        Row t of the table is added to every batch element at position t. Raise
        ValueError when T exceeds max_len: there is no row for that position.
        """
        raise NotImplementedError("LearnedPositionalEmbedding.forward() is unwritten")
