"""ROOM 7.1 - THE NORM AND THE NONLINEARITY  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Three small pieces that every block of the tower is built from: LayerNorm,
GELU and the two-layer MLP. Each one is a few lines; the trials check them
against PyTorch's own to 1e-6 (outputs *and* gradients for LayerNorm).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalise each vector along the LAST dimension, then scale and shift.

    y = (x - mean) / sqrt(var + eps) * weight + bias

    ``mean`` and ``var`` are taken over the last dim only, per position, so a
    (B, T, D) input is normalised as B*T independent D-vectors. The variance
    is the *biased* one (divide by D, not D-1), exactly like ``nn.LayerNorm``.
    """

    def __init__(self, ndim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = x_hat * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU: x * Phi(x), where Phi is the standard normal CDF.

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """The tanh approximation used by GPT-2: within ~1e-3 of the exact GELU."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class MLP(nn.Module):
    """The position-wise feed-forward sub-layer: widen 4x, GELU, narrow back.

    fc:   (.., D) -> (.., 4D)
    gelu: elementwise
    proj: (.., 4D) -> (.., D)

    Parameter count: fc has 4D*D + 4D, proj has D*4D + D, total 8D^2 + 5D.
    """

    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(gelu(self.fc(x))))
