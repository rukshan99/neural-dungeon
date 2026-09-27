"""ROOM 7.1 - THE NORM AND THE NONLINEARITY

    The tower's ground floor is a mason's yard. Three kinds of brick are cut
    here, and every floor above is built from nothing else: a norm that makes
    each vector comparable to its neighbours, a curve that bends without
    breaking, and a two-layer forge that widens and narrows.

LAYERNORM normalises every vector along the last dimension on its own:

    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)      # divide by D, not D-1
    y    = (x - mean) / sqrt(var + eps) * weight + bias

For a (B, T, D) stream that is B*T separate normalisations, one per token
position. ``weight`` starts at ones and ``bias`` at zeros, so a fresh
LayerNorm is the identity on already-normalised input. eps goes INSIDE the
square root. The trial compares outputs and gradients with nn.LayerNorm to
1e-6, so write it with ordinary tensor ops and let autograd do the rest.

RMSNORM is LayerNorm with the centring and the bias removed:

    y = x * rsqrt(mean(x^2, -1, keepdim=True) + eps) * weight

One reduction instead of two, D parameters instead of 2D, the same shape
contract. On a vector whose mean is already zero it equals a bias-free
LayerNorm with the same eps; on any other vector it does not. Scaling the
input by c > 0 leaves the output unchanged (up to eps). Llama and most
models since use it in place of LayerNorm. The trial compares outputs and
gradients with nn.RMSNorm(ndim, eps=eps) to 1e-6.

GELU is x * Phi(x), where Phi is the standard normal CDF:

    gelu(x)      = 0.5 * x * (1 + erf(x / sqrt(2)))                     exact
    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))  GPT-2's approximation

Unlike ReLU it is smooth and lets a little of the negative side through:
gelu(-1) is about -0.159, not 0.

The MLP is the position-wise feed-forward sub-layer: ``fc`` widens D to 4D,
GELU bends, ``proj`` narrows 4D back to D. Same shape in and out, so it can
be added back onto the residual stream. 8D^2 + 5D parameters. The attribute
names ``fc`` and ``proj`` are not a suggestion: in room 3 you will load a
checkpoint whose keys are ``mlp.fc.weight`` and ``mlp.proj.weight``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension, from scratch.

    Parameters: ``weight`` (ndim,) initialised to ones and ``bias`` (ndim,)
    initialised to zeros, or ``bias = None`` when ``bias=False``. Store
    ``eps`` and use it. Must match ``nn.LayerNorm(ndim, eps=eps, bias=bias)``
    in outputs and gradients to 1e-6.
    """

    def __init__(self, ndim: int, eps: float = 1e-5, bias: bool = True):
        raise NotImplementedError("LayerNorm() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., ndim) -> (..., ndim), each trailing vector normalised on its own."""
        raise NotImplementedError("LayerNorm.forward() is unwritten")


class RMSNorm(nn.Module):
    """Root-mean-square normalisation over the last dimension, from scratch.

    One parameter, ``weight`` (ndim,) initialised to ones. No mean
    subtraction, no bias. Store ``eps`` and use it inside the root. Must
    match ``nn.RMSNorm(ndim, eps=eps)`` in outputs and gradients to 1e-6.
    """

    def __init__(self, ndim: int, eps: float = 1e-6):
        raise NotImplementedError("RMSNorm() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., ndim) -> (..., ndim): x * rsqrt(mean(x^2) + eps) * weight, per trailing vector."""
        raise NotImplementedError("RMSNorm.forward() is unwritten")


def gelu(x: torch.Tensor) -> torch.Tensor:
    """The exact GELU via ``torch.erf``. Must match ``F.gelu(x)`` to 1e-6."""
    raise NotImplementedError("gelu() is unwritten")


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """The tanh approximation. Within 1e-3 of the exact GELU; equal to ``F.gelu(x, approximate="tanh")`` to 1e-6."""
    raise NotImplementedError("gelu_tanh() is unwritten")


class MLP(nn.Module):
    """proj(gelu(fc(x))): D -> 4D -> D, with dropout on the way out.

    ``fc`` and ``proj`` are ``nn.Linear`` layers with those exact names,
    both with a bias when ``bias=True``. ``dropout`` is an ``nn.Dropout``
    (identity at p=0 and in eval mode). Use your own ``gelu``.
    """

    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0):
        raise NotImplementedError("MLP() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., D)."""
        raise NotImplementedError("MLP.forward() is unwritten")
