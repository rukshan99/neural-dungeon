"""SECRET - THE BATCH NORM CRUCIBLE   (optional)

    Behind the Hydra's lair, a crucible glows white. Whatever metal is poured
    in comes out at the same temperature: mean zero, unit variance, every
    time. The smiths call it batch normalization. Nobody agrees on why it
    works so well. Everybody agrees that it does.

BatchNorm1d normalizes each FEATURE across the BATCH, then lets two learnable
vectors undo the normalization if the network wants it back:

    mu   = x.mean(axis=0)                     (D,)
    var  = x.var(axis=0)                      (D,)   biased, ddof=0
    xhat = (x - mu) / sqrt(var + eps)         (N, D) each column ~ mean 0, std 1
    out  = gamma * xhat + beta                (N, D)

Train mode also keeps running averages for test time, when a single example
has no batch to be normalized against:

    running_mean <- momentum * running_mean + (1 - momentum) * mu
    running_var  <- momentum * running_var  + (1 - momentum) * var

Eval mode uses running_mean / running_var in place of mu / var, updates
nothing, and is a fixed affine map per feature.

THE BACKWARD. mu and var depend on every row of x, so dL/dx has three paths:
directly through xhat, through mu, and through var. Written out and simplified
(with sums over the batch axis and dxhat = dout * gamma):

    dgamma = sum(dout * xhat)
    dbeta  = sum(dout)
    dx     = (1 / (N * sqrt(var + eps))) * (N * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))

The tempting shortcut dx = dxhat / sqrt(var + eps) treats mu and var as
constants. It is wrong, and the finite-difference trial will say so.
Gradients accumulate into self.grads, like every layer on this floor.
"""

from __future__ import annotations

import numpy as np


class BatchNorm1d:
    """Batch normalization over axis 0 for (N, num_features) inputs."""

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self.params = {"gamma": np.ones(num_features), "beta": np.zeros(num_features)}
        self.grads = {"gamma": np.zeros(num_features), "beta": np.zeros(num_features)}
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Train: normalize with batch statistics, update running stats, cache for backward.

        Eval: normalize with running statistics; touch nothing.
        """
        raise NotImplementedError("BatchNorm1d.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Add dgamma, dbeta into grads; return dx (N, num_features) by the three-term formula."""
        raise NotImplementedError("BatchNorm1d.backward() is unwritten")

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0
