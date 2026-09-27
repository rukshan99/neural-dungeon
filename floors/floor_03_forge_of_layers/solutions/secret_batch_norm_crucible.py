"""SECRET - THE BATCH NORM CRUCIBLE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Forward (train), per feature j over the batch of N:
    mu  = mean(x)          var = mean((x - mu)^2)            (biased, ddof=0)
    xhat = (x - mu) / sqrt(var + eps)
    out  = gamma * xhat + beta
    running <- momentum * running + (1 - momentum) * batch statistic

Backward, with dxhat = dout * gamma and sums over the batch axis:
    dgamma = sum(dout * xhat)         dbeta = sum(dout)
    dx = (1 / (N * sqrt(var + eps))) * (N * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))

The dx formula is what falls out of the chain rule once you notice that mu and
var both depend on every x in the batch: three paths (direct, through mu,
through var) that collapse into the three terms above.
"""

from __future__ import annotations

import numpy as np


class BatchNorm1d:
    """Normalize each feature over the batch, then let gamma/beta undo it if the network wants."""

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
        gamma, beta = self.params["gamma"], self.params["beta"]
        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            xhat = (x - mu) * inv_std
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
            self.cache = (xhat, inv_std)
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.cache = None
        return gamma * xhat + beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() after an eval-mode forward: BatchNorm only trains in train mode")
        xhat, inv_std = self.cache
        n = dout.shape[0]
        self.grads["gamma"] += np.sum(dout * xhat, axis=0)
        self.grads["beta"] += np.sum(dout, axis=0)
        dxhat = dout * self.params["gamma"]
        dx = (inv_std / n) * (n * dxhat - dxhat.sum(axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
        return dx

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0
