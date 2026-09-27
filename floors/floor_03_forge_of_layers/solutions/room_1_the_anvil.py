"""ROOM 3.1 - THE ANVIL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every layer is a function with a memory. ``forward`` computes the output and
stashes whatever ``backward`` will need in ``self.cache``. ``backward`` takes
the gradient of the loss with respect to the output (``dout``, same shape as
the output) and returns the gradient with respect to the input (``dx``, same
shape as the input), adding parameter gradients into ``self.grads`` on the way.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class. Subclasses fill in forward/backward; this holds the dicts."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset every gradient to zeros, in place, so the next backward starts fresh."""
        for g in self.grads.values():
            g[...] = 0.0


class Linear(Layer):
    """out = x @ W + b with W (in_features, out_features) and b (out_features,)."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        # LeCun-style default: std 1/sqrt(fan_in). Room 2 explains why the scale matters.
        self.params["W"] = rng.standard_normal((in_features, out_features)) / np.sqrt(in_features)
        self.params["b"] = np.zeros(out_features)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x  # backward needs the input to form dW = x.T @ dout
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        # out[n, j] = sum_i x[n, i] W[i, j] + b[j]
        #   dL/dW[i, j] = sum_n x[n, i] dout[n, j]   ->  x.T @ dout    (in, out)
        #   dL/db[j]    = sum_n dout[n, j]           ->  dout.sum(0)   (out,)
        #   dL/dx[n, i] = sum_j dout[n, j] W[i, j]   ->  dout @ W.T    (N, in)
        self.grads["W"] += x.T @ dout
        self.grads["b"] += dout.sum(axis=0)
        return dout @ self.params["W"].T


class ReLU(Layer):
    """out = max(x, 0). Gradient passes where x > 0 and is blocked elsewhere."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x > 0
        return np.where(self.cache, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.cache


class Tanh(Layer):
    """out = tanh(x). d tanh / dx = 1 - tanh(x)^2, so cache the OUTPUT, not the input."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.cache = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (1.0 - self.cache**2)


class SoftmaxCrossEntropy:
    """Softmax over the last axis + mean negative log-likelihood, fused for stability.

    forward(logits, y) -> float   logits (N, C), y (N,) integer classes
    backward()         -> (N, C)  dL/dlogits = (softmax(logits) - onehot(y)) / N
    """

    def __init__(self) -> None:
        self.cache = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        n = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)  # exp(1000) is inf; exp(0) is not
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probs = np.exp(log_probs)
        self.cache = (probs, y)
        return float(-log_probs[np.arange(n), y].mean())

    def backward(self) -> np.ndarray:
        probs, y = self.cache
        n = probs.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        return dlogits / n
