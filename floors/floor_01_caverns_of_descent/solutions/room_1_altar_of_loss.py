"""ROOM 1.1 - THE ALTAR OF LOSS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every loss here is a *mean* over the elements of the prediction, so every
gradient carries a 1/N. The two "from logits" losses never exponentiate a
large positive number: that is the whole point of writing them that way.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- regression
def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean of (y_pred - y_true)**2 over all elements."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """d mse / d y_pred = 2 (y_pred - y_true) / N."""
    return 2.0 * (y_pred - y_true) / y_pred.size


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean of |y_pred - y_true| over all elements."""
    return float(np.mean(np.abs(y_pred - y_true)))


def mae_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """d mae / d y_pred = sign(y_pred - y_true) / N (subgradient 0 at exact ties)."""
    return np.sign(y_pred - y_true) / y_pred.size


# ------------------------------------------------- binary classification
def binary_cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """-mean(y log p + (1 - y) log(1 - p)) with p clipped into [eps, 1 - eps]."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def binary_cross_entropy_grad(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """d bce / d p = (p - y) / (p (1 - p)) / N, evaluated at the clipped p."""
    p = np.clip(p, eps, 1.0 - eps)
    return (p - y) / (p * (1.0 - p)) / p.size


def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-z)) without ever overflowing.

    sigmoid(z) = exp(-softplus(-z)) and np.logaddexp(0, -z) is a stable softplus(-z).
    (0.5 * (1 + tanh(z / 2)) is another safe form.)
    """
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """mean(softplus(z) - y z), where softplus(z) = log(1 + exp(z)) = logaddexp(0, z).

    Equivalent to binary_cross_entropy(sigmoid(z), y) but exact for huge |z|:
    softplus(1000) is 1000, not inf, because logaddexp computes
    max(z, 0) + log1p(exp(-|z|)) under the hood.
    """
    z = np.asarray(z, dtype=float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def bce_with_logits_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d/dz of bce_with_logits = (sigmoid(z) - y) / N.  The classic 'p - y'."""
    z = np.asarray(z, dtype=float)
    return (sigmoid(z) - y) / z.size


# --------------------------------------------- multi-class classification
def log_softmax(logits: np.ndarray) -> np.ndarray:
    """log(softmax(logits)) along the last axis, computed as z - logsumexp(z).

    Subtracting the row max first makes every exp() argument <= 0, so nothing
    overflows, and the max cancels out of the result exactly.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean over the batch of -log softmax(logits)[i, labels[i]]."""
    n = logits.shape[0]
    logp = log_softmax(logits)
    return float(-np.mean(logp[np.arange(n), labels]))


def softmax_cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(softmax(logits) - onehot(labels)) / N, shape (N, C)."""
    n = logits.shape[0]
    grad = np.exp(log_softmax(logits))
    grad[np.arange(n), labels] -= 1.0
    return grad / n
