"""ROOM 1.3 - THE DESCENT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def linear_forward(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """(N, D) @ (D,) + b -> (N,)."""
    return X @ w + b


def linear_loss_and_grads(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[float, np.ndarray, float]:
    """MSE of the linear model and its gradients w.r.t. w and b.

    residual r = X w + b - y   (N,)
    loss     = mean(r^2)
    d/dw     = 2/N X^T r       (D,)
    d/db     = 2/N sum(r)      scalar
    """
    n = y.shape[0]
    residual = linear_forward(X, w, b) - y
    loss = float(np.mean(residual**2))
    grad_w = 2.0 / n * X.T @ residual
    grad_b = float(2.0 / n * residual.sum())
    return loss, grad_w, grad_b


def gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    lr: float,
    steps: int,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """params <- params - lr * grad_fn(params), repeated `steps` times.

    Returns (final params, history) where history[k] is a copy of params after
    k updates (so history[0] is the starting point and len(history) == steps + 1).
    The input array is never modified.
    """
    params = np.array(params, dtype=float)  # a copy: the caller's array stays put
    history = [params.copy()]
    for k in range(1, steps + 1):
        params = params - lr * grad_fn(params)
        history.append(params.copy())
        if callback is not None:
            callback(k, params)
    return params, history


def standardize_features(X: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every column. Returns (X_std, mean, std) so the same transform can be reused."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + eps), mean, std


def fit_linear_regression(X: np.ndarray, y: np.ndarray, lr: float, steps: int) -> tuple[np.ndarray, float, list[float]]:
    """Gradient descent on the MSE from w = 0, b = 0.

    losses[k] is the loss at the parameters *before* update k (len == steps).
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    losses: list[float] = []
    for _ in range(steps):
        loss, grad_w, grad_b = linear_loss_and_grads(X, y, w, b)
        losses.append(loss)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b, losses
