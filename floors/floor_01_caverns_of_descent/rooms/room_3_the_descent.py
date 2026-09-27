"""ROOM 1.3 - THE DESCENT

    The corridor tilts. Water runs down the middle of it, always the same way.
    You do not need a map to find the bottom of a cavern: you need the slope
    under your feet and the nerve to keep stepping.

Gradient descent is the whole algorithm:

    params <- params - lr * grad(params)

The gradient points uphill (the direction of steepest increase), so you step
against it, scaled by the learning rate lr. Repeat. Everything else on this
floor is about choosing lr well and about the shape of the valley.

The model here is the smallest one worth fitting: y_hat = X w + b with the
mean squared error. Write the residual r = X w + b - y (shape (N,)); then

    loss   = mean(r^2)
    d/dw   = (2/N) X^T r        shape (D,)
    d/db   = (2/N) sum(r)       scalar

Check both against the oracle from Room 1.2 if you doubt them; the trial does.

The last part of the room is about *conditioning*. If one feature ranges over
[0, 1] and another over [0, 10], the loss surface is a canyon: steep across
the big feature, flat along the small one. The stable lr is set by the steep
wall, so progress along the flat direction is a crawl. Standardizing the
columns (subtract mean, divide by std) turns the canyon into a bowl, and the
same fit takes a handful of steps instead of hundreds. Room 1.4 makes this
precise; the Lich at the end of the floor lives in a canyon nobody can
standardize away.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def linear_forward(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predictions of a linear model: X (N, D) @ w (D,) + b -> (N,)."""
    raise NotImplementedError("linear_forward() is unwritten")


def linear_loss_and_grads(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[float, np.ndarray, float]:
    """MSE loss of the linear model on (X, y), and its gradients.

    Returns (loss, grad_w, grad_b): loss a Python float, grad_w shape (D,),
    grad_b a Python float. y has shape (N,). Formulas in the module docstring.
    """
    raise NotImplementedError("linear_loss_and_grads() is unwritten")


def gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    lr: float,
    steps: int,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generic gradient descent on a single parameter array.

    grad_fn(params) returns the gradient, same shape as params. Perform exactly
    `steps` updates params <- params - lr * grad_fn(params). After the k-th
    update (k = 1, 2, ..., steps) call callback(k, params) if a callback was given.

    Return (final params, history), where history is a list of COPIES of params:
    history[0] is the starting point and history[k] the value after k updates,
    so len(history) == steps + 1. Never modify the array passed in; the caller
    may still be holding it. (Start with a copy; `params - lr * g` also makes a
    new array, `params -= lr * g` does not.)
    """
    raise NotImplementedError("gradient_descent() is unwritten")


def standardize_features(X: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column of X (N, D). Returns (X_std, mean, std).

    X_std = (X - mean) / (std + eps) with mean and std of shape (D,) computed
    over axis 0. Returning the statistics matters: at prediction time new data
    must be transformed with the TRAINING mean and std, not its own.
    """
    raise NotImplementedError("standardize_features() is unwritten")


def fit_linear_regression(X: np.ndarray, y: np.ndarray, lr: float, steps: int) -> tuple[np.ndarray, float, list[float]]:
    """Fit y ~ X w + b by gradient descent on the MSE, starting from w = 0, b = 0.

    Returns (w, b, losses) where losses[k] is the loss at the parameters used to
    compute update k (i.e. BEFORE that update), so len(losses) == steps. Use
    linear_loss_and_grads; you may drive it with gradient_descent by packing
    (w, b) into one vector, or write the loop directly.
    """
    raise NotImplementedError("fit_linear_regression() is unwritten")
