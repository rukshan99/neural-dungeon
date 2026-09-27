"""ROOM 1.1 - THE ALTAR OF LOSS

    The first cavern is a chapel. On the altar, a brass gauge with a single
    needle. It does not measure distance or time. It measures how wrong you
    are. Every corridor on this floor slopes toward the needle's zero.

A *loss function* turns (prediction, truth) into one non-negative number, and
training is nothing but walking that number downhill. To walk downhill you
need the slope, so every loss here comes with its gradient with respect to
the prediction. Conventions for the whole room:

* Losses are the MEAN over all elements (or over the batch for softmax
  cross-entropy), returned as a plain Python float. Because of the mean,
  every gradient carries a factor 1/N.
* Gradients have exactly the shape of the prediction they differentiate.
* Two losses take *logits* (raw scores before sigmoid/softmax). Computing
  them by first squashing and then taking logs is a numerical trap:
  sigmoid(1000) rounds to 1.0, log(1 - 1.0) is -inf, and your training run
  is over. The fix is algebra, not clipping:

      log(1 + exp(z))   ==  max(z, 0) + log(1 + exp(-|z|))     (softplus)
      log softmax(z)_i  ==  z_i - max(z) - log(sum_j exp(z_j - max(z)))

  Both forms only ever exponentiate numbers <= 0. np.logaddexp(0, z) is a
  ready-made stable softplus; np.log1p is log(1 + x) accurate near x = 0.

The trial checks known values, symmetry, that huge logits stay finite, and
that every gradient matches a numerical derivative. Room 1.2 builds the
oracle that does that check; here the trial carries its own.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- regression
def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error: mean over ALL elements of (y_pred - y_true) ** 2.

    y_pred and y_true have the same shape (any shape). Return a Python float.
    """
    raise NotImplementedError("mse() is unwritten")


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """d mse / d y_pred, same shape as y_pred.

    Differentiate mean((p - t)^2) = (1/N) sum (p - t)^2 with respect to each p_i.
    N is the total number of elements (y_pred.size).
    """
    raise NotImplementedError("mse_grad() is unwritten")


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean absolute error: mean over all elements of |y_pred - y_true|. Python float."""
    raise NotImplementedError("mae() is unwritten")


def mae_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """d mae / d y_pred, same shape as y_pred.

    d|u|/du is sign(u); at u == 0 it is undefined and any value in [-1, 1] is a
    valid subgradient. np.sign gives 0 there, which is fine.
    """
    raise NotImplementedError("mae_grad() is unwritten")


# ------------------------------------------------- binary classification
def binary_cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """BCE from probabilities: -mean(y log p + (1 - y) log(1 - p)).

    p holds probabilities in [0, 1]; y holds labels in {0, 1} (as floats).
    Clip p into [eps, 1 - eps] first so that p == 0 or p == 1 gives a large
    finite loss instead of inf/NaN. Return a Python float.
    """
    raise NotImplementedError("binary_cross_entropy() is unwritten")


def binary_cross_entropy_grad(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """d bce / d p, same shape as p, evaluated at the CLIPPED p.

    Differentiate -(y log p + (1 - y) log(1 - p)) with respect to p and put
    the two terms over a common denominator: (p - y) / (p (1 - p)). Then the 1/N.
    """
    raise NotImplementedError("binary_cross_entropy_grad() is unwritten")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-z)) for any real z, WITHOUT overflow warnings.

    The naive formula computes exp(1000) for z = -1000. Either use two branches
    (np.where on the sign of z), or note sigmoid(z) = exp(-softplus(-z)) and use
    np.logaddexp, or use 0.5 * (1 + tanh(z / 2)). sigmoid(1000) must be exactly
    1.0 and sigmoid(-1000) must be 0.0.
    """
    raise NotImplementedError("sigmoid() is unwritten")


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """BCE straight from logits: mean(softplus(z) - y * z). Python float.

    Derivation: -[y log s + (1-y) log(1-s)] with s = sigmoid(z) simplifies to
    softplus(z) - y z, where softplus(z) = log(1 + exp(z)). Use the stable
    softplus (see the module docstring). For z = 1000, y = 0 the loss of that
    element is exactly 1000.0, not inf.
    """
    raise NotImplementedError("bce_with_logits() is unwritten")


def bce_with_logits_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d/dz of bce_with_logits, same shape as z.

    d softplus(z)/dz = sigmoid(z), so the per-element derivative is
    sigmoid(z) - y. This "prediction minus target" pattern shows up for every
    loss paired with its natural output function. Then the 1/N.
    """
    raise NotImplementedError("bce_with_logits_grad() is unwritten")


# --------------------------------------------- multi-class classification
def log_softmax(logits: np.ndarray) -> np.ndarray:
    """log(softmax(logits)) along the LAST axis, same shape as logits.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j). Taking the log:
    log softmax(z)_i = z_i - logsumexp(z), and a stable logsumexp subtracts
    the max first: m + log(sum exp(z - m)). Every row of exp(log_softmax(z))
    must sum to 1, and adding a constant to a row must not change the output.
    """
    raise NotImplementedError("log_softmax() is unwritten")


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean over the batch of -log softmax(logits)[i, labels[i]]. Python float.

    logits: (N, C) float scores. labels: (N,) integer class indices in [0, C).
    Pick the log-probability of the correct class for each row with fancy
    indexing (logp[np.arange(N), labels]), negate, average over N.
    """
    raise NotImplementedError("softmax_cross_entropy() is unwritten")


def softmax_cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """d/d logits of softmax_cross_entropy, shape (N, C).

    Per row: softmax(z) - onehot(label). Then divide by N for the mean. Each row
    of the result sums to zero (probabilities sum to 1, so does a one-hot).
    """
    raise NotImplementedError("softmax_cross_entropy_grad() is unwritten")
