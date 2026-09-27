"""SECRET - THE CALIBRATION MIRROR  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A model is calibrated when "90% confident" is right 90% of the time. ECE bins
predictions by confidence and averages the gap between confidence and accuracy
inside each bin, weighted by how full the bin is. Temperature scaling divides
the logits by a single scalar T chosen to minimise NLL on held-out data; T > 1
softens an over-confident model without changing a single argmax.
"""

from __future__ import annotations

import numpy as np


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of logits / temperature, computed stably. (N, K) -> (N, K)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def mean_nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the correct class under softmax(logits / T)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    labels = np.asarray(labels).astype(np.int64).ravel()
    return float(-log_probs[np.arange(labels.shape[0]), labels].mean())


def _bin_index(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin b covers (b/n, (b+1)/n]; bin 0 also takes confidence 0."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.digitize(confidences, edges[1:-1], right=True)


def reliability_table(confidences, correct, n_bins: int = 10) -> np.ndarray:
    """(n_bins, 3): [count, mean confidence, accuracy] per bin; NaN where the bin is empty."""
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    hit = np.asarray(correct, dtype=np.float64).ravel()
    if conf.shape != hit.shape:
        raise ValueError("confidences and correct must pair up")
    idx = _bin_index(conf, n_bins)
    table = np.full((n_bins, 3), np.nan)
    for b in range(n_bins):
        mask = idx == b
        table[b, 0] = mask.sum()
        if mask.any():
            table[b, 1] = conf[mask].mean()
            table[b, 2] = hit[mask].mean()
    return table


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """sum_b (n_b / N) * |accuracy_b - confidence_b| over the non-empty bins."""
    table = reliability_table(confidences, correct, n_bins)
    n = table[:, 0].sum()
    if n == 0:
        return 0.0
    filled = table[:, 0] > 0
    return float(np.sum(table[filled, 0] / n * np.abs(table[filled, 2] - table[filled, 1])))


def temperature_scale(logits, labels, grid) -> float:
    """The temperature in ``grid`` that minimises mean NLL of softmax(logits / T) at the labels."""
    grid = [float(t) for t in grid]
    if not grid:
        raise ValueError("the grid is empty")
    losses = [mean_nll(logits, labels, t) for t in grid]
    return grid[int(np.argmin(losses))]
