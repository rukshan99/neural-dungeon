"""ROOM 0.4 - THE SPEEDRUN GALLERY  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
Each function here replaces a slow loop with array operations.
"""

from __future__ import annotations

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """(N,) integer labels -> (N, num_classes) float32 one-hot matrix."""
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """(N,) -> (N - k + 1,) where out[i] = mean(x[i : i + k]). Use a cumulative sum."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    return (c[k:] - c[:-k]) / k


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax that does not overflow for large inputs."""
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def count_neighbors_within(points: np.ndarray, radius: float) -> np.ndarray:
    """(N, D) -> (N,) ints: for each point, how many *other* points lie within radius.

    Uses |a - b|^2 = |a|^2 + |b|^2 - 2 a.b so no (N, N, D) tensor is ever built.
    """
    sq = (points**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * points @ points.T
    d2 = np.maximum(d2, 0.0)
    within = d2 <= radius**2
    return within.sum(axis=1) - 1
