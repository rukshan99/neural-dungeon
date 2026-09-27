"""ROOM 0.3 - THE BROADCASTING BRIDGE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
No Python loops anywhere in this file: the trial checks.
"""

from __future__ import annotations

import numpy as np


def center_rows(x: np.ndarray) -> np.ndarray:
    """Subtract each row's mean from that row. (N, D) -> (N, D), every row sums to ~0."""
    return x - x.mean(axis=1, keepdims=True)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit L2 norm. Rows of all zeros stay zero (that is what eps is for)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def standardize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score every column: subtract the column mean, divide by the column std."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,), (M,) -> (N, M) where out[i, j] = a[i] + b[j]."""
    return a[:, None] + b[None, :]


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D), (M, D) -> (N, M) of Euclidean distances. Build an (N, M, D) difference tensor."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def scale_channels(images: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Multiply every channel of channels-last images (B, H, W, C) by scale (C,)."""
    return images * scale


def clamp_to_range(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clip each column j of x (N, D) into [low[j], high[j]]. low/high have shape (D,)."""
    return np.minimum(np.maximum(x, low), high)
