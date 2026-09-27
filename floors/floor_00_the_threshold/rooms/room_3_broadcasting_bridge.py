"""ROOM 0.3 - THE BROADCASTING BRIDGE

    A rope bridge over a chasm. A sign reads: NO LOOPS. The planks are held up by
    broadcasting alone. Write a `for` and the trial will feel the bridge sway.

Broadcasting is how numpy combines arrays of different shapes without copying.
The rules (memorize them; the boss checks):

    1. Align the two shapes at their RIGHT edge. Pad the shorter one with 1s on
       the left:      (N, D) and (D,)  ->  (N, D) and (1, D)
    2. Walk the dimensions pairwise. Each pair must be equal, or one must be 1.
    3. A 1 is stretched to match the other size. The result takes the larger.
    4. Any other mismatch is an error.

The tool you will use most is *inserting a size-1 axis* with ``None`` (a.k.a.
``np.newaxis``): ``a[:, None]`` turns (N,) into (N, 1) so it can broadcast
against (N, M) or against ``b[None, :]`` of shape (1, M). ``keepdims=True`` on a
reduction is the same trick applied automatically.

No Python loops in this file: no for, while, or comprehensions. The trial reads
your source.
"""

from __future__ import annotations

import numpy as np


def center_rows(x: np.ndarray) -> np.ndarray:
    """Subtract each row's mean from that row. (N, D) -> (N, D). Every row then sums to ~0.

    Hint: ``x.mean(axis=1)`` has shape (N,), which will NOT broadcast against (N, D)
    the way you want. You need (N, 1). Look at ``keepdims``.
    """
    raise NotImplementedError("center_rows() is unwritten")


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit L2 norm. A row of zeros must stay zeros, not become NaN.

    That is what ``eps`` is for: divide by (norm + eps).
    """
    raise NotImplementedError("normalize_rows() is unwritten")


def standardize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score every column: (x - column_mean) / (column_std + eps). (N, D) -> (N, D)."""
    raise NotImplementedError("standardize_columns() is unwritten")


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,), (M,) -> (N, M) where out[i, j] = a[i] + b[j]."""
    raise NotImplementedError("outer_sum() is unwritten")


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D), (M, D) -> (N, M) Euclidean distances, out[i, j] = ||a[i] - b[j]||.

    Build the (N, M, D) tensor of differences with two inserted axes, square,
    sum over D, sqrt. (Floor 0.4 shows a leaner way; here the tensor is the lesson.)
    """
    raise NotImplementedError("pairwise_distances() is unwritten")


def scale_channels(images: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Multiply every channel of channels-last images (B, H, W, C) by scale (C,).

    This one is a single line. Notice *why* it works: rule 1 aligns C with C.
    """
    raise NotImplementedError("scale_channels() is unwritten")


def clamp_to_range(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clip each column j of x (N, D) into [low[j], high[j]]. low and high have shape (D,).

    ``np.minimum`` / ``np.maximum`` broadcast like any other operation.
    """
    raise NotImplementedError("clamp_to_range() is unwritten")
