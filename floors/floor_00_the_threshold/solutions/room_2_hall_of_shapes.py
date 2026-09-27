"""ROOM 0.2 - THE HALL OF SHAPES  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import numpy as np


def flatten_batch(x: np.ndarray) -> np.ndarray:
    """(B, ...) -> (B, everything_else). Keep the batch axis, flatten the rest."""
    return x.reshape(x.shape[0], -1)


def to_channels_first(x: np.ndarray) -> np.ndarray:
    """(B, H, W, C) -> (B, C, H, W). Move the channel axis without touching the data."""
    return np.transpose(x, (0, 3, 1, 2))


def stack_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two (N, D) arrays -> one (N, 2, D) array where out[i, 0] = a[i], out[i, 1] = b[i]."""
    return np.stack([a, b], axis=1)


def last_column(x: np.ndarray) -> np.ndarray:
    """(N, D) -> (N,) holding the last column."""
    return x[:, -1]


def every_other_row(x: np.ndarray) -> np.ndarray:
    """Rows 0, 2, 4, ... of a 2-D array."""
    return x[::2]


def add_batch_axis(x: np.ndarray) -> np.ndarray:
    """(H, W) -> (1, H, W). A single example pretending to be a batch of one."""
    return x[None, ...]


# The Prophecy: write the shape each expression produces, *before* running it.
# The trial evaluates the expressions and compares.
SHAPE_PROPHECY: dict[str, tuple[int, ...]] = {
    "np.zeros((3, 4)).T": (4, 3),
    "np.zeros((2, 3, 4)).reshape(6, -1)": (6, 4),
    "np.zeros((5,))[:, None]": (5, 1),
    "np.zeros((2, 3, 4)).sum(axis=1)": (2, 4),
    "np.zeros((2, 3, 4)).mean(axis=(0, 2))": (3,),
    "np.zeros((2, 3, 4)).sum(axis=1, keepdims=True)": (2, 1, 4),
    "np.zeros((3, 1, 4)) + np.zeros((5, 4))": (3, 5, 4),
    "np.zeros((4, 3)) @ np.zeros((3, 2))": (4, 2),
    "np.zeros((2, 3, 4)).transpose(2, 0, 1)": (4, 2, 3),
    "np.zeros((6,))[None, :, None]": (1, 6, 1),
}
