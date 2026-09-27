"""ROOM 0.2 - THE HALL OF SHAPES

    A long gallery. Every statue has been rearranged overnight, and each plinth
    bears only a shape: (B, H, W, C). (N, 2, D). (1, H, W). Put the statues back.

Two ideas do most of the work in this room:

* ``reshape`` changes how the *same* memory is interpreted. It never moves data.
  ``x.reshape(B, -1)`` lets numpy compute one dimension for you.
* ``transpose`` (or ``np.moveaxis``) changes the *order of axes*. It also returns a
  view, but the elements you see are now in a different order. Reshaping when you
  meant to transpose is the most common shape bug in the dungeon.

Also useful: ``x[None, ...]`` and ``x[:, None]`` insert an axis of size 1;
``np.stack`` adds a new axis, ``np.concatenate`` extends an existing one.

Then the Prophecy: fill in SHAPE_PROPHECY *before* you run anything. The trial
evaluates every expression and compares it with your prediction. Guessing and
then fixing is allowed, but you learn more by committing first.
"""

from __future__ import annotations

import numpy as np


def flatten_batch(x: np.ndarray) -> np.ndarray:
    """(B, ...) -> (B, everything_else). Keep the batch axis, flatten the rest.

    flatten_batch(np.zeros((4, 3, 5))).shape == (4, 15)
    """
    raise NotImplementedError("flatten_batch() is unwritten")


def to_channels_first(x: np.ndarray) -> np.ndarray:
    """(B, H, W, C) -> (B, C, H, W). Move the channel axis. Do not reorder pixels."""
    raise NotImplementedError("to_channels_first() is unwritten")


def stack_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two (N, D) arrays -> one (N, 2, D) array with out[i, 0] = a[i] and out[i, 1] = b[i]."""
    raise NotImplementedError("stack_pairs() is unwritten")


def last_column(x: np.ndarray) -> np.ndarray:
    """(N, D) -> (N,) holding the last column of x."""
    raise NotImplementedError("last_column() is unwritten")


def every_other_row(x: np.ndarray) -> np.ndarray:
    """Rows 0, 2, 4, ... of a 2-D array, as a 2-D array."""
    raise NotImplementedError("every_other_row() is unwritten")


def add_batch_axis(x: np.ndarray) -> np.ndarray:
    """(H, W) -> (1, H, W): a single example dressed up as a batch of one."""
    raise NotImplementedError("add_batch_axis() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY
# Replace every None with the shape tuple you *predict*. Write (3,) for 1-D.
# Do not run the expressions first. Commit, then let the trial judge you.
# ---------------------------------------------------------------------------
SHAPE_PROPHECY: dict[str, tuple[int, ...] | None] = {
    "np.zeros((3, 4)).T": None,
    "np.zeros((2, 3, 4)).reshape(6, -1)": None,
    "np.zeros((5,))[:, None]": None,
    "np.zeros((2, 3, 4)).sum(axis=1)": None,
    "np.zeros((2, 3, 4)).mean(axis=(0, 2))": None,
    "np.zeros((2, 3, 4)).sum(axis=1, keepdims=True)": None,
    "np.zeros((3, 1, 4)) + np.zeros((5, 4))": None,
    "np.zeros((4, 3)) @ np.zeros((3, 2))": None,
    "np.zeros((2, 3, 4)).transpose(2, 0, 1)": None,
    "np.zeros((6,))[None, :, None]": None,
}
