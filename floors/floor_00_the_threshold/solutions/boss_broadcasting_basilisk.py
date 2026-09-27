"""BOSS - THE BROADCASTING BASILISK  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Basilisk petrifies numpy's broadcasting helpers, so the rules are applied by
hand. The rules:

1. Align shapes at the *right* (trailing) edge; pad the shorter with 1s on the left.
2. Two dimensions are compatible if they are equal, or one of them is 1.
3. The result dimension is the larger of the two (the 1 is stretched).
4. If any dimension pair is incompatible, broadcasting fails.
"""

from __future__ import annotations

import numpy as np


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The shape numpy would produce by broadcasting these shapes together, or None.

    Must not call numpy's broadcast helpers or build any arrays: some of the
    shapes the Basilisk shows you have 10**15 elements.
    """
    if not shapes:
        return ()
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    result = []
    for dims in zip(*padded):
        non_one = {d for d in dims if d != 1}
        if len(non_one) > 1:
            return None
        result.append(non_one.pop() if non_one else 1)
    return tuple(result)


def along_axis(v: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    """Reshape a 1-D vector so it broadcasts along ``axis`` of an ``ndim``-D array.

    along_axis(np.array([1, 2, 3]), ndim=4, axis=1).shape == (1, 3, 1, 1)
    """
    if v.ndim != 1:
        raise ValueError("along_axis expects a 1-D vector")
    axis = axis % ndim
    shape = [1] * ndim
    shape[axis] = v.shape[0]
    return v.reshape(shape)


def scale_along(x: np.ndarray, scale: np.ndarray, axis: int) -> np.ndarray:
    """Multiply x by a per-index scale along any axis, e.g. per-channel gain for (B, C, H, W)."""
    return x * along_axis(scale, x.ndim, axis)


def petrified(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out x wherever mask is True. mask has x's shape *without its last axis*.

    Example: x is (B, T, D), mask is (B, T). Every feature vector at a masked
    (b, t) becomes zeros.
    """
    return np.where(mask[..., None], 0.0, x)


# The Basilisk's riddles: for each pair of shapes, the broadcast result or "error".
BASILISK_RIDDLES: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...] | str] = {
    ((8, 1, 6, 1), (7, 1, 5)): (8, 7, 6, 5),
    ((5, 4), (1,)): (5, 4),
    ((5, 4), (4,)): (5, 4),
    ((15, 3, 5), (15, 1, 5)): (15, 3, 5),
    ((15, 3, 5), (3, 5)): (15, 3, 5),
    ((15, 3, 5), (3, 1)): (15, 3, 5),
    ((3,), (4,)): "error",
    ((2, 1), (8, 4, 3)): "error",
    ((4, 1), (4,)): (4, 4),
    ((1, 1, 1), ()): (1, 1, 1),
}
