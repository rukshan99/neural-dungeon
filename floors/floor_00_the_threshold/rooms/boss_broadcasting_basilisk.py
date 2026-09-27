"""BOSS - THE BROADCASTING BASILISK

                 .-~~~-.
               .'  o o  `.        "You lean on the machine to tell you
              /  .-'''-.  \\        which shapes fit. Lean, then, and fall."
             |  /  ___  \\  |
             | |  (o o)  | |         The Basilisk's gaze PETRIFIES numpy's
             |  \\  `-'  /  |         broadcasting helpers. During the fight,
              \\  `-...-'  /          np.broadcast_shapes, np.broadcast_to,
               `.  ~~~  .'           np.broadcast_arrays and np.broadcast
              ,-'`-...-'`-.          all raise. Some of its eyes have
             /   (ssss)    \\         10**15 elements, so you cannot even
                                     build the arrays to look.

WEAKNESS: someone who knows the four broadcasting rules by heart and can apply
them with plain Python on shape tuples.

The fight has three phases:

  Phase 1  broadcast_shape(*shapes)  - the rules, applied by hand.
  Phase 2  along_axis / scale_along / petrified - use rule 1 on purpose, by
           inserting size-1 axes exactly where you need them.
  Phase 3  BASILISK_RIDDLES - stare back: predict ten broadcasts, some of which
           are errors.

Run:  dungeon fight 0
"""

from __future__ import annotations

import numpy as np


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The shape numpy would produce by broadcasting ``shapes`` together, or None if it can't.

    Apply the rules on the tuples directly:
      1. pad shorter shapes with leading 1s to the longest length,
      2. for each position, the dims must all be equal or 1,
      3. the result dim is the non-1 value (or 1 if all are 1),
      4. any conflict -> return None.

    ``broadcast_shape()`` with no arguments returns ``()``.
    Do NOT create arrays or call numpy's broadcast helpers. The trial is watching.
    """
    raise NotImplementedError("broadcast_shape() is unwritten")


def along_axis(v: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    """Reshape a 1-D vector so it broadcasts along ``axis`` of an ``ndim``-D array.

    along_axis(np.array([1, 2, 3]), ndim=4, axis=1).shape == (1, 3, 1, 1)
    Negative axes are allowed (axis=-1 means the last axis). Raise ValueError if
    v is not 1-D.
    """
    raise NotImplementedError("along_axis() is unwritten")


def scale_along(x: np.ndarray, scale: np.ndarray, axis: int) -> np.ndarray:
    """Multiply x by a per-index ``scale`` along ``axis``.

    Example: per-channel gain for channels-FIRST images (B, C, H, W) with axis=1.
    """
    raise NotImplementedError("scale_along() is unwritten")


def petrified(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out ``x`` wherever ``mask`` is True. ``mask`` has x's shape WITHOUT the last axis.

    x: (B, T, D), mask: (B, T) -> every feature vector x[b, t, :] with mask[b, t]
    becomes zeros. Works for any number of leading dims. ``np.where`` is your friend.
    """
    raise NotImplementedError("petrified() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 3: STARE BACK
# For each pair of shapes, write the broadcast result as a tuple, or the string
# "error" if numpy would refuse. Predict first; the trial checks against numpy.
# ---------------------------------------------------------------------------
BASILISK_RIDDLES: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...] | str | None] = {
    ((8, 1, 6, 1), (7, 1, 5)): None,
    ((5, 4), (1,)): None,
    ((5, 4), (4,)): None,
    ((15, 3, 5), (15, 1, 5)): None,
    ((15, 3, 5), (3, 5)): None,
    ((15, 3, 5), (3, 1)): None,
    ((3,), (4,)): None,
    ((2, 1), (8, 4, 3)): None,
    ((4, 1), (4,)): None,
    ((1, 1, 1), ()): None,
}
