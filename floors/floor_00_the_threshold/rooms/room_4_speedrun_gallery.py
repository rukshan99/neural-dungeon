"""ROOM 0.4 - THE SPEEDRUN GALLERY

    Four paintings. In each, a tiny figure runs the same corridor over and over,
    one step at a time. The corridor is a Python `for` loop. The figure is your
    CPU. Free them.

Python loops over array elements are slow because every iteration pays for
interpretation, type checks and boxing a float into an object. numpy operations
run the same loop in compiled C over a contiguous buffer, often with SIMD.
Vectorizing is not a micro-optimization: it is regularly 100x.

The trial file contains the slow reference loops. Your versions must produce the
same answers and be at least 10x faster on the trial's inputs. (The trial times
both on *your* machine, so the bar is relative, not absolute.)

Tools you will want: fancy indexing ``out[np.arange(n), labels]``, ``np.cumsum``,
``keepdims``, and the identity ``|a - b|^2 = |a|^2 + |b|^2 - 2 a.b``.
"""

from __future__ import annotations

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """(N,) integer labels -> (N, num_classes) float32 with a single 1.0 per row.

    Slow version (in the trial): a loop that sets out[i, labels[i]] = 1.
    Fast version: index all N rows at once with two index arrays.
    """
    raise NotImplementedError("one_hot() is unwritten")


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """(N,) -> (N - k + 1,) where out[i] = mean(x[i : i + k]).

    A cumulative sum turns every window sum into one subtraction:
    prepend a 0 to cumsum(x), then window_sum[i] = c[i + k] - c[i].
    """
    raise NotImplementedError("moving_average() is unwritten")


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax: exp(x) / sum(exp(x)) along axis 1, for a (N, D) array.

    exp(1000.0) overflows to inf. Subtract each row's max first; the result is
    mathematically identical and numerically safe. (You will meet this trick
    again on every floor that has a softmax, which is most of them.)
    """
    raise NotImplementedError("softmax_rows() is unwritten")


def count_neighbors_within(points: np.ndarray, radius: float) -> np.ndarray:
    """(N, D) points -> (N,) ints: for each point, how many OTHER points lie within radius.

    Do not build an (N, N, D) tensor. Use squared norms and one matrix product:
        d2[i, j] = |p_i|^2 + |p_j|^2 - 2 p_i . p_j
    Clamp tiny negatives (floating point) to 0. Exclude the point itself.
    """
    raise NotImplementedError("count_neighbors_within() is unwritten")
