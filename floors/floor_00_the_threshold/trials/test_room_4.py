"""TRIAL 0.4 - THE SPEEDRUN GALLERY

Correctness first, then the clock. The slow reference loops live here so you
can see exactly what you are racing. Your version must be at least SPEEDUP
times faster on the same input, on your own machine.
"""

import math
import time

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_4_speedrun_gallery")

rng = np.random.default_rng(4)
SPEEDUP = 20.0


# ----------------------------------------------------------------- the slow loops
def slow_one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i in range(len(labels)):
        out[i, labels[i]] = 1.0
    return out


def slow_moving_average(x, k):
    out = np.empty(len(x) - k + 1)
    for i in range(len(out)):
        out[i] = sum(x[i : i + k]) / k
    return out


def slow_softmax_rows(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        m = max(x[i])
        exps = [math.exp(v - m) for v in x[i]]
        s = sum(exps)
        for j in range(x.shape[1]):
            out[i, j] = exps[j] / s
    return out


def slow_count_neighbors_within(points, radius):
    n = len(points)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and math.dist(points[i], points[j]) <= radius:
                counts[i] += 1
    return counts


def _time(fn, *args, repeats=3):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _race(name, fast, slow, *args):
    t_slow = _time(slow, *args, repeats=1)
    t_fast = _time(fast, *args)
    ratio = t_slow / max(t_fast, 1e-9)
    assert ratio >= SPEEDUP, (
        f"{name}: the loop took {t_slow * 1e3:.1f} ms, yours took {t_fast * 1e3:.1f} ms "
        f"({ratio:.1f}x). The gallery demands {SPEEDUP:.0f}x. Is there still a Python loop in there?"
    )


# -------------------------------------------------------------------- one_hot
def test_one_hot_is_correct():
    labels = np.array([0, 2, 1, 2])
    out = room.one_hot(labels, 3)
    assert out.shape == (4, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out, slow_one_hot(labels, 3))


def test_one_hot_speedrun():
    labels = rng.integers(0, 10, size=200_000)
    _race("one_hot", room.one_hot, slow_one_hot, labels, 10)


# ------------------------------------------------------------- moving_average
def test_moving_average_is_correct():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(room.moving_average(x, 2), [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(room.moving_average(x, 5), [3.0])
    x = rng.standard_normal(50)
    np.testing.assert_allclose(room.moving_average(x, 7), slow_moving_average(x, 7), atol=1e-12)


def test_moving_average_speedrun():
    x = rng.standard_normal(100_000)
    _race("moving_average", room.moving_average, slow_moving_average, x, 50)


# --------------------------------------------------------------- softmax_rows
def test_softmax_rows_sum_to_one_and_match_reference():
    x = rng.standard_normal((5, 6))
    out = room.softmax_rows(x)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(out, slow_softmax_rows(x), atol=1e-12)


def test_softmax_rows_survives_huge_inputs():
    x = np.array([[1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0]])
    out = room.softmax_rows(x)
    assert not np.any(np.isnan(out)), "exp(1000) overflowed. Subtract the row max before exponentiating."
    np.testing.assert_allclose(out[0], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0], atol=1e-12)


def test_softmax_rows_speedrun():
    x = rng.standard_normal((20_000, 16))
    _race("softmax_rows", room.softmax_rows, slow_softmax_rows, x)


# ----------------------------------------------------- count_neighbors_within
def _points_and_safe_radius(n, seed):
    """Points plus a radius that no pairwise distance sits within 1e-6 of (no coin flips)."""
    r = np.random.default_rng(seed)
    pts = r.random((n, 2))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    radius = 0.1
    while np.abs(d - radius).min() < 1e-6:
        radius += 1e-5
    return pts, radius


def test_count_neighbors_is_correct_and_excludes_self():
    pts, radius = _points_and_safe_radius(60, seed=7)
    out = room.count_neighbors_within(pts, radius)
    assert out.shape == (60,)
    np.testing.assert_array_equal(out, slow_count_neighbors_within(pts, radius))


def test_count_neighbors_speedrun():
    pts, radius = _points_and_safe_radius(700, seed=8)
    _race("count_neighbors_within", room.count_neighbors_within, slow_count_neighbors_within, pts, radius)
