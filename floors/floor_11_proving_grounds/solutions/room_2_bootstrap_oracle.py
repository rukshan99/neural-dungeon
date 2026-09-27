"""ROOM 11.2 - THE BOOTSTRAP ORACLE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The bootstrap treats your sample as the population: resample it with
replacement many times, recompute the statistic each time, and the spread of
those recomputations is your estimate of the sampling error. The paired
version resamples ITEM INDICES so that both systems' scores on the same item
travel together; that is what makes it sensitive to small consistent gains.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values,
    statistic: Callable = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic(values)``.

    Draw ``n_boot`` resamples of size n with replacement, compute the statistic
    on each, and return the (100*alpha/2, 100*(1-alpha/2)) percentiles as
    (low, high). ``rng=None`` means ``np.random.default_rng(0)``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(values, dtype=np.float64).ravel()
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([statistic(x[row]) for row in idx], dtype=np.float64)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def paired_bootstrap(
    a_scores,
    b_scores,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """(delta_mean, ci_low, ci_high, p_two_sided) for delta = mean(b - a), resampling paired indices.

    The p-value is the percentile-bootstrap version: twice the smaller of
    P(delta* <= 0) and P(delta* >= 0) over the bootstrap deltas, capped at 1.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(a_scores, dtype=np.float64).ravel()
    b = np.asarray(b_scores, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired scores must line up item by item: got {a.shape[0]} vs {b.shape[0]}")
    n = a.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    d = b - a
    delta = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))  # the SAME indices for a and b
    deltas = d[idx].mean(axis=1)
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return delta, float(low), float(high), min(1.0, p)


def is_significant(ci: tuple[float, float]) -> bool:
    """True when the interval does not contain 0 (a difference you can defend)."""
    low, high = ci
    return bool(low > 0.0 or high < 0.0)


def required_sample_size(p: float, margin: float, z: float = 1.96) -> int:
    """n = z^2 p (1 - p) / margin^2, rounded UP. Normal approximation for a proportion."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    return int(math.ceil(z * z * p * (1.0 - p) / (margin * margin)))


# ---------------------------------------------------------------------------
# THE ORACLE'S PROPHECY
# For each scenario, two systems are scored on the same n items with the stated
# accuracies. Can a paired bootstrap (95% CI) tell them apart?
# ---------------------------------------------------------------------------
ORACLE_PROPHECY: dict[str, str | None] = {
    "n=50: 80% vs 84%": "not distinguishable",
    "n=2000: 80% vs 84%": "distinguishable",
    "n=200: 60% vs 75%": "distinguishable",
}
