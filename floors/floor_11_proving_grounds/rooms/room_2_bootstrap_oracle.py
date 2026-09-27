"""ROOM 11.2 - THE BOOTSTRAP ORACLE

    Behind a curtain sits an oracle with a bag of numbered stones. She will not
    tell you the future. She will tell you how many different futures your
    numbers are consistent with, by pulling the stones out again and again,
    always putting each one back before the next draw.

Any score computed on n items is an ESTIMATE. Run the same system on a
different n items and you get a different number. The bootstrap estimates that
spread without any formula for the statistic: treat the sample as the
population, draw n items WITH REPLACEMENT, recompute the statistic, repeat B
times. The middle 95% of the B recomputations is a 95% percentile confidence
interval.

Comparing two systems, resample ITEM INDICES and take both systems' scores at
those indices together. Items you both got right or both got wrong cancel; the
paired bootstrap only sees where you differ, which is why it can detect a small
consistent gain that two separate intervals would miss.

Rule of thumb: the standard error of a proportion is sqrt(p(1-p)/n). At n = 50
and p = 0.8 that is 0.057, so a 95% interval is about +-0.11 wide. Two systems
four points apart at n = 50 are indistinguishable, however real the gap is.
The normal-approximation sample size for a margin m is

    n = z^2 p (1 - p) / m^2        (z = 1.96 for 95%)

Fill in ORACLE_PROPHECY before running the trial. Every function that draws
random numbers takes an ``rng`` (a ``np.random.Generator``); use it and only it,
so that the same seed always gives the same interval. ``rng=None`` means
``np.random.default_rng(0)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values,
    statistic: Callable = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic(values)`` as (low, high) Python floats.

    ``n_boot`` resamples of size n drawn with replacement (``rng.integers(0, n, size=(n_boot, n))``
    gives all the indices at once). low = the 100*alpha/2 percentile of the resampled statistics,
    high = the 100*(1-alpha/2) percentile. ``statistic`` receives a 1-D array and returns a scalar;
    do not assume it accepts ``axis``. Raise ValueError on an empty sample.
    """
    raise NotImplementedError("bootstrap_ci() is unwritten")


def paired_bootstrap(
    a_scores,
    b_scores,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """(delta_mean, ci_low, ci_high, p_value_two_sided) for delta = mean(b_scores - a_scores).

    a_scores[i] and b_scores[i] are two systems' scores on the SAME item i; raise ValueError if
    the lengths differ. Resample indices once per bootstrap round and use the same indices for
    both arrays. The two-sided p-value is the percentile-bootstrap one:
        p = min(1, 2 * min(P(delta* <= 0), P(delta* >= 0)))
    over the bootstrap deltas. Identical arrays give (0.0, 0.0, 0.0, 1.0).
    """
    raise NotImplementedError("paired_bootstrap() is unwritten")


def is_significant(ci: tuple[float, float]) -> bool:
    """True when the interval (low, high) does not contain 0.0."""
    raise NotImplementedError("is_significant() is unwritten")


def required_sample_size(p: float, margin: float, z: float = 1.96) -> int:
    """ceil(z^2 p (1 - p) / margin^2): items needed to estimate a proportion near ``p`` to +-``margin``.

    Raise ValueError unless 0 < p < 1 and margin > 0. Return a Python int.
    """
    raise NotImplementedError("required_sample_size() is unwritten")


# ---------------------------------------------------------------------------
# THE ORACLE'S PROPHECY
# In each scenario two systems are scored (1 = right, 0 = wrong) on the SAME n
# items and end up with exactly the stated accuracies. The trial runs a paired
# bootstrap with a 95% interval. Will the interval exclude zero? Write
# "distinguishable" or "not distinguishable" for each. Reason from the standard
# error before you run anything.
# ---------------------------------------------------------------------------
ORACLE_PROPHECY: dict[str, str | None] = {
    "n=50: 80% vs 84%": None,
    "n=2000: 80% vs 84%": None,
    "n=200: 60% vs 75%": None,
}
