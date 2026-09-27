"""SECRET - THE SEQUENTIAL SENTINEL   (optional)

    A lamp on the parapet, and beside it a sentinel who looks at the canary's
    numbers every single morning and has never once fooled herself. Ask her
    how. "I decided what would convince me," she says, "before I looked."

THE PEEKING PROBLEM. A 95% confidence interval is a promise about ONE look. If
you check the canary every day for a month and stop the moment the interval
excludes zero, you have taken thirty looks, and the chance that at least one
of them lies is closer to 20% than 5%. The Watchtower boss peeks daily; one of
its seeds rolled back a better model on a chance interval. Fixed-horizon tests
fix this by never peeking. Sequential tests fix it by being DESIGNED for
peeking.

WALD'S SPRT ON DISCORDANT PAIRS. Pair the arms' outcomes (the i-th baseline
request with the i-th canary request; any pairing works as long as it ignores
the outcomes). A pair where both succeeded or both failed says nothing about
which arm is better: drop it. Among the DISCORDANT pairs, "B won" happens with
probability

    q = p_b (1 - p_a) / (p_b (1 - p_a) + p_a (1 - p_b))

which is exactly 1/2 when p_a = p_b, and q1 > 1/2 when B is ``min_effect``
better. That turns the two-sample question into a one-sample test on a coin:

    H0: q = 1/2        H1: q = q1
    each B win  adds  ln(q1 / 0.5)        to the log-likelihood ratio
    each A win  adds  ln((1 - q1) / 0.5)
    stop with "b_better"      when  llr >= ln((1 - beta) / alpha)
    stop with "no_difference" when  llr <= ln(beta / (1 - alpha))

Wald's bounds guarantee (approximately) false-positive rate <= alpha and
false-negative rate <= beta NO MATTER HOW OFTEN YOU LOOK, and the test usually
stops well before the fixed-horizon sample size. The trial runs 200 A/A streams
(the sentinel should declare a winner about alpha of the time; a naive daily
z-test does so about four times as often) and 200 streams with a real 5-point
lift (right most of the time, and early).
"""

from __future__ import annotations

import math  # noqa: F401 - log for the likelihood ratio and the bounds
from statistics import NormalDist  # noqa: F401 - inv_cdf for the fixed-horizon comparison


def discordant_probability(p_a: float, p_b: float) -> float:
    """P(B succeeded | exactly one of the two succeeded) = p_b (1 - p_a) / (p_b (1 - p_a) + p_a (1 - p_b)).

    Raise ValueError unless both rates are strictly between 0 and 1. Equal rates give exactly 0.5.
    """
    raise NotImplementedError("discordant_probability() is unwritten")


def wald_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper) = (ln(beta / (1 - alpha)), ln((1 - beta) / alpha)). ValueError outside (0, 1)."""
    raise NotImplementedError("wald_bounds() is unwritten")


class SequentialSentinel:
    """SPRT on discordant pairs: H0 q = 1/2 (no difference) vs H1 q = q1 (B is min_effect better)."""

    def __init__(self, p_a: float, min_effect: float, alpha: float = 0.05, beta: float = 0.2) -> None:
        if min_effect <= 0.0:
            raise ValueError("min_effect must be positive: the smallest improvement worth detecting")
        self.p_a = float(p_a)
        self.min_effect = float(min_effect)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q1 = discordant_probability(p_a, p_a + min_effect)
        self.lower, self.upper = wald_bounds(alpha, beta)
        self.llr = 0.0
        self.n_pairs = 0
        self.n_discordant = 0
        self.decision: str | None = None  # "b_better" | "no_difference" | None while undecided

    def update(self, a_success, b_success) -> str | None:
        """Feed one (A outcome, B outcome) pair; return the decision once made, None while undecided.

        Once decided, ignore further pairs (return the decision, count nothing). Otherwise count the
        pair; if the outcomes are equal, return None; else count it as discordant, add ln(q1 / 0.5) for
        a B win or ln((1 - q1) / 0.5) for an A win, and decide when ``llr`` reaches a Wald bound.
        """
        raise NotImplementedError("SequentialSentinel.update() is unwritten")

    def feed(self, a_outcomes, b_outcomes) -> str | None:
        """Feed pairs in order, stopping at the first decision; return it (None if the stream ran out).
        Raise ValueError if the two sequences differ in length."""
        raise NotImplementedError("SequentialSentinel.feed() is unwritten")


def fixed_horizon_pairs(p_a: float, min_effect: float, alpha: float = 0.05, beta: float = 0.2, two_sided: bool = False) -> int:
    """Pairs a fixed-horizon z-test needs for the same alpha and power:

        ceil((z_alpha + z_beta)^2 * (p_a (1 - p_a) + p_b (1 - p_b)) / min_effect^2),   p_b = p_a + min_effect

    with z_alpha = NormalDist().inv_cdf(1 - alpha) one-sided (1 - alpha/2 two-sided) and
    z_beta = NormalDist().inv_cdf(1 - beta). ValueError if min_effect <= 0 or a rate leaves (0, 1).
    """
    raise NotImplementedError("fixed_horizon_pairs() is unwritten")
