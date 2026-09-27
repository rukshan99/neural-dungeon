"""SECRET - THE SEQUENTIAL SENTINEL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Pair the arms' outcomes. Pairs where both succeeded or both failed say nothing
about which arm is better, so drop them. Among the DISCORDANT pairs, "B won" is
a coin flip under the null (q = 1/2) and biased towards B under the
alternative (q = q1). That turns a two-sample comparison into Wald's one-sample
SPRT on a Bernoulli stream: add log(q1 / 0.5) for each B win, log((1 - q1) /
0.5) for each A win, stop when the sum leaves (log B, log A).
"""

from __future__ import annotations

import math
from statistics import NormalDist


def discordant_probability(p_a: float, p_b: float) -> float:
    """P(B succeeded | exactly one of the two succeeded) = p_b (1 - p_a) / (p_b (1 - p_a) + p_a (1 - p_b))."""
    if not 0.0 < p_a < 1.0 or not 0.0 < p_b < 1.0:
        raise ValueError("rates must be strictly between 0 and 1")
    b_only = p_b * (1.0 - p_a)
    a_only = p_a * (1.0 - p_b)
    return b_only / (b_only + a_only)


def wald_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper) = (log(beta / (1 - alpha)), log((1 - beta) / alpha))."""
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must be strictly between 0 and 1")
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


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
        self._win = math.log(self.q1 / 0.5)
        self._loss = math.log((1.0 - self.q1) / 0.5)
        self.llr = 0.0
        self.n_pairs = 0
        self.n_discordant = 0
        self.decision: str | None = None  # "b_better" | "no_difference" | None while undecided

    def update(self, a_success, b_success) -> str | None:
        """Feed one (A outcome, B outcome) pair. Returns the decision once made; None while undecided."""
        if self.decision is not None:
            return self.decision
        self.n_pairs += 1
        a, b = bool(a_success), bool(b_success)
        if a == b:
            return None  # concordant: no evidence either way
        self.n_discordant += 1
        self.llr += self._win if b else self._loss
        if self.llr >= self.upper:
            self.decision = "b_better"
        elif self.llr <= self.lower:
            self.decision = "no_difference"
        return self.decision

    def feed(self, a_outcomes, b_outcomes) -> str | None:
        """Feed pairs in order, stopping at the first decision. Lengths must match."""
        if len(a_outcomes) != len(b_outcomes):
            raise ValueError("a_outcomes and b_outcomes must pair up")
        for a, b in zip(a_outcomes, b_outcomes):
            if self.update(a, b) is not None:
                break
        return self.decision


def fixed_horizon_pairs(p_a: float, min_effect: float, alpha: float = 0.05, beta: float = 0.2, two_sided: bool = False) -> int:
    """Pairs a fixed-horizon z-test needs: ceil((z_alpha + z_beta)^2 (p_a(1-p_a) + p_b(1-p_b)) / min_effect^2)."""
    if min_effect <= 0.0:
        raise ValueError("min_effect must be positive")
    p_b = p_a + min_effect
    if not 0.0 < p_a < 1.0 or not 0.0 < p_b < 1.0:
        raise ValueError("rates must stay strictly between 0 and 1")
    z_alpha = NormalDist().inv_cdf(1.0 - (alpha / 2.0 if two_sided else alpha))
    z_beta = NormalDist().inv_cdf(1.0 - beta)
    variance = p_a * (1.0 - p_a) + p_b * (1.0 - p_b)
    return int(math.ceil((z_alpha + z_beta) ** 2 * variance / min_effect**2))
