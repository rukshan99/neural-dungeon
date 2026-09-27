"""ROOM 13.3 - THE CANARY CAGE

    A brass cage the size of a wardrobe, with a slot in the top. A sliver of
    the day's traffic drops through the slot to a new model waiting inside.
    If the model sings, more traffic follows. If it stops singing, the cage
    door slams and the old model takes everything back. Nobody in the
    watchtower has an opinion about the new model. They have the cage.

A CANARY release sends a fraction of live traffic to a candidate and compares
it with the incumbent on the same days. Four pieces, each small:

ASSIGNMENT. ``sha256(f"{salt}:{request_id}")`` -> take the first 8 bytes as an
integer -> divide by 2**64 -> a uniform number u in [0, 1). Canary if
``u < canary_fraction``. Hashing makes it deterministic (the same request or
user always lands in the same arm, with no lookup table), and comparing ONE
uniform against the fraction makes ramps monotone: everyone in the 5% canary is
still in the 25% canary. Change the salt to reshuffle for a new experiment.

COMPARISON. The two arms saw DIFFERENT requests, so Floor 11's paired bootstrap
does not apply. Resample each arm with replacement on its own, compute
mean(canary*) - mean(baseline*) each round, and take percentiles:

    delta = mean(canary) - mean(baseline)
    ci    = (alpha/2, 1 - alpha/2) percentiles of the bootstrap deltas
    p     = min(1, 2 * min(P(delta* <= 0), P(delta* >= 0)))

A 100-request canary against a 900-request baseline has an interval about
+-7 points wide at 85% accuracy. It cannot see a two-point change. It is not
supposed to; it is supposed to catch disasters cheaply. Ramp for the rest.

GUARDRAILS. Quality is not the only thing that can go wrong. p95 latency,
error rate and cost per request each get a hard limit (the SLO). A metric the
canary did not report is a FAILED guardrail: no gauge, no promotion.

THE DECISION, written once as code:

    rollback  if any guardrail fails, or the CI is entirely below 0
    promote   if the CI is entirely above 0, delta >= min_effect, and guardrails pass
    extend    otherwise (not enough evidence yet: keep collecting)

``min_effect`` separates "statistically significant" from "worth shipping".

THE RAMP. 1% -> 5% -> 25% -> 50% -> 100%. Each "promote" moves one step up;
"extend" stays; "rollback" goes to 0. Ramping is what turns a canary that can
only see disasters into a release that has been checked at every scale.
"""

from __future__ import annotations

import hashlib  # noqa: F401 - sha256 for assign_arm
from collections.abc import Sequence

import numpy as np

DEFAULT_RAMP = (0.01, 0.05, 0.25, 0.5, 1.0)


# ----------------------------------------------------------------- assignment


def assign_arm(request_id: str, canary_fraction: float, salt: str = "") -> str:
    """"canary" or "baseline" for this request, deterministically.

    ``hashlib.sha256(f"{salt}:{request_id}".encode()).digest()[:8]`` -> ``int.from_bytes(..., "big") / 2**64``
    is uniform in [0, 1). Canary when it is below ``canary_fraction``. Raise ValueError unless
    0 <= canary_fraction <= 1. Fraction 0 sends nobody, fraction 1 sends everybody.
    """
    raise NotImplementedError("assign_arm() is unwritten")


# ----------------------------------------------------------------- comparison


def compare_arms(
    baseline_scores,
    canary_scores,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Unpaired bootstrap of delta = mean(canary) - mean(baseline).

    Resample EACH arm independently with replacement (``rng.integers(0, n, size=(n_boot, n))`` per arm),
    compute the difference of means per round, and return
    {"delta", "ci_low", "ci_high", "p_value", "n_baseline", "n_canary"} as Python numbers.
    ``rng=None`` means ``np.random.default_rng(0)``. Raise ValueError if either arm is empty.
    (Constant arms give an interval that collapses onto delta and a p-value of 0.)
    """
    raise NotImplementedError("compare_arms() is unwritten")


# ----------------------------------------------------------------- guardrails


def guardrail_check(canary_metrics: dict, slo: dict) -> dict:
    """Every key of ``slo`` is a maximum the canary must respect.

    Return {"passed": bool, "violations": [metric names, in slo order], "details": {name: {"observed", "limit", "ok"}}}.
    A metric present in ``slo`` but missing from ``canary_metrics`` is a violation.
    """
    raise NotImplementedError("guardrail_check() is unwritten")


def release_decision(comparison: dict, guardrails: dict, min_effect: float = 0.0) -> str:
    """"promote" | "rollback" | "extend" from a compare_arms() result and a guardrail_check() result.

    rollback if not guardrails["passed"] or comparison["ci_high"] < 0;
    promote if comparison["ci_low"] > 0 and comparison["delta"] >= min_effect (guardrails already passed);
    extend otherwise.
    """
    raise NotImplementedError("release_decision() is unwritten")


# ----------------------------------------------------------------------- ramp


def ramp_schedule(steps: Sequence[float] = DEFAULT_RAMP) -> list[float]:
    """Validate a ramp and return it as a list of floats.

    Raise ValueError unless the steps are strictly increasing, every step is in (0, 1], and the last
    step is exactly 1.0 (full traffic). An empty ramp is invalid.
    """
    raise NotImplementedError("ramp_schedule() is unwritten")


def advance_ramp(fraction: float, decision: str, steps: Sequence[float] = DEFAULT_RAMP) -> float:
    """The canary fraction after ``decision``: "promote" -> the first step strictly above ``fraction``
    (1.0 if there is none); "extend" -> ``fraction`` unchanged; "rollback" -> 0.0. Other decisions raise ValueError."""
    raise NotImplementedError("advance_ramp() is unwritten")
