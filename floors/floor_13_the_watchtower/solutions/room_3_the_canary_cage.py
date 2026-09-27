"""ROOM 13.3 - THE CANARY CAGE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Assignment hashes the request id so the same request always lands in the same
arm and raising the fraction never moves anyone OUT of the canary. The
comparison is an UNPAIRED bootstrap because the two arms saw different
requests. The decision rule is written once, as code, so that nobody argues
about it at 2 a.m.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

DEFAULT_RAMP = (0.01, 0.05, 0.25, 0.5, 1.0)


# ----------------------------------------------------------------- assignment


def assign_arm(request_id: str, canary_fraction: float, salt: str = "") -> str:
    """"canary" or "baseline", deterministically, from sha256(salt:request_id)."""
    if not 0.0 <= canary_fraction <= 1.0:
        raise ValueError("canary_fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{salt}:{request_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    return "canary" if u < canary_fraction else "baseline"


# ----------------------------------------------------------------- comparison


def compare_arms(
    baseline_scores,
    canary_scores,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Unpaired bootstrap of delta = mean(canary) - mean(baseline).

    Each arm is resampled with replacement on its own (different requests in
    each arm, so there is nothing to pair). Returns delta, the percentile
    interval, the two-sided p-value and both sample sizes.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(baseline_scores, dtype=np.float64).ravel()
    b = np.asarray(canary_scores, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("both arms need at least one score")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    deltas = np.empty(n_boot)
    rows = max(1, int(2_000_000 // max(a.size, b.size)))  # chunk so the index array stays small
    for start in range(0, n_boot, rows):
        k = min(rows, n_boot - start)
        idx_a = rng.integers(0, a.size, size=(k, a.size))
        idx_b = rng.integers(0, b.size, size=(k, b.size))
        deltas[start : start + k] = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))))
    return {
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p,
        "n_baseline": int(a.size),
        "n_canary": int(b.size),
    }


# ----------------------------------------------------------------- guardrails


def guardrail_check(canary_metrics: dict, slo: dict) -> dict:
    """Every SLO key is a maximum. A metric you did not measure is a violation."""
    details: dict[str, dict] = {}
    violations: list[str] = []
    for name, limit in slo.items():
        observed = canary_metrics.get(name)
        ok = observed is not None and float(observed) <= float(limit)
        details[name] = {"observed": observed, "limit": limit, "ok": ok}
        if not ok:
            violations.append(name)
    return {"passed": not violations, "violations": violations, "details": details}


def release_decision(comparison: dict, guardrails: dict, min_effect: float = 0.0) -> str:
    """rollback if a guardrail fails or the CI is entirely below 0; promote if the CI is
    entirely above 0 and delta >= min_effect; otherwise extend."""
    if not guardrails["passed"]:
        return "rollback"
    if comparison["ci_high"] < 0.0:
        return "rollback"
    if comparison["ci_low"] > 0.0 and comparison["delta"] >= min_effect:
        return "promote"
    return "extend"


# ----------------------------------------------------------------------- ramp


def ramp_schedule(steps: Sequence[float] = DEFAULT_RAMP) -> list[float]:
    """Validated ramp: strictly increasing fractions in (0, 1] ending at 1.0."""
    steps = [float(s) for s in steps]
    if not steps or steps[-1] != 1.0:
        raise ValueError("a ramp must end at 1.0 (full traffic)")
    if any(not 0.0 < s <= 1.0 for s in steps):
        raise ValueError("every step must be in (0, 1]")
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("steps must be strictly increasing")
    return steps


def advance_ramp(fraction: float, decision: str, steps: Sequence[float] = DEFAULT_RAMP) -> float:
    """promote -> the next step above ``fraction`` (1.0 stays); extend -> unchanged; rollback -> 0.0."""
    schedule = ramp_schedule(steps)
    if decision == "rollback":
        return 0.0
    if decision == "extend":
        return float(fraction)
    if decision == "promote":
        return next((s for s in schedule if s > fraction), 1.0)
    raise ValueError(f"unknown decision {decision!r}")
