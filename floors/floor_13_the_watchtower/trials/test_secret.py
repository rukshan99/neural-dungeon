"""SECRET - THE SEQUENTIAL SENTINEL

A sequential test you may peek at every day. On A/A traffic it declares a
winner rarely (about alpha); on a real five-point lift it stops early and
right most of the time. The naive daily peeker, run alongside, fools itself
far more often.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

sentinel = load_room(__file__, "secret_sequential_sentinel")

pytestmark = pytest.mark.secret

P_A, EFFECT, HORIZON, SEEDS = 0.80, 0.05, 3000, 200
ALPHA, BETA = 0.05, 0.20


# ---------------------------------------------------------------- arithmetic
def test_discordant_probability_by_hand():
    assert math.isclose(sentinel.discordant_probability(0.8, 0.8), 0.5), "Equal rates: among pairs where exactly one succeeded, B won half the time."
    got = sentinel.discordant_probability(0.80, 0.85)
    assert math.isclose(got, 0.17 / 0.29), f"p_b(1-p_a) / (p_b(1-p_a) + p_a(1-p_b)) = 0.17 / 0.29 = {0.17 / 0.29:.4f}; you said {got:.4f}."
    with pytest.raises(ValueError):
        sentinel.discordant_probability(0.0, 0.5)


def test_wald_bounds_by_hand():
    lower, upper = sentinel.wald_bounds(ALPHA, BETA)
    assert math.isclose(lower, math.log(0.2 / 0.95)) and math.isclose(upper, math.log(0.8 / 0.05)), (
        f"lower = ln(beta / (1 - alpha)) = {math.log(0.2 / 0.95):.4f}, upper = ln((1 - beta) / alpha) = {math.log(0.8 / 0.05):.4f}; got ({lower:.4f}, {upper:.4f})."
    )


def test_fixed_horizon_pairs_matches_the_normal_approximation():
    z_a, z_b = 1.6448536269514722, 0.8416212335729143
    expected = math.ceil((z_a + z_b) ** 2 * (0.8 * 0.2 + 0.85 * 0.15) / 0.05**2)
    assert sentinel.fixed_horizon_pairs(P_A, EFFECT, ALPHA, BETA) == expected, f"One-sided at alpha 0.05, power 0.8: {expected} pairs."
    assert sentinel.fixed_horizon_pairs(P_A, EFFECT, ALPHA, BETA, two_sided=True) > expected, "Two-sided needs z_{alpha/2}, so more pairs."


# ------------------------------------------------------------------- updates
def test_only_discordant_pairs_move_the_log_likelihood_ratio():
    s = sentinel.SequentialSentinel(P_A, EFFECT, ALPHA, BETA)
    assert s.update(True, True) is None and s.update(False, False) is None
    assert s.llr == 0.0, "Both succeeded or both failed: no evidence about which arm is better."
    s.update(False, True)
    assert math.isclose(s.llr, math.log(s.q1 / 0.5)), "A pair where only B succeeded adds ln(q1 / 0.5)."
    s.update(True, False)
    assert math.isclose(s.llr, math.log(s.q1 / 0.5) + math.log((1 - s.q1) / 0.5)), "A pair where only A succeeded adds ln((1 - q1) / 0.5)."
    assert s.n_pairs == 4 and s.n_discordant == 2


def test_the_sentinel_stops_at_a_bound_and_then_ignores_the_stream():
    s = sentinel.SequentialSentinel(P_A, EFFECT, ALPHA, BETA)
    decision = None
    n = 0
    while decision is None:
        decision = s.update(False, True)  # B wins every discordant pair
        n += 1
    assert decision == "b_better" and s.llr >= s.upper, "Crossing the upper bound means B is better."
    assert n == math.ceil(s.upper / math.log(s.q1 / 0.5)), "It should take exactly ceil(upper / step) B-wins."
    assert s.update(True, False) == "b_better" and s.n_pairs == n, "Once decided, the sentinel stops counting."
    t = sentinel.SequentialSentinel(P_A, EFFECT, ALPHA, BETA)
    assert t.feed([True] * 40, [False] * 40) == "no_difference", "Forty pairs where only A succeeded: cross the lower bound."


# --------------------------------------------------------------- calibration
def _aa_stream(seed):
    rng = np.random.default_rng(seed)
    return rng.random(HORIZON) < P_A, rng.random(HORIZON) < P_A


def _naive_peeker_false_positive(seed, days=30, per_day=100):
    """A z-test at one-sided 5% every day for a month. Flags if ANY day looks significant."""
    rng = np.random.default_rng(seed)
    a = (rng.random(days * per_day) < P_A).astype(float)
    b = (rng.random(days * per_day) < P_A).astype(float)
    n = np.arange(per_day, days * per_day + 1, per_day)
    ma, mb = np.cumsum(a)[n - 1] / n, np.cumsum(b)[n - 1] / n
    se = np.sqrt(np.clip(ma * (1 - ma) + mb * (1 - mb), 1e-12, None) / n)
    return bool(np.any((mb - ma) / se > 1.6449))


def test_on_aa_traffic_the_sentinel_rarely_declares_a_winner_and_the_peeker_often_does():
    false_positives = 0
    for seed in range(SEEDS):
        a, b = _aa_stream(seed)
        false_positives += sentinel.SequentialSentinel(P_A, EFFECT, ALPHA, BETA).feed(a, b) == "b_better"
    rate = false_positives / SEEDS
    assert rate <= ALPHA + 0.05, (
        f"On {SEEDS} A/A streams the sentinel declared B better {false_positives} times ({rate:.1%}); alpha is {ALPHA:.0%}. Check the bounds and the step sizes."
    )
    naive = sum(_naive_peeker_false_positive(seed) for seed in range(SEEDS)) / SEEDS
    assert naive > rate and naive > 0.10, (
        f"Sanity check on the lesson: a daily z-test peeked at for 30 days false-alarms {naive:.1%} of the time (expected ~20%), the sentinel {rate:.1%}."
    )


def test_on_a_real_five_point_lift_the_sentinel_stops_early_and_right():
    correct, stops = 0, []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        a = rng.random(HORIZON) < P_A
        b = rng.random(HORIZON) < P_A + EFFECT
        s = sentinel.SequentialSentinel(P_A, EFFECT, ALPHA, BETA)
        correct += s.feed(a, b) == "b_better"
        stops.append(s.n_pairs)
    power = correct / SEEDS
    assert power >= 1.0 - BETA - 0.10, f"With beta = {BETA} the sentinel should find the lift at least ~{1 - BETA:.0%} of the time; it managed {power:.1%}."
    fixed = sentinel.fixed_horizon_pairs(P_A, EFFECT, ALPHA, BETA)
    mean_stop = float(np.mean(stops))
    assert mean_stop < fixed, (
        f"A fixed-horizon test needs {fixed} pairs; the sentinel stopped after {mean_stop:.0f} on average. Sequential testing should be cheaper when the effect is real."
    )
    assert max(stops) <= HORIZON
