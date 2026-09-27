"""TRIAL 13.3 - THE CANARY CAGE

Deterministic arm assignment that respects the fraction, an unpaired bootstrap
that tells two arms apart only when it should, guardrails that fail closed, a
decision rule with every branch exercised, and a ramp that only moves forward
on evidence.
"""

import hashlib
import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_3_the_canary_cage")


# ----------------------------------------------------------------- assignment
def test_the_same_request_always_lands_in_the_same_arm():
    arms = {room.assign_arm("req-42", 0.3, salt="s1") for _ in range(20)}
    assert len(arms) == 1 and arms <= {"canary", "baseline"}, f"assign_arm must be deterministic and return 'canary' or 'baseline'; got {arms}."


@pytest.mark.parametrize("fraction", [0.1, 0.5])
def test_the_fraction_is_respected_over_many_requests(fraction):
    ids = [f"req-{i}" for i in range(20_000)]
    share = sum(room.assign_arm(i, fraction, salt="tower") == "canary" for i in ids) / len(ids)
    assert abs(share - fraction) < 0.01, (
        f"Asked for {fraction:.0%} canary, got {share:.3%} over 20,000 ids. Map the hash to a uniform number in [0, 1) and compare it with the fraction."
    )


@pytest.mark.parametrize("request_id", ["req-7", "req-13", "tower-1"])
def test_the_assignment_follows_the_sha256_recipe_so_every_server_agrees(request_id):
    u = int.from_bytes(hashlib.sha256(f"tower:{request_id}".encode()).digest()[:8], "big") / 2**64
    just_above, just_below = room.assign_arm(request_id, u + 1e-6, salt="tower"), room.assign_arm(request_id, u - 1e-6, salt="tower")
    assert (just_above, just_below) == ("canary", "baseline"), (
        f"sha256('tower:{request_id}') -> first 8 bytes -> / 2**64 gives u = {u:.6f}: a fraction just above u must take this request "
        f"and one just below must not; got {just_above!r} / {just_below!r}. Python's hash() is salted per process, so two servers would "
        "disagree on the arm; hash f'{salt}:{request_id}' with hashlib.sha256."
    )


def test_the_edges_of_the_fraction_and_bad_input():
    assert all(room.assign_arm(f"r{i}", 0.0) == "baseline" for i in range(200)), "Fraction 0 sends nobody to the canary."
    assert all(room.assign_arm(f"r{i}", 1.0) == "canary" for i in range(200)), "Fraction 1 sends everybody."
    with pytest.raises(ValueError):
        room.assign_arm("r", 1.5)


def test_a_new_salt_reshuffles_but_keeps_the_share():
    ids = [f"req-{i}" for i in range(5000)]
    first = [room.assign_arm(i, 0.2, salt="a") for i in ids]
    second = [room.assign_arm(i, 0.2, salt="b") for i in ids]
    assert first != second, "Changing the salt must change who is in the canary; otherwise every experiment shares the same unlucky users."
    assert abs(sum(a == "canary" for a in second) / 5000 - 0.2) < 0.02


def test_raising_the_fraction_never_kicks_anyone_out_of_the_canary():
    ids = [f"req-{i}" for i in range(5000)]
    small = {i for i in ids if room.assign_arm(i, 0.05, salt="ramp") == "canary"}
    large = {i for i in ids if room.assign_arm(i, 0.25, salt="ramp") == "canary"}
    assert small <= large, (
        "Everyone in the 5% canary must still be in the 25% canary: compare ONE hashed uniform against the fraction, "
        "so ramping only adds users and a request never flips between arms mid-experiment."
    )


# ----------------------------------------------------------------- comparison
def test_compare_arms_reports_the_gap_with_an_interval_and_a_p_value():
    rng = np.random.default_rng(0)
    baseline = (rng.random(1000) < 0.80).astype(float)
    canary = (rng.random(400) < 0.90).astype(float)  # a different number of requests: unpaired
    result = room.compare_arms(baseline, canary, rng=np.random.default_rng(1))
    for key in ("delta", "ci_low", "ci_high", "p_value"):
        assert key in result, f"compare_arms is missing {key!r}."
    assert math.isclose(result["delta"], canary.mean() - baseline.mean()), "delta is mean(canary) - mean(baseline)."
    assert result["ci_low"] <= result["delta"] <= result["ci_high"], "The interval must contain the point estimate."
    assert result["ci_low"] > 0.0, f"A ten-point gap at these sizes is unmistakable; the interval ({result['ci_low']:.3f}, {result['ci_high']:.3f}) should sit above 0."
    assert 0.0 <= result["p_value"] < 0.05


def test_compare_arms_is_unpaired_and_honest_about_noise():
    rng = np.random.default_rng(2)
    baseline = (rng.random(900) < 0.85).astype(float)
    canary = (rng.random(100) < 0.85).astype(float)  # same true rate, tiny canary
    result = room.compare_arms(baseline, canary, rng=np.random.default_rng(3))
    assert result["ci_low"] < 0.0 < result["ci_high"], (
        f"Two arms with the same true rate: the interval ({result['ci_low']:.3f}, {result['ci_high']:.3f}) must straddle zero. "
        "A 100-request canary cannot see a difference that is not there."
    )
    assert result["p_value"] > 0.05
    width = result["ci_high"] - result["ci_low"]
    assert width > 0.1, f"With only 100 canary requests the interval should be wide (~0.14), not {width:.3f}. Resample EACH arm on its own."


def test_compare_arms_resamples_both_arms():
    rng = np.random.default_rng(20)
    baseline = (rng.random(300) < 0.5).astype(float)
    canary = (rng.random(300) < 0.5).astype(float)
    result = room.compare_arms(baseline, canary, rng=np.random.default_rng(21))
    width = result["ci_high"] - result["ci_low"]
    assert 0.14 < width < 0.19, (
        f"Two arms of 300 coin flips: the standard error of the difference is sqrt(0.25/300 + 0.25/300) = 0.041, so a 95% interval is about "
        f"0.16 wide; yours is {width:.3f}. Both arms are noisy. Holding the baseline fixed and resampling only the canary gives about 0.11: "
        "too narrow by a factor of sqrt(2), and a canary that looks more certain than it is."
    )


def test_compare_arms_is_exact_on_constant_arms_and_refuses_empty_ones():
    result = room.compare_arms([0.8] * 10, [0.9] * 10)
    assert math.isclose(result["delta"], 0.1) and math.isclose(result["ci_low"], 0.1) and math.isclose(result["ci_high"], 0.1), (
        f"Constant arms: every resample gives the same 0.1 gap, so the interval collapses to (0.1, 0.1); got {result}."
    )
    assert result["p_value"] == 0.0
    tie = room.compare_arms([0.5] * 5, [0.5] * 5)
    assert tie["p_value"] == 1.0, (
        f"Identical constant arms: every bootstrap delta is 0, so both tails are 1 and the two-sided p-value is capped at 1.0 (min(1, 2 * min(...))), not {tie['p_value']}."
    )
    with pytest.raises(ValueError):
        room.compare_arms([], [1.0, 0.0])
    same = room.compare_arms([1, 0, 1, 1], [1, 1, 0, 1], rng=np.random.default_rng(9))
    again = room.compare_arms([1, 0, 1, 1], [1, 1, 0, 1], rng=np.random.default_rng(9))
    assert same == again, "Same rng seed, same interval: draw randomness only from the rng you were given."


# ----------------------------------------------------------------- guardrails
SLO = {"p95_latency_ms": 1000.0, "error_rate": 0.01, "cost_per_request": 0.003}


def test_guardrails_pass_when_every_metric_is_under_its_limit():
    result = room.guardrail_check({"p95_latency_ms": 820.0, "error_rate": 0.004, "cost_per_request": 0.0021}, SLO)
    assert result["passed"] is True and result["violations"] == [], f"All three metrics are under their limits; got {result}."


def test_guardrails_name_what_broke():
    result = room.guardrail_check({"p95_latency_ms": 1300.0, "error_rate": 0.004, "cost_per_request": 0.0041}, SLO)
    assert result["passed"] is False
    assert set(result["violations"]) == {"p95_latency_ms", "cost_per_request"}, f"Name every violated guardrail; got {result['violations']}."


def test_an_unmeasured_guardrail_is_a_failed_guardrail():
    result = room.guardrail_check({"p95_latency_ms": 820.0, "error_rate": 0.004}, SLO)
    assert result["passed"] is False and "cost_per_request" in result["violations"], (
        "A metric in the SLO that the canary did not report cannot be verified. Fail closed: no gauge, no promotion."
    )


# -------------------------------------------------------------------- decision
PASS = {"passed": True, "violations": []}
FAIL = {"passed": False, "violations": ["p95_latency_ms"]}


def _cmp(delta, low, high):
    return {"delta": delta, "ci_low": low, "ci_high": high, "p_value": 0.01}


def test_promote_when_the_interval_is_above_zero_and_guardrails_hold():
    assert room.release_decision(_cmp(0.03, 0.01, 0.05), PASS) == "promote"


def test_rollback_when_the_interval_is_below_zero():
    assert room.release_decision(_cmp(-0.03, -0.05, -0.01), PASS) == "rollback"


def test_rollback_when_a_guardrail_fails_even_if_quality_looks_better():
    assert room.release_decision(_cmp(0.03, 0.01, 0.05), FAIL) == "rollback", (
        "A canary that answers better but breaks the latency SLO does not ship. Guardrails are checked first."
    )


def test_extend_when_the_interval_straddles_zero():
    assert room.release_decision(_cmp(0.01, -0.02, 0.04), PASS) == "extend", "Not enough evidence either way: keep collecting."


def test_extend_when_significant_but_smaller_than_the_minimum_effect():
    assert room.release_decision(_cmp(0.004, 0.001, 0.007), PASS, min_effect=0.01) == "extend", (
        "Statistically significant is not the same as worth shipping: below min_effect, do not promote."
    )
    assert room.release_decision(_cmp(0.02, 0.011, 0.03), PASS, min_effect=0.01) == "promote"


# ------------------------------------------------------------------------ ramp
def test_the_default_ramp_climbs_to_full_traffic():
    assert room.ramp_schedule() == [0.01, 0.05, 0.25, 0.5, 1.0]
    assert room.ramp_schedule((0.1, 0.5, 1.0)) == [0.1, 0.5, 1.0]


@pytest.mark.parametrize("bad", [(0.5, 0.25, 1.0), (0.1, 0.5), (0.0, 0.5, 1.0), ()])
def test_a_ramp_must_increase_and_end_at_one(bad):
    with pytest.raises(ValueError):
        room.ramp_schedule(bad)


def test_advance_ramp_moves_only_on_promote():
    assert room.advance_ramp(0.01, "promote") == 0.05
    assert room.advance_ramp(0.5, "promote") == 1.0
    assert room.advance_ramp(1.0, "promote") == 1.0, "At full traffic there is nowhere further to go."
    assert room.advance_ramp(0.25, "extend") == 0.25, "extend keeps the fraction."
    assert room.advance_ramp(0.25, "rollback") == 0.0, "rollback sends the canary fraction to 0."
    assert room.advance_ramp(0.03, "promote") == 0.05, "A fraction between steps advances to the next step above it."
    with pytest.raises(ValueError):
        room.advance_ramp(0.25, "shrug")
