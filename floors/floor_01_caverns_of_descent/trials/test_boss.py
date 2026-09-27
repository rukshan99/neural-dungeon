"""BOSS FIGHT - THE LEARNING-RATE LICH

Phase 1: the warmup + cosine schedule, checked to the decimal.
Phase 2: the learning-rate range test against the 2 / a_max bound.
Phase 3: the kappa = 1000 valley in 2000 gradient evaluations, then the
         Rosenbrock valley in 5000, with one implementation.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

boss = load_room(__file__, "boss_learning_rate_lich")

pytestmark = pytest.mark.boss


# --------------------------------------------------------------- problems
def _quadratic(a):
    a = np.asarray(a, dtype=float)
    return (lambda x: float(0.5 * np.sum(a * x * x))), (lambda x: a * x)


def _rosenbrock():
    def f(p):
        x, y = p
        return float((1 - x) ** 2 + 100 * (y - x * x) ** 2)

    def g(p):
        x, y = p
        return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])

    return f, g


class _Counter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


# --------------------------------------------------------------- phase 1
def test_phase_1_warmup_climbs_linearly_to_lr_max():
    lrs = [boss.cosine_with_warmup(s, total_steps=100, warmup_steps=10, lr_max=0.1) for s in range(11)]
    assert type(lrs[0]) is float, "Return a Python float."
    assert lrs[0] == 0.0, f"Warmup starts at 0 (lr_max * 0 / warmup_steps), got {lrs[0]}."
    assert math.isclose(lrs[5], 0.05), f"Halfway through a 10-step warmup lr is lr_max / 2 = 0.05, got {lrs[5]}."
    assert math.isclose(lrs[10], 0.1), f"At step == warmup_steps the lr is exactly lr_max, got {lrs[10]}."
    assert all(b > a for a, b in zip(lrs, lrs[1:])), "Warmup must be strictly increasing."


def test_phase_1_cosine_decays_through_the_midpoint_to_lr_min():
    kw = dict(total_steps=110, warmup_steps=10, lr_max=1.0, lr_min=0.1)
    assert math.isclose(boss.cosine_with_warmup(60, **kw), 0.55), (
        f"Halfway through the decay (step 60 of 10..110) cos(pi/2) = 0 gives (lr_max + lr_min)/2 = 0.55, got {boss.cosine_with_warmup(60, **kw)}."
    )
    assert math.isclose(boss.cosine_with_warmup(110, **kw), 0.1), f"At total_steps the lr is lr_min = 0.1, got {boss.cosine_with_warmup(110, **kw)}."
    assert math.isclose(boss.cosine_with_warmup(500, **kw), 0.1), "Past total_steps the lr stays clamped at lr_min."
    tail = [boss.cosine_with_warmup(s, **kw) for s in range(10, 111)]
    assert all(b <= a + 1e-15 for a, b in zip(tail, tail[1:])), "After warmup the schedule is non-increasing."
    assert min(tail) >= 0.1 - 1e-15 and max(tail) <= 1.0 + 1e-15, "The lr must stay inside [lr_min, lr_max]."


def test_phase_1_no_warmup_means_full_speed_at_step_zero():
    assert math.isclose(boss.cosine_with_warmup(0, total_steps=50, warmup_steps=0, lr_max=0.3), 0.3), (
        "With warmup_steps = 0 there is no warmup: lr(0) == lr_max. (Beware dividing by warmup_steps.)"
    )


@pytest.mark.parametrize("step,progress", [(100, 0.0), (325, 0.25), (550, 0.5), (775, 0.75), (1000, 1.0)])
def test_phase_1_the_curve_is_a_cosine_to_the_decimal(step, progress):
    expected = 1e-5 + 0.5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * progress))
    got = boss.cosine_with_warmup(step, total_steps=1000, warmup_steps=100, lr_max=3e-4, lr_min=1e-5)
    assert math.isclose(got, expected, rel_tol=1e-12), (
        f"step {step}: progress through the decay is {progress}, so lr = lr_min + 0.5 (lr_max - lr_min)(1 + cos(pi * {progress})) = {expected:.6e}; got {got:.6e}."
    )


# --------------------------------------------------------------- phase 2
def test_phase_2_the_range_test_finds_the_cliff():
    loss, grad = _quadratic([1.0, 250.0])
    edge = 2.0 / 250.0
    factors = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.05, 1.2, 1.5, 2.0])
    candidates = list(np.random.default_rng(2).permutation(edge * factors))
    got = boss.lr_range_test(grad, loss, np.array([1.0, 1.0]), candidates, steps=50)
    assert type(got) is float, "Return a Python float."
    assert math.isclose(got, 0.95 * edge, rel_tol=1e-9), (
        f"GD on this valley diverges for lr > 2 / a_max = {edge:.4f}; the largest surviving candidate is {0.95 * edge:.5f}, you returned {got:.5f}. "
        "(Candidates arrive shuffled: return the largest survivor, not the last.)"
    )


def test_phase_2_on_a_tilted_valley_the_bound_is_still_two_over_lambda_max():
    r = np.random.default_rng(7)
    Q, _ = np.linalg.qr(r.standard_normal((3, 3)))
    A = Q @ np.diag([2.0, 30.0, 400.0]) @ Q.T  # curvatures no longer axis-aligned
    loss = lambda x: float(0.5 * x @ A @ x)  # noqa: E731
    grad = lambda x: A @ x  # noqa: E731
    edge = 2.0 / 400.0
    factors = np.array([0.2, 0.6, 0.8, 0.9, 0.97, 1.03, 1.1, 1.5])
    got = boss.lr_range_test(grad, loss, r.standard_normal(3), list(edge * factors), steps=60)
    assert math.isclose(got, 0.97 * edge, rel_tol=1e-9), (
        f"The largest eigenvalue of the Hessian is 400, so the edge is {edge:.5f} and the answer {0.97 * edge:.6f}; you returned {got:.6f}."
    )


def test_phase_2_when_every_step_is_too_bold_the_test_says_so():
    loss, grad = _quadratic([1.0, 250.0])
    with pytest.raises(ValueError):
        boss.lr_range_test(grad, loss, np.array([1.0, 1.0]), [0.01, 0.05, 0.1], steps=30)


# --------------------------------------------------------------- phase 3
def test_phase_3_the_lich_falls_in_its_own_valley():
    loss, grad = _quadratic([1.0, 1000.0])
    counted = _Counter(grad)
    x0 = np.array([1.0, 1.0])
    x = boss.slay_the_lich(counted, loss, x0, max_steps=2000)
    assert counted.calls <= 2000, f"You used {counted.calls} gradient evaluations; the budget is 2000."
    np.testing.assert_array_equal(x0, [1.0, 1.0], err_msg="slay_the_lich modified x0.")
    assert np.shape(x) == (2,), f"Return the final parameters with x0's shape (2,), got {np.shape(x)}."
    dist = float(np.linalg.norm(x))
    assert dist < 1e-3, (
        f"After {counted.calls} gradient evaluations you are at distance {dist:.2e} from the bottom; the Lich demands < 1e-3. "
        "Plain GD must keep lr < 2/1000 and then needs ~3500 steps. Per-coordinate steps (Adam) or momentum, with a "
        "schedule that decays to ~0 so the iterate can settle, do it in a few hundred."
    )


def test_phase_3_the_lich_flees_into_the_banana_valley():
    loss, grad = _rosenbrock()
    counted = _Counter(grad)
    x0 = np.array([-1.5, 2.0])
    x = boss.slay_the_lich(counted, loss, x0, max_steps=5000)
    assert counted.calls <= 5000, f"You used {counted.calls} gradient evaluations; the budget is 5000."
    final = loss(x)
    assert math.isfinite(final), f"The loss is {final}: your optimizer diverged. The valley walls here have curvature ~2000 at the start."
    assert final < 1e-2, (
        f"Rosenbrock from (-1.5, 2.0): loss {final:.3e} at x = {np.round(x, 3).tolist()} after {counted.calls} evaluations; "
        "the Lich demands < 1e-2 (the minimum is 0 at (1, 1)). Adam with lr_max ~ 0.1 and a cosine decay gets there in ~700 steps; "
        "a constant tiny lr does not."
    )
