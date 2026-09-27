"""TRIAL 1.4 - THE PROPHECY OF CURVES

Nothing to implement. The chamber rolls the marble with each learning rate,
classifies what happened, and compares with what you said would happen.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_4_prophecy_of_curves")

VERDICTS = ("converges_smoothly", "converges_oscillating", "diverges")


def _roll(a, lr, x0=1.0, steps=30):
    """Gradient descent on f = 0.5 * a * x^2 (works for scalar a or a vector of curvatures)."""
    a = np.asarray(a, dtype=float)
    xs = [np.full(a.shape, x0) if a.ndim else float(x0)]
    for _ in range(steps):
        xs.append(xs[-1] - lr * a * xs[-1])
    return np.array(xs)


def _classify(xs):
    if not np.all(np.isfinite(xs[-1])) or abs(xs[-1]) > abs(xs[0]):
        return "diverges"
    if np.any(xs[:-1] * xs[1:] < 0):
        return "converges_oscillating"
    return "converges_smoothly"


# ------------------------------------------------------------ one curve
def test_the_curve_prophecy_is_complete():
    assert room.CURVE_A == 4.0 and tuple(room.CURVE_LEARNING_RATES) == (0.1, 0.25, 0.4, 0.6), "Do not change the curve or its learning rates."
    assert set(room.CURVE_PROPHECY) == set(room.CURVE_LEARNING_RATES), "CURVE_PROPHECY needs one entry per learning rate, no more."


@pytest.mark.parametrize("lr", (0.1, 0.25, 0.4, 0.6), ids=lambda lr: f"lr={lr}")
def test_the_marble_does_what_you_said(lr):
    prediction = room.CURVE_PROPHECY.get(lr)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied lr = {lr}. Fill in CURVE_PROPHECY.")
    assert prediction in VERDICTS, f"{prediction!r} is not one of {VERDICTS}."
    xs = _roll(room.CURVE_A, lr)
    verdict = _classify(xs)
    r = 1 - lr * room.CURVE_A
    assert prediction == verdict, (
        f"For lr = {lr} you said {prediction!r}; the marble {verdict.replace('_', ' ')}. Each step multiplies x by "
        f"r = 1 - lr * a = {r:+.2f}. First iterates: {np.round(xs[:5], 3).tolist()}."
    )


def test_one_learning_rate_lands_on_the_minimum_in_a_single_step():
    lr = room.ONE_STEP_LR
    if lr is None:
        raise NotImplementedError("You have not named ONE_STEP_LR.")
    assert lr in room.CURVE_LEARNING_RATES, f"ONE_STEP_LR must be one of {room.CURVE_LEARNING_RATES}, got {lr}."
    x1 = _roll(room.CURVE_A, lr, steps=1)[-1]
    assert abs(x1) < 1e-12, (
        f"After one step at lr = {lr} the marble sits at x = {x1:.3f}, not 0. It lands exactly when 1 - lr * a = 0, i.e. lr = 1/a."
    )


def test_the_largest_stable_learning_rate_is_the_edge_of_the_cliff():
    lr = room.LARGEST_STABLE_LR
    if lr is None:
        raise NotImplementedError("You have not named LARGEST_STABLE_LR.")
    just_below, just_above = _roll(room.CURVE_A, 0.99 * lr), _roll(room.CURVE_A, 1.01 * lr)
    assert _classify(just_below) != "diverges", f"lr = {0.99 * lr:.4f} (1% below your answer) already diverges, so your answer is too large."
    assert _classify(just_above) == "diverges", f"lr = {1.01 * lr:.4f} (1% above your answer) still converges, so your answer is not the largest."
    assert abs(lr - 2 / room.CURVE_A) < 1e-9, f"The edge is |1 - lr * a| = 1, i.e. lr = 2/a = {2 / room.CURVE_A}; you said {lr}."


# ---------------------------------------------------------- the 2-D bowl
def test_the_bowl_prophecy_is_complete():
    assert tuple(room.BOWL_A) == (1.0, 25.0), "Do not change the bowl."
    assert set(room.BOWL_PROPHECY) == {"coordinate_that_sets_the_stable_lr", "largest_stable_lr", "slowest_coordinate_at_lr_0.07"}, (
        "Do not rename or remove the bowl's questions; answer them."
    )


def test_the_steep_coordinate_caps_the_learning_rate():
    prediction = room.BOWL_PROPHECY.get("coordinate_that_sets_the_stable_lr")
    if prediction is None:
        raise NotImplementedError("You have not prophesied 'coordinate_that_sets_the_stable_lr'.")
    a = np.asarray(room.BOWL_A)
    lr = 1.1 * 2 / a.max()  # a hair over the edge
    xs = _roll(a, lr, steps=40)
    grew = np.abs(xs[-1]) > np.abs(xs[0])
    culprit = int(np.flatnonzero(grew)[0])
    assert prediction == culprit, (
        f"You said coordinate {prediction}. At lr = {lr:.3f}, coordinate {culprit} (curvature {a[culprit]}) blew up to "
        f"{xs[-1][culprit]:.2e} while the other shrank to {xs[-1][1 - culprit]:.2e}. The condition lr < 2/a_i must hold for "
        "EVERY coordinate, so the largest curvature sets the cap."
    )


def test_the_bowl_has_a_cliff_edge_too():
    lr = room.BOWL_PROPHECY.get("largest_stable_lr")
    if lr is None:
        raise NotImplementedError("You have not prophesied 'largest_stable_lr' for the bowl.")
    a = np.asarray(room.BOWL_A)
    below, above = _roll(a, 0.99 * lr, steps=60), _roll(a, 1.01 * lr, steps=60)
    assert np.all(np.abs(below[-1]) < 1), f"lr = {0.99 * lr:.4f} (1% below your answer) already diverges in some coordinate: {below[-1]}."
    assert np.any(np.abs(above[-1]) > 1), f"lr = {1.01 * lr:.4f} (1% above your answer) still converges in both coordinates: {above[-1]}."
    assert abs(lr - 2 / a.max()) < 1e-9, f"The edge is 2 / max(a) = {2 / a.max()}; you said {lr}."


def test_the_flat_coordinate_is_the_slow_one():
    prediction = room.BOWL_PROPHECY.get("slowest_coordinate_at_lr_0.07")
    if prediction is None:
        raise NotImplementedError("You have not prophesied 'slowest_coordinate_at_lr_0.07'.")
    a = np.asarray(room.BOWL_A)
    xs = _roll(a, 0.07, steps=50)
    slowest = int(np.argmax(np.abs(xs[-1])))
    r = 1 - 0.07 * a
    assert prediction == slowest, (
        f"You said coordinate {prediction}. After 50 steps at lr = 0.07: |x| = {np.abs(xs[-1]).round(8).tolist()}. "
        f"Per-step factors are r = {r.round(3).tolist()}: the steep coordinate oscillates but shrinks fast; "
        "the flat one crawls. Once the steep wall caps lr, the flat floor sets the pace. That is ill-conditioning."
    )
