"""TRIAL 1.1 - THE ALTAR OF LOSS

Known values, symmetry, huge logits that must stay finite, and every gradient
held against a central-difference oracle. (The trial carries its own oracle;
you build a better one in Room 1.2.)
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_1_altar_of_loss")

rng = np.random.default_rng(11)


# ------------------------------------------------------------------ helpers
def _numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2 * eps)
    return grad


def _rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def _assert_gradient_matches(name, loss_fn, grad_fn, x, tol=1e-5):
    g_ana = np.asarray(grad_fn(x))
    assert g_ana.shape == x.shape, (
        f"{name}: the gradient must have the prediction's shape {x.shape}, got {g_ana.shape}."
    )
    g_num = _numerical_grad(loss_fn, x)
    rel = _rel_error(g_num, g_ana)
    assert rel < tol, (
        f"{name}: the analytic gradient disagrees with finite differences (relative error {rel:.2e}). "
        "Did you forget the 1/N from the mean, or a factor of 2?"
    )


def _finite_or_fail(fn, what):
    """Call fn with overflow/invalid promoted to errors, so a naive exp(1000) is caught by name."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return fn()
    except FloatingPointError as exc:
        pytest.fail(f"{what} overflowed or produced inf/NaN ({exc}). Never exponentiate a large positive number.")


# --------------------------------------------------------------- regression
def test_a_perfect_prediction_has_zero_loss_and_returns_a_float():
    y = rng.standard_normal((3, 4))
    for name, fn in [("mse", room.mse), ("mae", room.mae)]:
        loss = fn(y.copy(), y)
        assert loss == 0.0, f"{name} of a perfect prediction should be 0.0, got {loss!r}."
        assert type(loss) is float, f"{name} must return a plain Python float, not {type(loss).__name__}. Wrap it: float(...)."


def test_mse_and_mae_know_their_values_and_do_not_care_who_is_who():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    assert math.isclose(room.mse(a, b), 4.0 / 3.0), f"mse([1,2,3],[1,2,5]) is (0+0+4)/3 = 1.333..., got {room.mse(a, b)}."
    assert math.isclose(room.mae(a, b), 2.0 / 3.0), f"mae([1,2,3],[1,2,5]) is (0+0+2)/3 = 0.666..., got {room.mae(a, b)}."
    assert room.mse(a, b) == room.mse(b, a) and room.mae(a, b) == room.mae(b, a), "Both losses are symmetric in their arguments."


def test_mse_is_a_mean_not_a_sum():
    y_pred = np.ones((10, 5))
    y_true = np.zeros((10, 5))
    assert math.isclose(room.mse(y_pred, y_true), 1.0), (
        f"50 errors of 1.0 each average to 1.0, got {room.mse(y_pred, y_true)}. mean, not sum."
    )


def test_mse_gradient_matches_the_oracle():
    y_true = rng.standard_normal((4, 3))
    y_pred = rng.standard_normal((4, 3))
    _assert_gradient_matches("mse_grad", lambda p: room.mse(p, y_true), lambda p: room.mse_grad(p, y_true), y_pred)
    np.testing.assert_allclose(room.mse_grad(y_pred, y_true), 2 * (y_pred - y_true) / 12, err_msg="mse_grad is 2 (p - t) / N.")


def test_mae_gradient_matches_the_oracle_away_from_the_kinks():
    y_pred = rng.standard_normal((5, 2))
    # keep every |p - t| well above the finite-difference step so no element sits on a kink
    y_true = y_pred - np.sign(rng.standard_normal((5, 2))) * (0.1 + rng.random((5, 2)))
    _assert_gradient_matches("mae_grad", lambda p: room.mae(p, y_true), lambda p: room.mae_grad(p, y_true), y_pred)


# ------------------------------------------------- binary classification
def test_binary_cross_entropy_of_a_coin_flip_is_log_two():
    p = np.full(6, 0.5)
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert math.isclose(room.binary_cross_entropy(p, y), math.log(2.0), rel_tol=1e-9), (
        f"Predicting 0.5 for everything costs log(2) = 0.6931 per element, got {room.binary_cross_entropy(p, y)}."
    )
    assert type(room.binary_cross_entropy(p, y)) is float


def test_binary_cross_entropy_survives_certainty():
    right = _finite_or_fail(lambda: room.binary_cross_entropy(np.array([0.0, 1.0]), np.array([0.0, 1.0])), "binary_cross_entropy at p in {0, 1}")
    assert right < 1e-6, f"Confidently right should cost ~0, got {right}."
    wrong = _finite_or_fail(lambda: room.binary_cross_entropy(np.array([0.0, 1.0]), np.array([1.0, 0.0])), "binary_cross_entropy at p in {0, 1}")
    assert math.isfinite(wrong), "log(0) is -inf. Clip p into [eps, 1 - eps] before taking logs."
    assert math.isclose(wrong, -math.log(1e-7), rel_tol=1e-3), (
        f"Confidently wrong with eps = 1e-7 costs -log(1e-7) = 16.12 per element, got {wrong}."
    )


def test_binary_cross_entropy_gradient_matches_the_oracle():
    p = rng.uniform(0.05, 0.95, size=(3, 4))
    y = (rng.random((3, 4)) < 0.5).astype(float)
    _assert_gradient_matches(
        "binary_cross_entropy_grad",
        lambda q: room.binary_cross_entropy(q, y),
        lambda q: room.binary_cross_entropy_grad(q, y),
        p,
    )


def test_sigmoid_is_stable_at_both_ends():
    z = np.array([-1000.0, -20.0, 0.0, 20.0, 1000.0])
    s = _finite_or_fail(lambda: room.sigmoid(z), "sigmoid")
    assert s.shape == z.shape
    assert s[2] == 0.5 and s[0] == 0.0 and s[4] == 1.0, f"sigmoid(-1000, 0, 1000) must be exactly (0, 0.5, 1), got {s[[0, 2, 4]]}."
    np.testing.assert_allclose(s[1], 1 / (1 + math.exp(20)), rtol=1e-9)
    np.testing.assert_allclose(s[3], 1 / (1 + math.exp(-20)), rtol=1e-9)


def test_bce_with_logits_agrees_with_bce_of_sigmoid_when_nothing_overflows():
    z = rng.standard_normal(8) * 3
    y = (rng.random(8) < 0.5).astype(float)
    via_probs = room.binary_cross_entropy(1 / (1 + np.exp(-z)), y, eps=1e-15)
    direct = room.bce_with_logits(z, y)
    assert math.isclose(direct, via_probs, rel_tol=1e-8), (
        f"For moderate logits the two routes agree: sigmoid-then-bce = {via_probs}, bce_with_logits = {direct}."
    )
    assert type(direct) is float


def test_bce_with_logits_survives_a_thousand():
    z = np.array([1000.0, -1000.0, 1000.0, -1000.0])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    loss = _finite_or_fail(lambda: room.bce_with_logits(z, y), "bce_with_logits at |z| = 1000")
    assert math.isfinite(loss), "bce_with_logits(1000, ...) must be finite. softplus(z) = max(z, 0) + log1p(exp(-|z|))."
    # confidently wrong twice (1000 each), confidently right twice (0 each): mean 500
    assert math.isclose(loss, 500.0, rel_tol=1e-12), f"Expected exactly (1000 + 1000 + 0 + 0) / 4 = 500, got {loss}."


def test_bce_with_logits_gradient_is_sigmoid_minus_target_over_n():
    z = rng.standard_normal((2, 5)) * 2
    y = (rng.random((2, 5)) < 0.5).astype(float)
    _assert_gradient_matches("bce_with_logits_grad", lambda q: room.bce_with_logits(q, y), lambda q: room.bce_with_logits_grad(q, y), z)
    np.testing.assert_allclose(room.bce_with_logits_grad(z, y), (1 / (1 + np.exp(-z)) - y) / 10, rtol=1e-9)


# --------------------------------------------- multi-class classification
def test_log_softmax_rows_are_log_probabilities_and_ignore_shifts():
    logits = rng.standard_normal((4, 6))
    logp = room.log_softmax(logits)
    assert logp.shape == logits.shape
    np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-12, err_msg="exp(log_softmax) rows must sum to 1.")
    shifted = room.log_softmax(logits + rng.standard_normal((4, 1)) * 100)
    np.testing.assert_allclose(shifted, logp, atol=1e-9, err_msg="Adding a constant to a row must not change log_softmax.")
    huge = _finite_or_fail(lambda: room.log_softmax(np.array([[1000.0, 0.0, -1000.0]])), "log_softmax at 1000")
    np.testing.assert_allclose(huge, [[0.0, -1000.0, -2000.0]], atol=1e-9)


def test_softmax_cross_entropy_of_uniform_logits_is_log_c():
    logits = np.zeros((7, 5))
    labels = np.array([0, 1, 2, 3, 4, 0, 1])
    loss = room.softmax_cross_entropy(logits, labels)
    assert type(loss) is float
    assert math.isclose(loss, math.log(5.0)), (
        f"Uniform logits over 5 classes cost log(5) = 1.609 per row and the loss is the MEAN over the 7 rows, got {loss}."
    )


def test_softmax_cross_entropy_is_shift_invariant_and_survives_huge_logits():
    logits = rng.standard_normal((3, 4))
    labels = np.array([2, 0, 3])
    base = room.softmax_cross_entropy(logits, labels)
    shifted = room.softmax_cross_entropy(logits + 1000.0, labels)
    assert math.isclose(base, shifted, rel_tol=1e-9), f"Shifting every logit by 1000 changed the loss from {base} to {shifted}."
    wrong = _finite_or_fail(lambda: room.softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([1])), "softmax_cross_entropy at 1000")
    assert math.isclose(wrong, 1000.0, rel_tol=1e-12), f"logits (1000, 0) with label 1 cost exactly 1000, got {wrong}."
    right = room.softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([0]))
    assert right < 1e-12, f"logits (1000, 0) with label 0 cost ~0, got {right}."


def test_softmax_cross_entropy_gradient_matches_the_oracle_and_rows_sum_to_zero():
    logits = rng.standard_normal((4, 5))
    labels = rng.integers(0, 5, size=4)
    _assert_gradient_matches(
        "softmax_cross_entropy_grad",
        lambda z: room.softmax_cross_entropy(z, labels),
        lambda z: room.softmax_cross_entropy_grad(z, labels),
        logits,
    )
    grad = room.softmax_cross_entropy_grad(logits, labels)
    np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-12, err_msg="softmax - onehot sums to zero along each row.")
    assert np.all(grad[np.arange(4), labels] < 0), "The correct class's logit should be pushed UP (negative gradient)."
