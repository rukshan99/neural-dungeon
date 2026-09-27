"""SECRET - THE BATCH NORM CRUCIBLE

Train-mode output is normalized per feature; running statistics accumulate and
rule eval mode; and the full three-term backward is checked, for x, gamma and
beta, against finite differences.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

crucible = load_room(__file__, "secret_batch_norm_crucible")

pytestmark = pytest.mark.secret

rng = np.random.default_rng(99)


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
    return grad


def relative_error(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def test_train_mode_output_has_zero_mean_and_unit_std_per_feature():
    bn = crucible.BatchNorm1d(4)
    x = rng.standard_normal((64, 4)) * np.array([1.0, 10.0, 0.1, 100.0]) + np.array([0.0, 5.0, -3.0, 50.0])
    out = bn.forward(x)
    assert out.shape == x.shape, f"output shape must equal input shape {x.shape}, got {out.shape}"
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10, err_msg="each FEATURE (column) is centred over the batch")
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-3, err_msg=(
        "each feature has unit std (up to eps). Normalize over axis=0, per column, not per row."
    ))


def test_gamma_and_beta_set_the_output_scale_and_shift():
    bn = crucible.BatchNorm1d(3)
    bn.params["gamma"][...] = [2.0, 0.5, -1.0]
    bn.params["beta"][...] = [1.0, -2.0, 0.0]
    out = bn.forward(rng.standard_normal((100, 3)) * 7 + 3)
    np.testing.assert_allclose(out.mean(axis=0), [1.0, -2.0, 0.0], atol=1e-10, err_msg="mean of each column is beta")
    np.testing.assert_allclose(out.std(axis=0), [2.0, 0.5, 1.0], atol=1e-3, err_msg="std of each column is |gamma|")


def test_running_statistics_follow_the_momentum_rule():
    bn = crucible.BatchNorm1d(2, momentum=0.9)
    x = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
    bn.forward(x)
    mu, var = x.mean(axis=0), x.var(axis=0)
    np.testing.assert_allclose(bn.running_mean, 0.9 * 0.0 + 0.1 * mu, atol=1e-12, err_msg=(
        "running_mean <- momentum * running_mean + (1 - momentum) * batch mean, starting from zeros"
    ))
    np.testing.assert_allclose(bn.running_var, 0.9 * 1.0 + 0.1 * var, atol=1e-12, err_msg=(
        "running_var <- momentum * running_var + (1 - momentum) * batch var (biased, ddof=0), starting from ones"
    ))
    bn.forward(x)
    np.testing.assert_allclose(bn.running_mean, 0.9 * (0.1 * mu) + 0.1 * mu, atol=1e-12, err_msg="a second forward applies the rule again")


def test_eval_mode_uses_running_statistics_and_touches_nothing():
    bn = crucible.BatchNorm1d(3)
    for _ in range(50):  # let the running stats converge toward the data's stats
        bn.forward(rng.standard_normal((32, 3)) * 4.0 + 2.0)
    rm, rv = bn.running_mean.copy(), bn.running_var.copy()
    bn.training = False
    x = rng.standard_normal((5, 3)) * 4.0 + 2.0
    out = bn.forward(x)
    expected = bn.params["gamma"] * (x - rm) / np.sqrt(rv + bn.eps) + bn.params["beta"]
    np.testing.assert_allclose(out, expected, atol=1e-10, err_msg=(
        "eval mode normalizes with running_mean / running_var, not with this batch's statistics"
    ))
    np.testing.assert_array_equal(bn.running_mean, rm, err_msg="eval mode must not update running statistics")
    np.testing.assert_array_equal(bn.running_var, rv)
    # a batch of one is the whole point of running stats: it must not collapse to zero
    single = bn.forward(np.array([[10.0, -10.0, 0.0]]))
    assert not np.allclose(single, 0.0), "a single example in eval mode must not be normalized against itself"


def test_backward_gradient_checks_for_x_gamma_and_beta():
    bn = crucible.BatchNorm1d(5)
    bn.params["gamma"][...] = rng.standard_normal(5)
    bn.params["beta"][...] = rng.standard_normal(5)
    x = rng.standard_normal((8, 5)) * 3.0 + 1.0
    dout = rng.standard_normal((8, 5))

    def f():
        return np.sum(bn.forward(x) * dout)

    bn.zero_grad()
    bn.forward(x)
    dx = bn.backward(dout)
    assert dx.shape == x.shape, f"dx must have x's shape {x.shape}, got {dx.shape}"
    err_x = relative_error(dx, numerical_gradient(f, x))
    assert err_x < 1e-5, (
        f"dx disagrees with finite differences (relative error {err_x:.2e}). mu and var depend on x: the shortcut "
        "dx = dout * gamma / sqrt(var + eps) drops two of the three terms."
    )
    err_g = relative_error(bn.grads["gamma"], numerical_gradient(f, bn.params["gamma"]))
    assert err_g < 1e-5, f"dgamma disagrees with finite differences (relative error {err_g:.2e}). dgamma = sum(dout * xhat, axis=0)."
    err_b = relative_error(bn.grads["beta"], numerical_gradient(f, bn.params["beta"]))
    assert err_b < 1e-5, f"dbeta disagrees with finite differences (relative error {err_b:.2e}). dbeta = sum(dout, axis=0)."


def test_backward_dx_rows_average_to_zero_and_are_uncorrelated_with_xhat():
    """Two consequences of the exact formula: the batch mean of dx is 0 per feature, and sum(dx * xhat) is 0 per feature.

    Both hold exactly when eps = 0 (eps nudges the second one by eps / (var + eps)), so this layer is built with eps=0.
    """
    bn = crucible.BatchNorm1d(4, eps=0.0)
    x = rng.standard_normal((16, 4)) * 2.0
    dout = rng.standard_normal((16, 4))
    bn.forward(x)
    dx = bn.backward(dout)
    xhat = (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0))
    np.testing.assert_allclose(dx.sum(axis=0), 0.0, atol=1e-9, err_msg=(
        "the gradient through a normalization sums to zero over the batch: shifting every x equally changes nothing"
    ))
    np.testing.assert_allclose((dx * xhat).sum(axis=0), 0.0, atol=1e-9, err_msg=(
        "the gradient is orthogonal to xhat: scaling every x equally changes nothing either"
    ))


def test_gradients_accumulate_and_zero_grad_resets():
    bn = crucible.BatchNorm1d(3)
    x = rng.standard_normal((10, 3))
    dout = rng.standard_normal((10, 3))
    bn.zero_grad()
    bn.forward(x)
    bn.backward(dout)
    once = bn.grads["gamma"].copy()
    bn.forward(x)
    bn.backward(dout)
    np.testing.assert_allclose(bn.grads["gamma"], 2 * once, atol=1e-12, err_msg="backward must ADD into grads")
    bn.zero_grad()
    assert np.all(bn.grads["gamma"] == 0) and np.all(bn.grads["beta"] == 0)
