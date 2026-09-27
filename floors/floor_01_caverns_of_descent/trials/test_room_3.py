"""TRIAL 1.3 - THE DESCENT

The model, its loss and gradients, the generic descent loop, a fit on noisy
data, and the canyon that standardization flattens into a bowl.
"""

import math

import numpy as np

from dungeon.trials import load_room

room = load_room(__file__, "room_3_the_descent")

rng = np.random.default_rng(13)

W_TRUE = np.array([3.0, -0.5])
B_TRUE = 1.0


def _make_data(scales, n=200, noise=0.05, seed=3):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, len(scales))) * np.asarray(scales, dtype=float)
    y = X @ W_TRUE + B_TRUE + noise * r.standard_normal(n)
    return X, y


def _augment(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _hessian_eigs(X):
    Xa = _augment(X)
    return np.linalg.eigvalsh(2.0 / X.shape[0] * Xa.T @ Xa)


def _safe_lr(X):
    """1 / lambda_max of the MSE Hessian: the classic safe step for this loss."""
    return 1.0 / _hessian_eigs(X).max()


def _best_loss(X, y):
    theta = np.linalg.lstsq(_augment(X), y, rcond=None)[0]
    return float(np.mean((_augment(X) @ theta - y) ** 2))


def _numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2 * eps)
    return grad


# ---------------------------------------------------------------- the model
def test_linear_forward_is_a_matmul_plus_a_bias():
    X = rng.standard_normal((6, 3))
    w = rng.standard_normal(3)
    out = room.linear_forward(X, w, 0.5)
    assert out.shape == (6,), f"(N, D) @ (D,) + b has shape (N,) = (6,), got {out.shape}."
    np.testing.assert_allclose(out, X @ w + 0.5)


def test_the_loss_is_the_mean_squared_residual_and_the_grads_have_the_right_shapes():
    X, y = _make_data((1.0, 1.0))
    w, b = np.array([1.0, 2.0]), -0.3
    loss, grad_w, grad_b = room.linear_loss_and_grads(X, y, w, b)
    assert type(loss) is float, f"loss must be a Python float, got {type(loss).__name__}."
    assert math.isclose(loss, float(np.mean((X @ w + b - y) ** 2)))
    assert np.shape(grad_w) == (2,), f"grad_w must have w's shape (2,), got {np.shape(grad_w)}."
    assert np.ndim(grad_b) == 0, f"grad_b is a scalar, got something of shape {np.shape(grad_b)}."


def test_the_gradients_match_finite_differences():
    X, y = _make_data((1.0, 1.0))
    w, b = np.array([0.7, -1.2]), 0.4
    _, grad_w, grad_b = room.linear_loss_and_grads(X, y, w, b)
    num_w = _numerical_grad(lambda v: room.linear_loss_and_grads(X, y, v, b)[0], w)
    num_b = _numerical_grad(lambda v: room.linear_loss_and_grads(X, y, w, float(v))[0], np.array(b))
    np.testing.assert_allclose(grad_w, num_w, rtol=1e-6, atol=1e-8, err_msg="grad_w should be (2/N) X^T r. Remember the 2 and the 1/N.")
    assert math.isclose(float(grad_b), float(num_b), rel_tol=1e-6, abs_tol=1e-8), (
        f"grad_b should be (2/N) sum(r) = {float(num_b):.6f}, got {float(grad_b):.6f}."
    )


def test_the_gradient_vanishes_at_the_least_squares_solution():
    X, y = _make_data((1.0, 1.0))
    theta = np.linalg.lstsq(_augment(X), y, rcond=None)[0]
    _, grad_w, grad_b = room.linear_loss_and_grads(X, y, theta[:-1], float(theta[-1]))
    assert np.linalg.norm(grad_w) < 1e-8 and abs(grad_b) < 1e-8, (
        "At the exact least-squares minimum the gradient is zero. Yours is not: the formula is off."
    )


# ---------------------------------------------------------- the descent loop
def test_gradient_descent_shrinks_a_bowl_geometrically():
    x0 = np.array([2.0, -1.0, 0.5])
    final, history = room.gradient_descent(lambda x: x, x0, lr=0.1, steps=10)  # f = 0.5 ||x||^2
    np.testing.assert_allclose(final, x0 * 0.9**10, rtol=1e-12, err_msg="x <- x - 0.1 x = 0.9 x each step, so x_10 = 0.9^10 x_0.")
    assert len(history) == 11, f"history holds the start plus one entry per step: 11, got {len(history)}."
    np.testing.assert_array_equal(history[0], x0, err_msg="history[0] is the starting point.")
    np.testing.assert_allclose(history[5], x0 * 0.9**5, rtol=1e-12, err_msg="history[k] is the value after k updates.")


def test_gradient_descent_keeps_its_hands_off_your_array_and_calls_you_back():
    x0 = np.array([1.0, 1.0])
    before = x0.copy()
    seen = []
    room.gradient_descent(lambda x: 2 * x, x0, lr=0.25, steps=4, callback=lambda k, p: seen.append((k, p.copy())))
    np.testing.assert_array_equal(x0, before, err_msg="gradient_descent modified the params array you passed in. Start from a copy.")
    assert [k for k, _ in seen] == [1, 2, 3, 4], f"callback(k, params) is called after each update with k = 1..steps, got k = {[k for k, _ in seen]}."
    np.testing.assert_allclose(seen[-1][1], before * 0.5**4)


def test_a_learning_rate_past_the_cliff_diverges():
    # f = 0.5 x^2: each step multiplies x by (1 - lr). lr = 2.5 gives -1.5: it grows.
    final, history = room.gradient_descent(lambda x: x, np.array([1.0]), lr=2.5, steps=10)
    assert abs(final[0]) > 50, f"With lr = 2.5 on f = 0.5 x^2 the iterate should explode to |x| = 1.5^10 = 57.7, got {final[0]}. A preview of Room 1.4."


# ---------------------------------------------------------- standardization
def test_standardize_features_returns_zero_mean_unit_std_and_the_statistics():
    X, _ = _make_data((1.0, 10.0))
    X += np.array([0.0, 5.0])
    X_std, mean, std = room.standardize_features(X)
    assert X_std.shape == X.shape
    np.testing.assert_allclose(X_std.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(X_std.std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(mean, X.mean(axis=0)), "mean must be the column means (shape (D,))."
    np.testing.assert_allclose(std, X.std(axis=0)), "std must be the column stds (shape (D,))."


# ------------------------------------------------------------------ the fit
def test_fit_linear_regression_recovers_the_true_line():
    X, y = _make_data((1.0, 1.0))
    w, b, losses = room.fit_linear_regression(X, y, lr=0.1, steps=500)
    assert np.shape(w) == (2,)
    np.testing.assert_allclose(w, W_TRUE, atol=0.05, err_msg=f"True weights are {W_TRUE}; the fit found {w}.")
    assert abs(b - B_TRUE) < 0.05, f"True bias is {B_TRUE}; the fit found {b}."
    assert len(losses) == 500, f"losses has one entry per step (500), got {len(losses)}."
    assert math.isclose(losses[0], float(np.mean(y**2))), (
        "losses[0] is the loss at the starting point w = 0, b = 0, i.e. mean(y^2) - BEFORE the first update."
    )
    assert losses[-1] < 2 * _best_loss(X, y) + 1e-3, f"The final loss {losses[-1]:.4f} is far from the noise floor {_best_loss(X, y):.4f}."


def test_the_loss_never_climbs_at_a_safe_learning_rate():
    X, y = _make_data((1.0, 10.0))
    _, _, losses = room.fit_linear_regression(X, y, lr=_safe_lr(X), steps=300)
    climbs = np.flatnonzero(np.diff(losses) > 1e-12)
    assert climbs.size == 0, (
        f"At lr = 1/lambda_max the loss must be non-increasing, but it rose at steps {climbs[:5].tolist()}. "
        "Is the update params - lr * grad (minus, not plus)?"
    )


def test_standardizing_turns_the_canyon_into_a_bowl():
    X, y = _make_data((1.0, 10.0))
    X_std, _, _ = room.standardize_features(X)
    target = _best_loss(X, y) + 1e-6  # the same optimum: standardizing is an affine reparametrization

    def steps_to_target(Xm):
        _, _, losses = room.fit_linear_regression(Xm, y, lr=_safe_lr(Xm), steps=2000)
        hits = np.flatnonzero(np.asarray(losses) <= target)
        return int(hits[0]) if hits.size else None

    raw_steps, std_steps = steps_to_target(X), steps_to_target(X_std)
    kappa_raw = _hessian_eigs(X).max() / _hessian_eigs(X).min()
    kappa_std = _hessian_eigs(X_std).max() / _hessian_eigs(X_std).min()
    assert std_steps is not None, "The standardized fit never reached the optimum. Something is off in fit or standardize."
    assert raw_steps is None or std_steps * 10 < raw_steps, (
        f"Raw features (condition number {kappa_raw:.0f}) reached the optimum in {raw_steps} steps; standardized "
        f"(condition number {kappa_std:.2f}) took {std_steps}. Standardizing should be at least 10x faster."
    )
