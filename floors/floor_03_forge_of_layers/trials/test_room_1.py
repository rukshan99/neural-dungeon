"""TRIAL 3.1 - THE ANVIL

Every layer is struck twice: once forward, for shape and value, and once
backward, against central finite differences. The classic bugs (dW without the
transpose, db without the batch sum) have their own tests with their own names.
"""

import numpy as np

from dungeon.trials import load_room

room = load_room(__file__, "room_1_the_anvil")

rng = np.random.default_rng(31)


# ------------------------------------------------------------ the checker
def numerical_gradient(f, x, eps=1e-6):
    """d f() / d x by central differences, perturbing x in place one element at a time."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_layer_input_gradient(layer, x, dout, what):
    """Checks layer.backward(dout) against finite differences of sum(forward(x) * dout)."""
    layer.forward(x)
    dx = layer.backward(dout)
    assert dx.shape == x.shape, f"{what}: dx must have x's shape {x.shape}, got {dx.shape}"
    numeric = numerical_gradient(lambda: np.sum(layer.forward(x) * dout), x)
    err = relative_error(dx, numeric)
    assert err < 1e-5, f"{what}: dx disagrees with finite differences (relative error {err:.2e})."


# ------------------------------------------------------------ Linear
def test_linear_is_born_with_the_right_shapes():
    lin = room.Linear(3, 5, rng)
    assert lin.params["W"].shape == (3, 5), f"W must be (in, out) = (3, 5), got {lin.params['W'].shape}"
    assert lin.params["b"].shape == (5,), f"b must be (out,) = (5,), got {lin.params['b'].shape}"
    assert np.all(lin.params["b"] == 0.0), "b starts at zero."
    assert lin.params["W"].std() > 0, "W must be random, not constant: identical neurons stay identical forever."
    assert set(lin.grads) == {"W", "b"} and lin.grads["W"].shape == (3, 5) and lin.grads["b"].shape == (5,)
    assert np.all(lin.grads["W"] == 0.0) and np.all(lin.grads["b"] == 0.0), "grads start at zero."


def test_linear_forward_is_x_at_W_plus_b():
    lin = room.Linear(3, 5, rng)
    lin.params["b"][...] = rng.standard_normal(5)
    x = rng.standard_normal((4, 3))
    out = lin.forward(x)
    assert out.shape == (4, 5), f"(4, 3) @ (3, 5) + (5,) should be (4, 5), got {out.shape}"
    np.testing.assert_allclose(out, x @ lin.params["W"] + lin.params["b"], atol=1e-12)


def test_linear_backward_dx_survives_finite_differences():
    lin = room.Linear(3, 5, rng)
    x = rng.standard_normal((4, 3))
    dout = rng.standard_normal((4, 5))
    check_layer_input_gradient(lin, x, dout, "Linear")


def test_linear_dW_is_x_transpose_times_dout():
    lin = room.Linear(3, 5, rng)
    x = rng.standard_normal((7, 3))
    dout = rng.standard_normal((7, 5))
    lin.zero_grad()
    lin.forward(x)
    lin.backward(dout)
    dW = lin.grads["W"]
    assert dW.shape == (3, 5), (
        f"dW must have W's shape (3, 5), got {dW.shape}. dW = x.T @ dout: (in, N) @ (N, out) -> (in, out)."
    )
    np.testing.assert_allclose(dW, x.T @ dout, atol=1e-10, err_msg="dW must equal x.T @ dout.")
    numeric = numerical_gradient(lambda: np.sum(lin.forward(x) * dout), lin.params["W"])
    err = relative_error(dW, numeric)
    assert err < 1e-5, f"dW disagrees with finite differences (relative error {err:.2e})."


def test_linear_db_sums_over_the_batch_axis():
    lin = room.Linear(3, 5, rng)
    x = rng.standard_normal((7, 3))
    dout = rng.standard_normal((7, 5))
    lin.zero_grad()
    lin.forward(x)
    lin.backward(dout)
    db = lin.grads["b"]
    assert db.shape == (5,), (
        f"db must have b's shape (5,), got {db.shape}. Every example in the batch pushes on the same bias, "
        "so db = dout.sum(axis=0)."
    )
    np.testing.assert_allclose(db, dout.sum(axis=0), atol=1e-10, err_msg="db must be dout summed over axis 0.")


def test_linear_backward_when_every_shape_is_square_and_transposes_can_hide():
    # N == in == out: a transpose on the wrong operand still produces the right SHAPE.
    lin = room.Linear(4, 4, rng)
    x = rng.standard_normal((4, 4))
    dout = rng.standard_normal((4, 4))
    lin.zero_grad()
    lin.forward(x)
    dx = lin.backward(dout)
    np.testing.assert_allclose(lin.grads["W"], x.T @ dout, atol=1e-10, err_msg=(
        "With square shapes x @ dout.T has the right shape and the wrong value. dW = x.T @ dout."
    ))
    np.testing.assert_allclose(dx, dout @ lin.params["W"].T, atol=1e-10, err_msg=(
        "dx = dout @ W.T, not W @ dout or dout @ W. With square shapes only the value tells you."
    ))


def test_gradients_accumulate_until_zero_grad_wipes_the_slate():
    lin = room.Linear(3, 2, rng)
    x = rng.standard_normal((5, 3))
    dout = rng.standard_normal((5, 2))
    lin.zero_grad()
    lin.forward(x)
    lin.backward(dout)
    once = lin.grads["W"].copy()
    lin.forward(x)
    lin.backward(dout)
    np.testing.assert_allclose(lin.grads["W"], 2 * once, atol=1e-10, err_msg=(
        "backward must ADD into grads (two identical passes give twice the gradient). "
        "Use += rather than =. That is why zero_grad() exists."
    ))
    lin.zero_grad()
    assert np.all(lin.grads["W"] == 0.0) and np.all(lin.grads["b"] == 0.0), "zero_grad() must reset every gradient."


# ------------------------------------------------------------ ReLU / Tanh
def test_relu_forward_clamps_the_negatives():
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    out = room.ReLU().forward(x)
    np.testing.assert_array_equal(out, [[0.0, 0.0, 0.0, 0.5, 2.0]])


def test_relu_backward_lets_gradient_through_only_where_x_was_positive():
    relu = room.ReLU()
    x = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dout = np.array([[10.0, 20.0, 30.0, 40.0]])
    relu.forward(x)
    dx = relu.backward(dout)
    np.testing.assert_array_equal(dx, [[0.0, 0.0, 30.0, 40.0]], err_msg=(
        "ReLU's gradient is dout where x > 0 and 0 elsewhere. Mask dout, do not clamp it."
    ))


def test_relu_backward_survives_finite_differences():
    x = rng.standard_normal((6, 5)) + 0.01  # nudge off exact zeros
    dout = rng.standard_normal((6, 5))
    check_layer_input_gradient(room.ReLU(), x, dout, "ReLU")


def test_tanh_forward_and_backward():
    tanh = room.Tanh()
    x = rng.standard_normal((6, 5))
    out = tanh.forward(x)
    np.testing.assert_allclose(out, np.tanh(x), atol=1e-12)
    dout = rng.standard_normal((6, 5))
    check_layer_input_gradient(tanh, x, dout, "Tanh")
    tanh.forward(x)
    np.testing.assert_allclose(tanh.backward(dout), dout * (1 - np.tanh(x) ** 2), atol=1e-10, err_msg=(
        "d tanh / dx = 1 - tanh(x)^2 = 1 - out^2."
    ))


# ------------------------------------------------------------ SoftmaxCrossEntropy
def naive_softmax_ce(logits, y):
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def test_softmax_ce_matches_the_naive_formula_on_tame_logits():
    logits = rng.standard_normal((6, 4))
    y = rng.integers(0, 4, size=6)
    loss = room.SoftmaxCrossEntropy().forward(logits, y)
    assert isinstance(loss, float), f"forward must return a Python float, got {type(loss).__name__}"
    np.testing.assert_allclose(loss, naive_softmax_ce(logits, y), atol=1e-10)


def test_softmax_ce_of_uniform_logits_is_log_of_the_class_count():
    logits = np.zeros((5, 7))
    y = np.arange(5)
    np.testing.assert_allclose(room.SoftmaxCrossEntropy().forward(logits, y), np.log(7), atol=1e-12, err_msg=(
        "With all logits equal every class has probability 1/C, so the loss is log(C)."
    ))


def test_softmax_ce_survives_enormous_logits():
    logits = np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0]])
    y = np.array([0, 1])
    loss = room.SoftmaxCrossEntropy().forward(logits, y)
    assert np.isfinite(loss), "exp(1000) is inf and log(0) is -inf. Subtract the row max; use log-sum-exp."
    np.testing.assert_allclose(loss, (0.0 + np.log(3)) / 2, atol=1e-9)
    logits = np.array([[0.0, 1000.0]])
    loss = room.SoftmaxCrossEntropy().forward(logits, np.array([0]))
    assert np.isfinite(loss) and loss > 900, (
        "log softmax of a hopeless class must be very negative, not -inf: compute log_softmax directly, "
        "never log(prob) after prob rounds to 0."
    )


def test_softmax_ce_backward_is_probs_minus_onehot_over_N():
    logits = rng.standard_normal((6, 4))
    y = rng.integers(0, 4, size=6)
    ce = room.SoftmaxCrossEntropy()
    ce.forward(logits, y)
    dlogits = ce.backward()
    assert dlogits.shape == (6, 4), f"dlogits must have the logits' shape (6, 4), got {dlogits.shape}"
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    onehot = np.eye(4)[y]
    np.testing.assert_allclose(dlogits, (probs - onehot) / 6, atol=1e-10, err_msg=(
        "dlogits = (softmax - onehot) / N. If you are off by a factor of N you forgot the loss is a MEAN."
    ))
    np.testing.assert_allclose(dlogits.sum(axis=1), 0.0, atol=1e-12, err_msg=(
        "Every row of dlogits sums to zero: pushing one logit up pushes the others down by the same total."
    ))


def test_softmax_ce_backward_survives_finite_differences():
    logits = rng.standard_normal((5, 3))
    y = rng.integers(0, 3, size=5)
    ce = room.SoftmaxCrossEntropy()
    ce.forward(logits, y)
    dlogits = ce.backward()
    numeric = numerical_gradient(lambda: ce.forward(logits, y), logits)
    err = relative_error(dlogits, numeric)
    assert err < 1e-5, f"dlogits disagrees with finite differences (relative error {err:.2e})."


# ------------------------------------------------------------ everything at once
def test_a_two_layer_network_gradient_checks_end_to_end():
    lin1, act, lin2, ce = room.Linear(3, 4, rng), room.Tanh(), room.Linear(4, 2, rng), room.SoftmaxCrossEntropy()
    x = rng.standard_normal((5, 3))
    y = rng.integers(0, 2, size=5)

    def loss():
        return ce.forward(lin2.forward(act.forward(lin1.forward(x))), y)

    for layer in (lin1, lin2):
        layer.zero_grad()
    loss()
    lin1.backward(act.backward(lin2.backward(ce.backward())))
    for name, layer in (("lin1", lin1), ("lin2", lin2)):
        for pname in ("W", "b"):
            numeric = numerical_gradient(loss, layer.params[pname])
            err = relative_error(layer.grads[pname], numeric)
            assert err < 1e-5, (
                f"d loss / d {name}.{pname} disagrees with finite differences (relative error {err:.2e}). "
                "Each layer's backward must return dx so the layer below can continue."
            )
