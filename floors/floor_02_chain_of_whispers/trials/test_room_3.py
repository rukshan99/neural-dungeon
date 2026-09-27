"""TRIAL 2.3 - THE TENSOR WHISPER

Every op is checked two ways: its forward against numpy, and its backward
against central finite differences of that numpy forward, through a random
upstream signal so that transposes and reshapes cannot hide behind ones.
Then a two-layer MLP is compared with a hand-derived backward pass.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_3_tensor_whisper")
Tensor = room.Tensor

rng = np.random.default_rng(23)


# ------------------------------------------------------------- machinery
def _numerical_grad(f, arrays: list[np.ndarray], which: int, h: float = 1e-6) -> np.ndarray:
    """d f(arrays) / d arrays[which] by central differences, one element at a time."""
    x = arrays[which]
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + h
        up = f(*arrays)
        x[idx] = old - h
        down = f(*arrays)
        x[idx] = old
        g[idx] = (up - down) / (2 * h)
    return g


def _positive(shape):
    return rng.random(shape) + 0.5


def _away_from_zero(shape):
    x = rng.standard_normal(shape)
    return np.where(np.abs(x) < 0.15, 0.5, x)


def _check_op(name: str, op_t, op_np, *arrays: np.ndarray):
    """Forward vs numpy; backward vs finite differences of  sum(op(arrays) * signal)."""
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    tensors = [Tensor(a) for a in arrays]
    out = op_t(*tensors)
    expected = np.asarray(op_np(*arrays), dtype=np.float64)
    assert isinstance(out, Tensor), f"{name}: the op must return a Tensor, got {type(out).__name__}"
    assert out.shape == expected.shape, f"{name}: forward shape should be {expected.shape}, got {out.shape}"
    np.testing.assert_allclose(out.data, expected, rtol=1e-10, atol=1e-12, err_msg=f"{name}: forward value is wrong")

    signal = rng.standard_normal(expected.shape)  # a non-trivial upstream gradient
    loss = (out * Tensor(signal)).sum()
    loss.backward()

    def scalar(*arrs):
        return float(np.sum(op_np(*arrs) * signal))

    for i, (t, a) in enumerate(zip(tensors, arrays)):
        assert t.grad.shape == a.shape, (
            f"{name}: grad of input {i} has shape {t.grad.shape} but its data has shape {a.shape}. "
            "grad.shape must always equal data.shape: unbroadcast the upstream gradient."
        )
        numeric = _numerical_grad(scalar, [x.copy() for x in arrays], i)
        gap = np.max(np.abs(t.grad - numeric))
        assert np.allclose(t.grad, numeric, rtol=1e-5, atol=1e-6), (
            f"{name}: backward for input {i} (shape {a.shape}) disagrees with finite differences; "
            f"largest gap {gap:.3e}.\n  yours:   {np.round(t.grad, 4).tolist()}\n  numeric: {np.round(numeric, 4).tolist()}"
        )


# ---------------------------------------------------------- unbroadcast
def test_unbroadcast_sums_away_an_inserted_leading_axis():
    got = room.unbroadcast(np.ones((3, 4)), (4,))
    assert got.shape == (4,), f"unbroadcast(ones((3, 4)), (4,)) should have shape (4,), got {got.shape}"
    np.testing.assert_array_equal(got, [3.0, 3.0, 3.0, 3.0])


def test_unbroadcast_sums_a_stretched_axis_and_keeps_it_as_one():
    got = room.unbroadcast(np.ones((3, 4)), (3, 1))
    assert got.shape == (3, 1), f"expected shape (3, 1), got {got.shape}. Sum with keepdims=True over the stretched axis."
    np.testing.assert_array_equal(got, [[4.0], [4.0], [4.0]])


def test_unbroadcast_handles_both_kinds_of_stretching_at_once():
    got = room.unbroadcast(np.ones((2, 3, 4)), (3, 1))
    assert got.shape == (3, 1), f"expected (3, 1), got {got.shape}"
    np.testing.assert_array_equal(got, [[8.0], [8.0], [8.0]])


def test_unbroadcast_leaves_a_matching_shape_alone_and_collapses_to_a_scalar():
    g = rng.standard_normal((2, 5))
    np.testing.assert_array_equal(room.unbroadcast(g, (2, 5)), g)
    s = room.unbroadcast(np.ones(5), ())
    assert np.shape(s) == () and float(s) == 5.0, f"a scalar broadcast to (5,) gets the sum back: 5.0 with shape (), got {s!r}"


# ------------------------------------------------------------ the ops
def test_a_tensor_starts_with_a_zero_gradient_of_its_own_shape():
    t = Tensor(np.ones((2, 3)))
    assert t.grad.shape == (2, 3) and np.all(t.grad == 0.0)
    assert t.data.dtype == np.float64


OPS = [
    ("add_same_shape", lambda a, b: a + b, lambda a, b: a + b, [(3, 4), (3, 4)]),
    ("add_(3,4)+(4,)", lambda a, b: a + b, lambda a, b: a + b, [(3, 4), (4,)]),
    ("add_(3,1)+(1,4)", lambda a, b: a + b, lambda a, b: a + b, [(3, 1), (1, 4)]),
    ("add_(2,3,4)+(3,1)", lambda a, b: a + b, lambda a, b: a + b, [(2, 3, 4), (3, 1)]),
    ("add_scalar_2.5+x", lambda a: 2.5 + a, lambda a: 2.5 + a, [(2, 3)]),
    ("sub_(2,3)-(3,)", lambda a, b: a - b, lambda a, b: a - b, [(2, 3), (3,)]),
    ("rsub_1-x", lambda a: 1.0 - a, lambda a: 1.0 - a, [(2, 2)]),
    ("neg", lambda a: -a, lambda a: -a, [(3, 2)]),
    ("mul_same_shape", lambda a, b: a * b, lambda a, b: a * b, [(3, 4), (3, 4)]),
    ("mul_(3,1)*(1,4)", lambda a, b: a * b, lambda a, b: a * b, [(3, 1), (1, 4)]),
    ("mul_(2,3,4)*(4,)", lambda a, b: a * b, lambda a, b: a * b, [(2, 3, 4), (4,)]),
    ("mul_scalar_3*x", lambda a: 3.0 * a, lambda a: 3.0 * a, [(2, 3)]),
    ("matmul_(2,3)@(3,4)", lambda a, b: a @ b, lambda a, b: a @ b, [(2, 3), (3, 4)]),
    ("matmul_(1,5)@(5,1)", lambda a, b: a @ b, lambda a, b: a @ b, [(1, 5), (5, 1)]),
    ("sum_all", lambda a: a.sum(), lambda a: a.sum(), [(2, 3)]),
    ("sum_axis0", lambda a: a.sum(axis=0), lambda a: a.sum(axis=0), [(3, 4)]),
    ("sum_axis1_keepdims", lambda a: a.sum(axis=1, keepdims=True), lambda a: a.sum(axis=1, keepdims=True), [(3, 4)]),
    ("sum_axis=(0,2)", lambda a: a.sum(axis=(0, 2)), lambda a: a.sum(axis=(0, 2)), [(2, 3, 4)]),
    ("sum_axis=-1", lambda a: a.sum(axis=-1), lambda a: a.sum(axis=-1), [(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), lambda a: a.mean(), [(3, 4)]),
    ("mean_axis1_keepdims", lambda a: a.mean(axis=1, keepdims=True), lambda a: a.mean(axis=1, keepdims=True), [(3, 4)]),
    ("mean_axis0", lambda a: a.mean(axis=0), lambda a: a.mean(axis=0), [(5, 2)]),
    ("reshape_(2,6)->(3,4)", lambda a: a.reshape(3, 4), lambda a: a.reshape(3, 4), [(2, 6)]),
    ("reshape_tuple_(2,3,4)->(4,6)", lambda a: a.reshape((4, 6)), lambda a: a.reshape(4, 6), [(2, 3, 4)]),
    ("transpose_2d", lambda a: a.transpose(), lambda a: a.T, [(2, 3)]),
    ("T_property", lambda a: a.T, lambda a: a.T, [(3, 5)]),
    ("transpose_(2,0,1)", lambda a: a.transpose(2, 0, 1), lambda a: a.transpose(2, 0, 1), [(2, 3, 4)]),
    ("transpose_tuple_(1,0,2)", lambda a: a.transpose((1, 0, 2)), lambda a: a.transpose(1, 0, 2), [(2, 3, 4)]),
    ("pow_3", lambda a: a**3, lambda a: a**3, [(2, 3)]),
    ("pow_-1", lambda a: a**-1, lambda a: a**-1, [(2, 3)]),
    ("truediv_(3,4)/(4,)", lambda a, b: a / b, lambda a, b: a / b, [(3, 4), (4,)]),
]
POSITIVE_INPUT_OPS = [
    ("log", lambda a: a.log(), np.log, [(2, 3)]),
    ("pow_0.5", lambda a: a**0.5, np.sqrt, [(2, 3)]),
]
NONZERO_INPUT_OPS = [
    ("relu", lambda a: a.relu(), lambda a: np.maximum(a, 0.0), [(3, 4)]),
    ("exp", lambda a: a.exp(), np.exp, [(2, 3)]),
]


@pytest.mark.parametrize("name,op_t,op_np,shapes", OPS, ids=[o[0] for o in OPS])
def test_each_op_whispers_back_the_right_gradient(name, op_t, op_np, shapes):
    arrays = [_positive(s) if name.startswith("pow_-1") or name.startswith("truediv") else rng.standard_normal(s) for s in shapes]
    _check_op(name, op_t, op_np, *arrays)


@pytest.mark.parametrize("name,op_t,op_np,shapes", POSITIVE_INPUT_OPS, ids=[o[0] for o in POSITIVE_INPUT_OPS])
def test_ops_that_need_positive_inputs(name, op_t, op_np, shapes):
    _check_op(name, op_t, op_np, *[_positive(s) for s in shapes])


@pytest.mark.parametrize("name,op_t,op_np,shapes", NONZERO_INPUT_OPS, ids=[o[0] for o in NONZERO_INPUT_OPS])
def test_ops_checked_away_from_the_kink(name, op_t, op_np, shapes):
    _check_op(name, op_t, op_np, *[_away_from_zero(s) for s in shapes])


def test_relu_blocks_the_whisper_where_the_input_was_negative():
    x = Tensor(np.array([-2.0, -0.5, 0.5, 3.0]))
    y = x.relu()
    y.backward()
    np.testing.assert_array_equal(y.data, [0.0, 0.0, 0.5, 3.0])
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------- graph rules
def test_the_tensor_diamond_accumulates():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x * x + x
    y.backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1, err_msg=(
        "d(x*x + x)/dx = 2x + 1. x is used three times; the three whispers must be added with +=."
    ))


def test_a_non_scalar_root_is_seeded_with_ones():
    x = Tensor(rng.standard_normal((2, 3)))
    y = x * 2.0
    y.backward()
    np.testing.assert_allclose(y.grad, np.ones((2, 3)), err_msg="the root's grad after backward() is ones_like(data)")
    np.testing.assert_allclose(x.grad, 2.0 * np.ones((2, 3)), err_msg="d sum(2x)/dx = 2 everywhere")


def test_a_second_backward_adds_to_the_first():
    w = Tensor(np.array([0.5, -1.0]))
    x = Tensor(np.array([2.0, 3.0]))
    (w * x).sum().backward()
    (w * x).sum().backward()
    np.testing.assert_allclose(w.grad, 2 * x.data, err_msg=(
        "grads accumulate across backward() calls; two passes through fresh graphs give 2 * x"
    ))


def test_matmul_gradients_have_the_operands_shapes():
    a, b = Tensor(rng.standard_normal((2, 3))), Tensor(rng.standard_normal((3, 4)))
    c = a @ b
    c.sum().backward()
    assert a.grad.shape == (2, 3) and b.grad.shape == (3, 4), (
        f"dA must be (2, 3) and dB (3, 4); got {a.grad.shape} and {b.grad.shape}. "
        "dA = dC @ B.T and dB = A.T @ dC are the only arrangements whose shapes work."
    )
    np.testing.assert_allclose(a.grad, np.ones((2, 4)) @ b.data.T)
    np.testing.assert_allclose(b.grad, a.data.T @ np.ones((2, 4)))


# ------------------------------------------------------------- the mlp
def test_a_two_layer_mlp_matches_the_hand_derived_backward():
    X = rng.standard_normal((5, 3))
    W1 = rng.standard_normal((3, 4)) * 0.7
    b1 = rng.standard_normal(4) * 0.1
    W2 = rng.standard_normal((4, 2)) * 0.7
    b2 = rng.standard_normal(2) * 0.1
    Y = rng.standard_normal((5, 2))

    tX, tW1, tb1, tW2, tb2 = (Tensor(a) for a in (X, W1, b1, W2, b2))
    z = tX @ tW1 + tb1  # (5, 4) + (4,): broadcasting in anger
    h = z.relu()
    out = h @ tW2 + tb2
    loss = ((out - Tensor(Y)) ** 2).mean()
    loss.backward()

    # The same network, differentiated by hand.
    z_np = X @ W1 + b1
    h_np = np.maximum(z_np, 0.0)
    out_np = h_np @ W2 + b2
    np.testing.assert_allclose(loss.data, np.mean((out_np - Y) ** 2), err_msg="forward loss is wrong")
    d_out = 2.0 * (out_np - Y) / out_np.size  # d mean(sq)/d out
    d_W2 = h_np.T @ d_out
    d_b2 = d_out.sum(axis=0)
    d_h = d_out @ W2.T
    d_z = d_h * (z_np > 0.0)
    d_W1 = X.T @ d_z
    d_b1 = d_z.sum(axis=0)
    d_X = d_z @ W1.T

    for name, t, expected in [("W2", tW2, d_W2), ("b2", tb2, d_b2), ("W1", tW1, d_W1), ("b1", tb1, d_b1), ("X", tX, d_X)]:
        assert t.grad.shape == expected.shape, f"grad of {name} should be {expected.shape}, got {t.grad.shape}"
        np.testing.assert_allclose(t.grad, expected, rtol=1e-9, atol=1e-12, err_msg=(
            f"the gradient of {name} disagrees with the hand-derived backward pass"
        ))
