"""TRIAL 2.2 - THE NEURON'S WHISPER

Count the parameters, silence them, check the gradient numerically, then
teach four points to whisper XOR.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_neurons_whisper")
Value = room.Value


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ neuron
def test_a_neuron_holds_one_weight_per_input_plus_a_bias():
    n = room.Neuron(3, _rng())
    params = n.parameters()
    assert len(params) == 4, f"Neuron(3) should own 3 weights + 1 bias = 4 parameters, got {len(params)}"
    assert all(isinstance(p, Value) for p in params), "every parameter must be a Value from Room 2.1"
    assert n.b.data == 0.0, f"the bias starts at 0.0 by contract, got {n.b.data}"
    assert all(-1.0 <= w.data <= 1.0 for w in n.w), "weights are drawn from uniform(-1, 1)"


def test_the_same_seed_builds_the_same_neuron():
    n1 = room.Neuron(5, _rng(7))
    n2 = room.Neuron(5, _rng(7))
    assert [w.data for w in n1.w] == [w.data for w in n2.w], (
        "two Neurons built from generators with the same seed must have identical weights. "
        "Draw the weights one at a time, in order, from the rng you were given."
    )
    n3 = room.Neuron(5, _rng(8))
    assert [w.data for w in n1.w] != [w.data for w in n3.w], "a different seed must give different weights"


def test_a_neuron_computes_tanh_of_a_weighted_sum():
    n = room.Neuron(3, _rng())
    for w, val in zip(n.w, [0.5, -1.0, 2.0]):
        w.data = val
    n.b.data = 0.25
    x = [1.0, 2.0, -0.5]
    pre = 0.5 * 1.0 - 1.0 * 2.0 + 2.0 * -0.5 + 0.25  # -2.25
    out = n(x)
    assert isinstance(out, Value), f"a Neuron returns ONE Value, got {type(out).__name__}"
    assert math.isclose(out.data, math.tanh(pre)), f"tanh(w.x + b) = tanh({pre}) = {math.tanh(pre)}, got {out.data}"
    linear = room.Neuron(3, _rng(), nonlin=False)
    for w, val in zip(linear.w, [0.5, -1.0, 2.0]):
        w.data = val
    linear.b.data = 0.25
    assert math.isclose(linear(x).data, pre), f"with nonlin=False the output is w.x + b = {pre}, got {linear(x).data}"


def test_a_neuron_output_is_in_the_graph():
    n = room.Neuron(2, _rng())
    out = n([Value(1.0), Value(-1.0)])
    out.backward()
    assert any(p.grad != 0.0 for p in n.parameters()), (
        "backward() from the neuron's output must reach its parameters. "
        "Every + and * has to go through Value operations (sum needs a Value start)."
    )


# ----------------------------------------------------------------- layer
def test_a_layer_is_a_column_of_neurons():
    layer = room.Layer(3, 5, _rng())
    assert len(layer.parameters()) == 20, f"Layer(3, 5) owns 5 x (3 + 1) = 20 parameters, got {len(layer.parameters())}"
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 5, f"Layer(3, 5) returns a list of 5 Values, got {out!r}"
    assert all(isinstance(v, Value) for v in out)


# ------------------------------------------------------------------- mlp
@pytest.mark.parametrize(
    "n_in,n_outs,expected",
    [(2, [8, 1], 33), (3, [4, 4, 2], 46), (1, [1], 2)],
    ids=["2-8-1", "3-4-4-2", "1-1"],
)
def test_the_mlp_counts_its_parameters(n_in, n_outs, expected):
    mlp = room.MLP(n_in, n_outs, _rng())
    got = len(mlp.parameters())
    assert got == expected, (
        f"MLP({n_in}, {n_outs}) should have {expected} parameters "
        f"(sum over layers of n_out * (n_in + 1)), got {got}"
    )


def test_the_mlp_output_is_a_list_with_the_last_layer_size():
    mlp = room.MLP(2, [8, 3], _rng())
    out = mlp([0.5, -0.5])
    assert isinstance(out, list) and len(out) == 3, f"MLP(2, [8, 3]) returns a list of 3 Values, got {out!r}"


def test_the_last_layer_speaks_plainly_without_tanh():
    mlp = room.MLP(2, [8, 1], _rng())
    for p in mlp.parameters():
        p.data = 1.0
    out = mlp([1.0, 1.0])[0].data
    # hidden units are tanh(3) ~ 0.995 each; the linear output is 8 * 0.995 + 1 ~ 8.96.
    assert out > 1.0, (
        f"with every parameter set to 1.0 the output should be about 8.96, got {out:.3f}. "
        "The last layer must be linear: a tanh output can never exceed 1."
    )


def test_zero_grad_silences_every_parameter():
    mlp = room.MLP(2, [4, 1], _rng())
    loss = room.mse_loss([mlp(x)[0] for x in room.XOR_INPUTS], room.XOR_TARGETS)
    loss.backward()
    assert any(p.grad != 0.0 for p in mlp.parameters()), "backward() should have reached the parameters"
    mlp.zero_grad()
    loud = [p for p in mlp.parameters() if p.grad != 0.0]
    assert not loud, f"{len(loud)} parameter(s) still hold a gradient after zero_grad(); all must be exactly 0.0"


def test_mse_loss_is_the_mean_of_squared_errors():
    preds = [Value(1.0), Value(-2.0), Value(0.5)]
    loss = room.mse_loss(preds, [0.0, -1.0, 1.5])
    assert isinstance(loss, Value), "mse_loss must return a Value so it can be differentiated"
    expected = (1.0 + 1.0 + 1.0) / 3
    assert math.isclose(loss.data, expected), f"mean of squared errors should be {expected}, got {loss.data}"
    loss.backward()
    assert math.isclose(preds[0].grad, 2 * (1.0 - 0.0) / 3), (
        f"d mse / d pred_0 = 2 (pred - target) / n = {2 / 3:.4f}, got {preds[0].grad:.4f}"
    )


def test_the_mlp_gradient_agrees_with_finite_differences():
    mlp = room.MLP(2, [3, 1], _rng(3))
    xs, ys = room.XOR_INPUTS, room.XOR_TARGETS

    def loss_value() -> float:
        return room.mse_loss([mlp(x)[0] for x in xs], ys).data

    mlp.zero_grad()
    room.mse_loss([mlp(x)[0] for x in xs], ys).backward()
    params = mlp.parameters()
    for idx in (0, 4, len(params) - 1):
        p = params[idx]
        analytic = p.grad
        old = p.data
        h = 1e-6
        p.data = old + h
        up = loss_value()
        p.data = old - h
        down = loss_value()
        p.data = old
        numeric = (up - down) / (2 * h)
        assert math.isclose(analytic, numeric, rel_tol=1e-5, abs_tol=1e-8), (
            f"parameter {idx}: backward says {analytic:.6f}, finite differences say {numeric:.6f}"
        )


# -------------------------------------------------------------- training
def test_the_row_of_whisperers_learns_xor():
    model, losses = room.train_xor(steps=200, lr=0.1, seed=0)
    assert len(losses) == 200, f"one loss per step: expected 200 entries, got {len(losses)}"
    assert all(isinstance(v, float) for v in losses), "record loss.data (a float) each step, not the Value"
    assert losses[-1] < 0.05, (
        f"after 200 steps the XOR loss should be below 0.05, yours is {losses[-1]:.4f} "
        f"(started at {losses[0]:.4f}). Did you zero the gradients before each backward()? "
        "Did you subtract lr * grad rather than add it?"
    )
    assert losses[-1] < losses[0] / 10, "the loss should fall by at least 10x over training"
    preds = [model(x)[0].data for x in room.XOR_INPUTS]
    signs_ok = all((p > 0) == (t > 0) for p, t in zip(preds, room.XOR_TARGETS))
    assert signs_ok, f"the trained model predicts {np.round(preds, 3).tolist()} for targets {room.XOR_TARGETS}"


def test_the_same_seed_tells_the_same_story():
    _, first = room.train_xor(steps=60, lr=0.1, seed=11)
    _, second = room.train_xor(steps=60, lr=0.1, seed=11)
    assert np.allclose(first, second, rtol=1e-7, atol=1e-10), (
        "training with the same seed must reproduce the same losses (up to floating-point order). "
        f"Largest difference: {np.max(np.abs(np.array(first) - np.array(second))):.3e}"
    )
    _, third = room.train_xor(steps=60, lr=0.1, seed=12)
    assert not np.allclose(first, third), "a different seed should tell a different story"
