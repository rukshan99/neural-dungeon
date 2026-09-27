"""TRIAL 3.3 - THE BELLOWS

The plumbing first (minibatches cover every index, Sequential chains forward and
backward, SGD applies exactly the rule), then the fire: a small MLP must learn
the spirals.
"""

import math

import numpy as np

from dungeon.artifacts.toydata import make_spirals
from dungeon.trials import load_room

anvil = load_room(__file__, "room_1_the_anvil")
room = load_room(__file__, "room_3_the_bellows")


def he_mlp(sizes, rng):
    """Linear/ReLU stack with He init, built from Room 1's layers (the init is applied by hand)."""
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = anvil.Linear(fan_in, fan_out, rng)
        lin.params["W"][...] = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(anvil.ReLU())
    return room.Sequential(layers)


class LossSpy:
    """Wraps Room 1's SoftmaxCrossEntropy and records (batch loss, batch size) for every forward."""

    def __init__(self):
        self.inner = anvil.SoftmaxCrossEntropy()
        self.seen = []

    def forward(self, logits, y):
        value = self.inner.forward(logits, y)
        self.seen.append((value, len(y)))
        return value

    def backward(self):
        return self.inner.backward()


class ModeSpy:
    """A parameter-free layer that only records which mode it was in when forward ran."""

    def __init__(self):
        self.training = True
        self.params, self.grads = {}, {}
        self.seen_modes = []

    def forward(self, x):
        self.seen_modes.append(self.training)
        return x

    def backward(self, dout):
        return dout

    def zero_grad(self):
        pass


# ------------------------------------------------------------ iterate_minibatches
def test_every_index_is_visited_exactly_once_per_epoch():
    batches = list(room.iterate_minibatches(23, 5, np.random.default_rng(0)))
    assert all(isinstance(b, np.ndarray) for b in batches), "yield numpy index arrays"
    sizes = [len(b) for b in batches]
    assert sizes == [5, 5, 5, 5, 3], f"23 examples in batches of 5 -> sizes [5, 5, 5, 5, 3], got {sizes}. Keep the short last batch."
    seen = np.sort(np.concatenate(batches))
    np.testing.assert_array_equal(seen, np.arange(23), err_msg="Every index 0..22 exactly once. No repeats, no gaps.")


def test_batches_are_shuffled_reproducibly():
    a = np.concatenate(list(room.iterate_minibatches(50, 10, np.random.default_rng(3))))
    b = np.concatenate(list(room.iterate_minibatches(50, 10, np.random.default_rng(3))))
    c = np.concatenate(list(room.iterate_minibatches(50, 10, np.random.default_rng(4))))
    assert not np.array_equal(a, np.arange(50)), "shuffle=True must not visit examples in order 0, 1, 2, ..."
    np.testing.assert_array_equal(a, b, err_msg="the same rng seed must give the same order (use rng.permutation)")
    assert not np.array_equal(a, c), "different seeds must give different orders"


def test_shuffle_false_walks_in_order():
    batches = list(room.iterate_minibatches(7, 3, np.random.default_rng(0), shuffle=False))
    np.testing.assert_array_equal(np.concatenate(batches), np.arange(7))
    assert [len(b) for b in batches] == [3, 3, 1]


# ------------------------------------------------------------ Sequential
def test_sequential_forward_chains_the_layers_in_order():
    rng = np.random.default_rng(1)
    lin1, act, lin2 = anvil.Linear(3, 4, rng), anvil.ReLU(), anvil.Linear(4, 2, rng)
    model = room.Sequential([lin1, act, lin2])
    x = rng.standard_normal((5, 3))
    np.testing.assert_allclose(model.forward(x), lin2.forward(act.forward(lin1.forward(x))), atol=1e-12)


def test_sequential_params_are_the_layers_own_arrays_with_indexed_names():
    rng = np.random.default_rng(2)
    lin1, lin2 = anvil.Linear(3, 4, rng), anvil.Linear(4, 2, rng)
    model = room.Sequential([lin1, anvil.ReLU(), lin2])
    params = model.params()
    assert set(params) == {"0.W", "0.b", "2.W", "2.b"}, f"keys must be '<layer index>.<name>', got {sorted(params)}"
    assert params["0.W"] is lin1.params["W"] and params["2.b"] is lin2.params["b"], (
        "params() must return the layers' own arrays, not copies, so in-place updates reach the model."
    )
    assert set(model.grads()) == set(params), "grads() must have exactly the keys of params()"


def test_sequential_backward_runs_in_reverse_and_gradient_checks():
    rng = np.random.default_rng(3)
    model = room.Sequential([anvil.Linear(3, 4, rng), anvil.Tanh(), anvil.Linear(4, 2, rng)])
    loss = anvil.SoftmaxCrossEntropy()
    x = rng.standard_normal((6, 3))
    y = rng.integers(0, 2, size=6)

    def L():
        return loss.forward(model.forward(x), y)

    model.zero_grad()
    L()
    dx = model.backward(loss.backward())
    assert dx.shape == x.shape, f"Sequential.backward must return dx for the input, shape {x.shape}, got {dx.shape}"
    grads = model.grads()
    for name, p in model.params().items():
        numeric = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + 1e-6
            fp = L()
            p[idx] = old - 1e-6
            fm = L()
            p[idx] = old
            numeric[idx] = (fp - fm) / 2e-6
        err = np.max(np.abs(grads[name] - numeric) / np.maximum(1e-8, np.abs(grads[name]) + np.abs(numeric)))
        assert err < 1e-5, f"gradient of {name} disagrees with finite differences (relative error {err:.2e}). Is backward running the layers in REVERSE?"


def test_zero_grad_reaches_every_layer():
    rng = np.random.default_rng(4)
    model = room.Sequential([anvil.Linear(3, 4, rng), anvil.ReLU(), anvil.Linear(4, 2, rng)])
    loss = anvil.SoftmaxCrossEntropy()
    loss.forward(model.forward(rng.standard_normal((5, 3))), rng.integers(0, 2, size=5))
    model.backward(loss.backward())
    assert any(np.any(g != 0) for g in model.grads().values()), "after a backward, some gradient must be non-zero"
    model.zero_grad()
    assert all(np.all(g == 0) for g in model.grads().values()), "zero_grad() must reset every layer's gradients"


def test_train_and_eval_flip_the_switch_on_every_layer_that_has_one():
    spy = ModeSpy()
    model = room.Sequential([anvil.Linear(2, 2), spy])
    model.eval()
    assert spy.training is False and model.training is False
    model.train()
    assert spy.training is True and model.training is True


# ------------------------------------------------------------ SGD
def test_sgd_applies_exactly_the_update_rule():
    W = np.array([[1.0, -2.0], [0.5, 4.0]])
    b = np.array([1.0, -1.0])
    params = {"0.W": W, "0.b": b}
    grads = {"0.W": np.array([[0.1, 0.2], [0.3, 0.4]]), "0.b": np.array([0.5, 0.6])}
    opt = room.SGD(params, lr=0.1, weight_decay=0.5)
    opt.step(grads)
    np.testing.assert_allclose(W, [[1.0, -2.0], [0.5, 4.0]] - 0.1 * (grads["0.W"] + 0.5 * np.array([[1.0, -2.0], [0.5, 4.0]])), atol=1e-12, err_msg=(
        "W <- W - lr * (g + weight_decay * W)"
    ))
    np.testing.assert_allclose(b, [1.0, -1.0] - 0.1 * grads["0.b"], atol=1e-12, err_msg=(
        "biases are never decayed: b <- b - lr * g"
    ))
    assert params["0.W"] is W, "update in place (p -= ...); rebinding detaches the optimizer from the model"


def test_sgd_without_weight_decay_is_plain_gradient_descent():
    W = np.ones((2, 2))
    opt = room.SGD({"W": W}, lr=0.5)
    opt.step({"W": np.full((2, 2), 2.0)})
    np.testing.assert_allclose(W, 0.0, atol=1e-12)


# ------------------------------------------------------------ accuracy
class Oracle:
    """A 'model' with fixed logits, to test accuracy() without training anything."""

    def __init__(self, logits):
        self.logits = logits
        self.training = True
        self.modes = []

    def forward(self, X):
        self.modes.append(self.training)
        return self.logits

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def test_accuracy_counts_argmax_hits_in_eval_mode_and_restores_the_mode():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]])
    oracle = Oracle(logits)
    acc = room.accuracy(oracle, np.zeros((4, 2)), np.array([0, 1, 2, 1]))
    assert isinstance(acc, float) and abs(acc - 0.75) < 1e-12, f"3 of 4 argmaxes match: accuracy 0.75, got {acc}"
    assert oracle.modes == [False], "accuracy() must run the forward pass in eval mode"
    assert oracle.training is True, "accuracy() must put the model back in the mode it found it in"


# ------------------------------------------------------------ the fire
def test_train_pumps_one_batch_at_a_time():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((45, 2))
    y = rng.integers(0, 3, size=45)
    model = he_mlp([2, 8, 3], rng)
    calls = []
    original = model.forward
    model.forward = lambda x: (calls.append(len(x)), original(x))[1]
    loss = LossSpy()
    history = room.train(model, loss, room.SGD(model.params(), 0.1), X, y, epochs=2, batch_size=10, rng=rng)
    assert len(history) == 2, f"one loss per epoch: expected 2, got {len(history)}"
    assert len(calls) == 2 * math.ceil(45 / 10), f"45 examples, batch 10 -> 5 forwards per epoch; saw {len(calls)} over 2 epochs"
    assert sorted(calls[:5]) == [5, 10, 10, 10, 10], f"batch sizes in one epoch should be four 10s and a 5, got {sorted(calls[:5])}"
    # The epoch's loss is the per-EXAMPLE mean: weight each batch loss by its size, so the short last batch counts for 5/45, not 1/5.
    first_epoch = loss.seen[:5]
    weighted = sum(v * n for v, n in first_epoch) / 45
    unweighted = sum(v for v, _ in first_epoch) / 5
    assert math.isclose(history[0], weighted, rel_tol=1e-9), (
        f"epoch loss must be sum(batch_loss * len(idx)) / n = {weighted:.6f}, got {history[0]:.6f}"
        + (" (that is the plain mean of the five batch losses, which over-weights the short last batch)."
           if math.isclose(history[0], unweighted, rel_tol=1e-9) else ".")
    )


def test_the_bellows_teach_a_small_mlp_the_spirals():
    X, y = make_spirals(n_per_class=100, n_classes=3, noise=0.05, seed=0)
    X = X.astype(np.float64)
    rng = np.random.default_rng(0)
    model = he_mlp([2, 64, 64, 3], rng)
    opt = room.SGD(model.params(), lr=0.3)
    history = room.train(model, anvil.SoftmaxCrossEntropy(), opt, X, y, epochs=800, batch_size=32, rng=rng)
    assert len(history) == 800 and all(isinstance(v, float) for v in history), "return one Python float per epoch"
    assert all(np.isfinite(history)), "the loss became nan/inf: check your softmax stability and the learning rate"
    assert history[-1] < 0.5 * history[0], (
        f"the loss barely moved: {history[0]:.3f} -> {history[-1]:.3f}. Is the optimizer stepping? Are grads zeroed each batch?"
    )
    acc = room.accuracy(model, X, y)
    assert acc >= 0.90, (
        f"train accuracy {acc:.3f} after 800 epochs; the forge expects at least 0.90 on these spirals. "
        "Check: shuffling each epoch, zero_grad before backward, step with the model's own grads()."
    )


def test_the_loss_does_not_merely_wobble():
    X, y = make_spirals(n_per_class=50, n_classes=3, noise=0.05, seed=1)
    X = X.astype(np.float64)
    rng = np.random.default_rng(1)
    model = he_mlp([2, 32, 3], rng)
    history = room.train(model, anvil.SoftmaxCrossEntropy(), room.SGD(model.params(), lr=0.2), X, y, epochs=60, batch_size=16, rng=rng)
    first, last = np.mean(history[:5]), np.mean(history[-5:])
    assert last < first, f"mean loss over the last 5 epochs ({last:.3f}) should be below the first 5 ({first:.3f})"
