"""BOSS FIGHT - THE OVERFIT HYDRA

Phase 1: the regularizers, exactly (L2 penalty, weight decay in SGD, inverted dropout).
Phase 2: count the heads.
Phase 3: watch an unregularized model grow them, then slay the beast.
"""

import numpy as np
import pytest

from dungeon.artifacts.toydata import make_spirals
from dungeon.trials import load_room

anvil = load_room(__file__, "room_1_the_anvil")
bellows = load_room(__file__, "room_3_the_bellows")
boss = load_room(__file__, "boss_overfit_hydra")

pytestmark = pytest.mark.boss

LIES = 60  # flipped labels in the arena
WILD_HEADS = 20  # an unregularized model grows at least this many
SLAIN_HEADS = 15  # yours may keep at most this many


def hydra_arena(seed=7):
    """(X_train, y_noisy, y_clean, X_val, y_val): 300 training points, 60 of them lied about."""
    X_train, y_clean = make_spirals(n_per_class=100, n_classes=3, noise=0.05, seed=1)
    rng = np.random.default_rng(seed)
    n_flip = int(round(0.2 * len(y_clean)))
    flip = rng.choice(len(y_clean), size=n_flip, replace=False)
    y_noisy = y_clean.copy()
    y_noisy[flip] = (y_clean[flip] + rng.integers(1, 3, size=n_flip)) % 3
    X_val, y_val = make_spirals(n_per_class=200, n_classes=3, noise=0.05, seed=2)
    return X_train.astype(np.float64), y_noisy, y_clean, X_val.astype(np.float64), y_val


def he_mlp(sizes, rng):
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = anvil.Linear(fan_in, fan_out, rng)
        lin.params["W"][...] = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(anvil.ReLU())
    return bellows.Sequential(layers)


class Oracle:
    def __init__(self, pred):
        self.pred = np.asarray(pred)
        self.training = True
        self.modes = []

    def forward(self, X):
        self.modes.append(self.training)
        return np.eye(3)[self.pred]

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


# ------------------------------------------------------------ phase 1: L2
def test_phase_1_l2_penalty_taxes_weight_matrices_and_spares_biases():
    params = {"0.W": np.ones((2, 3)), "0.b": np.full(3, 100.0), "2.W": np.full((1, 1), 2.0), "2.b": np.array([-7.0])}
    value = boss.l2_penalty(params, weight_decay=0.1)
    assert isinstance(value, float), "return a Python float"
    np.testing.assert_allclose(value, 0.5 * 0.1 * (6 * 1.0 + 4.0), atol=1e-12, err_msg=(
        "0.5 * weight_decay * (sum of W**2 over 2-D params). Biases are exempt; if the 100s got in, you decayed them."
    ))
    assert boss.l2_penalty(params, 0.0) == 0.0


def test_phase_1_l2_penalty_gradient_is_the_decay_term_sgd_adds():
    W = np.array([[0.5, -1.5], [2.0, 0.25]])
    wd = 0.3
    numeric = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        old = W[idx]
        W[idx] = old + 1e-6
        fp = boss.l2_penalty({"W": W}, wd)
        W[idx] = old - 1e-6
        fm = boss.l2_penalty({"W": W}, wd)
        W[idx] = old
        numeric[idx] = (fp - fm) / 2e-6
    np.testing.assert_allclose(numeric, wd * W, atol=1e-6, err_msg=(
        "d penalty / dW should be weight_decay * W. That is why the penalty carries a 0.5."
    ))
    W0 = W.copy()
    opt = bellows.SGD({"W": W}, lr=0.1, weight_decay=wd)
    opt.step({"W": np.zeros_like(W)})
    np.testing.assert_allclose(W, W0 - 0.1 * wd * W0, atol=1e-12, err_msg=(
        "with zero gradient, SGD's weight decay alone must shrink W by lr * weight_decay * W"
    ))


def test_phase_1_weight_decay_never_touches_biases():
    W, b = np.ones((2, 2)), np.ones(2)
    opt = bellows.SGD({"W": W, "b": b}, lr=0.1, weight_decay=1.0)
    opt.step({"W": np.zeros((2, 2)), "b": np.zeros(2)})
    np.testing.assert_allclose(W, 0.9, atol=1e-12)
    np.testing.assert_allclose(b, 1.0, atol=1e-12, err_msg="biases must not be decayed (1-D parameters are exempt)")


# ------------------------------------------------------------ phase 1: Dropout
def test_phase_1_dropout_in_eval_mode_is_the_identity():
    drop = boss.Dropout(0.5, np.random.default_rng(0))
    drop.training = False
    x = np.random.default_rng(1).standard_normal((50, 20))
    out = drop.forward(x)
    np.testing.assert_array_equal(out, x, err_msg="eval mode: return x unchanged (inverted dropout needs no test-time scaling)")
    dout = np.random.default_rng(2).standard_normal((50, 20))
    np.testing.assert_array_equal(drop.backward(dout), dout, err_msg="eval mode backward is the identity too")


def test_phase_1_dropout_zeroes_about_p_and_scales_the_survivors_by_one_over_one_minus_p():
    p = 0.3
    drop = boss.Dropout(p, np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((400, 250)) + 5.0  # no zeros of its own
    out = drop.forward(x)
    assert out.shape == x.shape
    dropped = np.mean(out == 0.0)
    assert abs(dropped - p) < 0.02, f"about {p:.0%} of units should be zeroed, you zeroed {dropped:.1%}"
    kept = out != 0.0
    np.testing.assert_allclose(out[kept], x[kept] / (1 - p), atol=1e-12, err_msg=(
        "survivors must be scaled by exactly 1 / (1 - p) so the expected output equals the input"
    ))
    assert abs(out.mean() - x.mean()) < 0.05 * abs(x.mean()), "the expected value of the output must match the input"


def test_phase_1_dropout_draws_a_fresh_mask_every_forward():
    drop = boss.Dropout(0.5, np.random.default_rng(0))
    x = np.ones((30, 30))
    a = drop.forward(x)
    b = drop.forward(x)
    assert not np.array_equal(a, b), "each forward pass must draw a new mask from rng"


def test_phase_1_dropout_backward_uses_the_same_mask_as_forward():
    p = 0.4
    drop = boss.Dropout(p, np.random.default_rng(3))
    x = np.random.default_rng(4).standard_normal((60, 40)) + 5.0
    out = drop.forward(x)
    dout = np.random.default_rng(5).standard_normal((60, 40))
    dx = drop.backward(dout)
    killed = out == 0.0
    assert np.all(dx[killed] == 0.0), "a dropped unit contributed nothing forward, so it receives no gradient"
    np.testing.assert_allclose(dx[~killed], dout[~killed] / (1 - p), atol=1e-12, err_msg=(
        "kept units pass dout scaled by 1 / (1 - p), the same factor forward used. Cache the scaled mask."
    ))


def test_phase_1_dropout_with_p_zero_changes_nothing():
    drop = boss.Dropout(0.0, np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((10, 10))
    np.testing.assert_array_equal(drop.forward(x), x)


def test_phase_1_sequential_eval_silences_dropout_inside_a_model():
    rng = np.random.default_rng(6)
    lin1, lin2 = anvil.Linear(2, 8, rng), anvil.Linear(8, 3, rng)
    model = bellows.Sequential([lin1, anvil.ReLU(), boss.Dropout(0.5, rng), lin2])
    x = rng.standard_normal((20, 2))
    model.eval()
    a, b = model.forward(x), model.forward(x)
    np.testing.assert_array_equal(a, b, err_msg="in eval mode the model must be deterministic")
    np.testing.assert_allclose(a, lin2.forward(anvil.ReLU().forward(lin1.forward(x))), atol=1e-12)
    model.train()
    assert not np.array_equal(model.forward(x), a), "in train mode dropout must be active again"


# ------------------------------------------------------------ phase 2
def test_phase_2_count_heads_counts_learned_lies_only():
    y_clean = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_noisy = np.array([0, 2, 2, 1, 1, 0, 0, 1])  # flipped at 1, 3, 5
    pred = np.array([0, 2, 0, 0, 1, 0, 2, 1])  # agrees with the lie at 1 and 5; refuses it at 3; wrong elsewhere too
    oracle = Oracle(pred)
    heads = boss.count_heads(oracle, np.zeros((8, 2)), y_noisy, y_clean)
    assert isinstance(heads, int) and heads == 2, (
        f"heads = flipped AND predicted == noisy label: indices 1 and 5 -> 2, got {heads}. "
        "Index 3 is a flipped label the model refused (not a head); indices 2 and 6 are ordinary mistakes."
    )
    assert oracle.modes == [False], "predict in eval mode"
    assert oracle.training is True, "restore the model's mode"


def test_phase_2_a_perfect_refusal_has_no_heads():
    y_clean = np.array([0, 1, 2, 0])
    y_noisy = np.array([1, 1, 0, 0])
    assert boss.count_heads(Oracle(y_clean), np.zeros((4, 2)), y_noisy, y_clean) == 0


# ------------------------------------------------------------ phase 3
def test_phase_3_the_wild_hydra_grows_a_head_for_every_lie_it_can_swallow():
    X_train, y_noisy, y_clean, X_val, y_val = hydra_arena()
    rng = np.random.default_rng(0)
    model = he_mlp([2, 128, 128, 3], rng)
    opt = bellows.SGD(model.params(), lr=0.5)  # no weight decay, no dropout, no mirror
    bellows.train(model, anvil.SoftmaxCrossEntropy(), opt, X_train, y_noisy, epochs=1500, batch_size=64, rng=rng)
    heads = boss.count_heads(model, X_train, y_noisy, y_clean)
    train_acc = bellows.accuracy(model, X_train, y_noisy)
    val_acc = bellows.accuracy(model, X_val, y_val)
    assert heads >= WILD_HEADS, (
        f"An unregularized [2, 128, 128, 3] net trained for 1500 epochs should memorize at least {WILD_HEADS} of the "
        f"{LIES} lies; it learned {heads} (train acc on the noisy labels {train_acc:.2f}, val acc {val_acc:.2f}). "
        "If this fails your training loop is not fitting the data: check Room 3."
    )


def test_phase_3_slay_the_hydra():
    X_train, y_noisy, y_clean, X_val, y_val = hydra_arena()
    model = boss.slay_the_hydra(X_train, y_noisy, X_val, y_val, np.random.default_rng(0))
    val_acc = bellows.accuracy(model, X_val, y_val)
    train_acc = bellows.accuracy(model, X_train, y_noisy)
    heads = boss.count_heads(model, X_train, y_noisy, y_clean)
    assert val_acc >= 0.80, (
        f"validation accuracy {val_acc:.3f} < 0.80: the blade is dull. You regularized so hard the model never learned "
        "the spiral, or you stopped too early. Ease off weight decay / dropout, give it more epochs, watch the mirror."
    )
    assert heads <= SLAIN_HEADS, (
        f"the Hydra still has {heads} heads (of {LIES} lies); at most {SLAIN_HEADS} may remain. The model is memorizing "
        "flipped labels: add weight decay (1e-3 is a good start), stop early on validation loss, or shrink the network."
    )
    assert train_acc - val_acc <= 0.20, (
        f"train accuracy {train_acc:.3f} vs val {val_acc:.3f}: a gap of {train_acc - val_acc:.3f} means the model is "
        "fitting the training set far better than the task. Regularize."
    )
