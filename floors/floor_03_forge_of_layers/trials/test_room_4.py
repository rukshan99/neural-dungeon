"""TRIAL 3.4 - THE MIRROR OF VALIDATION

Early stopping must pick the argmin and wait out the patience; diagnose must
follow the stated rule; and a deliberately overfitting setup must show its
face in the mirror.
"""

import numpy as np
import pytest

from dungeon.artifacts.toydata import make_spirals
from dungeon.trials import load_room

anvil = load_room(__file__, "room_1_the_anvil")
bellows = load_room(__file__, "room_3_the_bellows")
room = load_room(__file__, "room_4_mirror_of_validation")

MIRROR_CURVES = {
    "the_eager_apprentice": (
        [1.10, 0.80, 0.55, 0.35, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03],
        [1.08, 0.85, 0.70, 0.62, 0.60, 0.61, 0.66, 0.72, 0.80, 0.88],
    ),
    "the_blunt_hammer": (
        [1.10, 1.05, 1.02, 1.00, 0.99, 0.98, 0.98, 0.97, 0.97, 0.97],
        [1.11, 1.06, 1.03, 1.01, 1.00, 0.99, 0.99, 0.98, 0.98, 0.98],
    ),
    "the_tempered_blade": (
        [1.10, 0.75, 0.52, 0.40, 0.33, 0.29, 0.26, 0.24, 0.23, 0.22],
        [1.09, 0.78, 0.56, 0.45, 0.39, 0.35, 0.33, 0.32, 0.31, 0.31],
    ),
}


def ref_diagnose(train, val, rise=0.10, floor=0.5):
    train, val = np.asarray(train, float), np.asarray(val, float)
    best = int(np.argmin(val))
    if val[-1] > (1 + rise) * val[best] and train[-1] < train[best]:
        return "overfitting"
    if train[-1] > floor * train[0]:
        return "underfitting"
    return "healthy"


def he_mlp(sizes, rng):
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = anvil.Linear(fan_in, fan_out, rng)
        lin.params["W"][...] = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(anvil.ReLU())
    return bellows.Sequential(layers)


def overfitting_arena():
    """45 noisy training points, 150 cleaner validation points: a recipe for memorization."""
    X_train, y_train = make_spirals(n_per_class=15, n_classes=3, noise=0.4, seed=3)
    X_val, y_val = make_spirals(n_per_class=50, n_classes=3, noise=0.1, seed=4)
    return X_train.astype(np.float64), y_train, X_val.astype(np.float64), y_val


class Oracle:
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


# ------------------------------------------------------------ early_stopping
def test_early_stopping_finds_the_argmin_and_waits_out_the_patience():
    val = [1.0, 0.9, 0.95, 0.97]
    assert room.early_stopping(val, patience=2) == (1, True), "best epoch is 1; two epochs have passed without a new low -> stop"
    assert room.early_stopping(val, patience=3) == (1, False), "best epoch is 1; only two epochs have passed, patience is 3 -> keep going"
    assert room.early_stopping([1.0, 0.8, 0.6, 0.5], patience=1) == (3, False), "still improving: the last epoch is the best, never stop"
    assert room.early_stopping([0.5], patience=1) == (0, False), "one epoch is never enough to stop"


def test_early_stopping_breaks_ties_toward_the_earliest_low():
    best, stop = room.early_stopping([0.9, 0.5, 0.5, 0.5], patience=2)
    assert best == 1, f"on a tie the FIRST minimum wins (np.argmin does this), got {best}"
    assert stop is True or stop == np.True_


def test_early_stopping_returns_a_plain_int_and_a_bool():
    best, stop = room.early_stopping([0.3, 0.2, 0.4], patience=1)
    assert isinstance(best, int) and int(best) == 1, "return the index as a Python int"
    assert bool(stop) is True


# ------------------------------------------------------------ diagnose
def test_diagnose_reads_a_textbook_overfit():
    train = np.linspace(1.1, 0.05, 20)
    val = np.concatenate([np.linspace(1.1, 0.5, 10), np.linspace(0.52, 0.9, 10)])
    assert room.diagnose(list(train), list(val)) == "overfitting", (
        "val climbed 80% off its minimum while train kept falling: that is overfitting"
    )


def test_diagnose_reads_a_model_that_never_learned():
    train = np.linspace(1.1, 0.95, 20)
    val = np.linspace(1.1, 0.97, 20)
    assert room.diagnose(list(train), list(val)) == "underfitting", (
        "train loss went from 1.10 to 0.95, nowhere near halving: that is underfitting"
    )


def test_diagnose_blesses_a_healthy_run():
    train = np.linspace(1.1, 0.2, 20)
    val = np.linspace(1.1, 0.3, 20)
    assert room.diagnose(list(train), list(val)) == "healthy"


def test_diagnose_tolerates_a_small_wobble_in_val():
    train = np.linspace(1.1, 0.2, 20)
    val = np.concatenate([np.linspace(1.1, 0.30, 15), np.full(5, 0.31)])
    assert room.diagnose(list(train), list(val)) == "healthy", (
        "val is only 3% above its minimum, well under OVERFIT_RISE (10%): not overfitting"
    )
    val_flat = np.concatenate([np.linspace(1.1, 0.30, 15), np.full(5, 0.34)])
    assert room.diagnose(list(train), list(val_flat)) == "overfitting", (
        "val is 13% above its minimum and train kept falling: overfitting. Use OVERFIT_RISE as the boundary."
    )


# ------------------------------------------------------------ evaluate_loss
def test_evaluate_loss_looks_in_eval_mode_and_leaves_the_model_as_it_found_it():
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    y = np.array([0, 1])
    oracle = Oracle(logits)
    value = room.evaluate_loss(oracle, anvil.SoftmaxCrossEntropy(), np.zeros((2, 2)), y)
    assert isinstance(value, float)
    np.testing.assert_allclose(value, anvil.SoftmaxCrossEntropy().forward(logits, y), atol=1e-12)
    assert oracle.modes == [False], "evaluate in eval mode"
    assert oracle.training is True, "restore the previous mode afterwards"


# ------------------------------------------------------------ train_with_validation
def test_the_mirror_records_one_pair_of_losses_per_epoch_and_shows_the_final_model():
    X_train, y_train, X_val, y_val = overfitting_arena()
    rng = np.random.default_rng(0)
    model = he_mlp([2, 16, 3], rng)
    loss = anvil.SoftmaxCrossEntropy()
    tr, va = room.train_with_validation(model, loss, bellows.SGD(model.params(), 0.1), X_train, y_train, X_val, y_val, 5, 8, rng)
    assert len(tr) == 5 and len(va) == 5, f"expected 5 train and 5 val losses, got {len(tr)} and {len(va)}"
    np.testing.assert_allclose(va[-1], room.evaluate_loss(model, loss, X_val, y_val), atol=1e-12, err_msg=(
        "the last val loss must describe the model as it is at the end of the last epoch (validate AFTER the epoch's updates)"
    ))


def test_the_mirror_does_not_touch_the_metal():
    """Training with the mirror held up must produce exactly the training curve of Room 3's train()."""
    X_train, y_train, X_val, y_val = overfitting_arena()
    plain = he_mlp([2, 16, 3], np.random.default_rng(9))
    mirrored = he_mlp([2, 16, 3], np.random.default_rng(9))
    ref = bellows.train(plain, anvil.SoftmaxCrossEntropy(), bellows.SGD(plain.params(), 0.1), X_train, y_train, 6, 8, np.random.default_rng(1))
    tr, _ = room.train_with_validation(mirrored, anvil.SoftmaxCrossEntropy(), bellows.SGD(mirrored.params(), 0.1), X_train, y_train, X_val, y_val, 6, 8, np.random.default_rng(1))
    np.testing.assert_allclose(tr, ref, atol=1e-12, err_msg=(
        "train losses differ from train()'s: validation must not consume the rng, update parameters, or change the batches. "
        "Call Room 3's train() for one epoch at a time."
    ))


def test_patience_stops_the_run_and_hands_back_the_best_epoch():
    X_train, y_train, X_val, y_val = overfitting_arena()
    rng = np.random.default_rng(0)
    model = he_mlp([2, 64, 64, 3], rng)
    loss = anvil.SoftmaxCrossEntropy()
    tr, va = room.train_with_validation(model, loss, bellows.SGD(model.params(), 0.2), X_train, y_train, X_val, y_val, 200, 8, rng, patience=10)
    assert len(va) < 200, "on this arena val loss bottoms out early; with patience=10 the run must stop well before 200 epochs"
    assert len(tr) == len(va)
    since_best = len(va) - 1 - int(np.argmin(va))
    assert since_best >= 10, f"the run stopped {since_best} epochs after the best; patience was 10"
    np.testing.assert_allclose(room.evaluate_loss(model, loss, X_val, y_val), min(va), rtol=1e-9, err_msg=(
        "the returned model must carry the parameters of the BEST epoch, not the last one: snapshot copies when val improves and restore them"
    ))


# ------------------------------------------------------------ the deliberate overfit
def test_a_big_net_on_tiny_noisy_data_climbs_in_the_mirror():
    X_train, y_train, X_val, y_val = overfitting_arena()
    rng = np.random.default_rng(0)
    model = he_mlp([2, 64, 64, 3], rng)
    tr, va = room.train_with_validation(model, anvil.SoftmaxCrossEntropy(), bellows.SGD(model.params(), 0.2), X_train, y_train, X_val, y_val, 200, 8, rng)
    best = int(np.argmin(va))
    assert best < 100, f"val loss should bottom out early (found its minimum at epoch {best})"
    assert va[-1] > 1.5 * va[best], (
        f"val loss should climb far above its minimum: min {va[best]:.3f} at epoch {best}, end {va[-1]:.3f}. "
        "If it does not, check that validation uses the model in eval mode and never trains on X_val."
    )
    assert tr[-1] < tr[0] * 0.5, f"meanwhile the train loss should keep falling ({tr[0]:.3f} -> {tr[-1]:.3f})"
    verdict = room.diagnose(tr, va)
    assert verdict == "overfitting", f"diagnose() should call this run overfitting, it said {verdict!r}"


# ------------------------------------------------------------ the prophecy
def test_the_curves_on_the_frame_are_untouched():
    assert set(room.MIRROR_CURVES) == set(MIRROR_CURVES), "do not add or remove curves"
    for name, (tr, va) in MIRROR_CURVES.items():
        assert list(room.MIRROR_CURVES[name][0]) == tr and list(room.MIRROR_CURVES[name][1]) == va, f"{name}: the curves were edited"


@pytest.mark.parametrize("name", list(MIRROR_CURVES), ids=list(MIRROR_CURVES))
def test_the_prophecy_of_the_mirror(name):
    prediction = room.MIRROR_PROPHECY.get(name)
    if prediction is None:
        raise NotImplementedError(f"You have not diagnosed {name!r}. Fill in MIRROR_PROPHECY.")
    tr, va = MIRROR_CURVES[name]
    truth = ref_diagnose(tr, va)
    assert prediction == truth, (
        f"You called {name!r} {prediction!r}. By the rule it is {truth!r}: val goes {va[0]:.2f} -> min {min(va):.2f} -> {va[-1]:.2f}, "
        f"train goes {tr[0]:.2f} -> {tr[-1]:.2f}."
    )
    assert room.diagnose(tr, va) == truth, f"your diagnose() disagrees with the rule on {name!r}"
