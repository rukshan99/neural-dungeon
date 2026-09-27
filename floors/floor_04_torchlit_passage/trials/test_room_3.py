"""TRIAL 4.3 - THE LOADER'S LANTERN

The dataset is measured, the loader's batches are weighed, its shuffle is
replayed, and one honest epoch of training and evaluation is run on the runes.
"""

import numpy as np
import pytest

from dungeon.artifacts.toydata import make_runes, train_val_split
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_3_loaders_lantern")

nn = torch.nn

X_SMALL, Y_SMALL = make_runes(n_per_class=5, seed=3)  # 50 tablets


def _labels_in_order(loader):
    return [int(v) for _, yb in loader for v in yb]


class ModeSpy(nn.Module):
    """A model that writes down the mode and the autograd state every time it is called."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 10)
        self.seen_training = []
        self.seen_grad_enabled = []

    def forward(self, x):
        self.seen_training.append(self.training)
        self.seen_grad_enabled.append(torch.is_grad_enabled())
        return self.lin(x.flatten(1))


class AlwaysThree(nn.Module):
    """Predicts class 3 for every rune, with great confidence and no parameters worth training."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Linear(1, 1)

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 10)
        logits[:, 3] = 1.0
        return logits


# ------------------------------------------------------------------ dataset
def test_the_dataset_knows_its_length_and_speaks_float32_int64():
    ds = room.RuneDataset(X_SMALL, Y_SMALL)
    assert len(ds) == 50, f"50 tablets went in; len() says {len(ds)}."
    item = ds[7]
    assert isinstance(item, tuple) and len(item) == 2, "__getitem__ returns a tuple (image, label)."
    x, y = item
    assert torch.is_tensor(x) and x.dtype == torch.float32 and tuple(x.shape) == (8, 8), (
        f"The image should be a float32 tensor of shape (8, 8); got {type(x).__name__} {getattr(x, 'dtype', '')} {tuple(getattr(x, 'shape', ()))}."
    )
    if torch.is_tensor(y):
        assert y.dtype == torch.int64 and y.ndim == 0, f"The label should be a 0-d int64 tensor; got dtype {y.dtype}, shape {tuple(y.shape)}."
    else:
        assert isinstance(y, int), f"The label should be an int64 tensor or a Python int, not {type(y).__name__}."
    assert int(y) == int(Y_SMALL[7]) and np.allclose(x.numpy(), X_SMALL[7]), "ds[i] must be the i-th example."


def test_the_dataset_cures_float64_at_the_border():
    ds = room.RuneDataset(X_SMALL.astype(np.float64), Y_SMALL.astype(np.int32))
    x, y = ds[0]
    assert x.dtype == torch.float32, (
        f"Fed float64 numpy, the dataset handed out {x.dtype}. numpy defaults to float64; nn.Linear is float32. "
        "Convert at the border: torch.as_tensor(X, dtype=torch.float32)."
    )
    assert int(y) == int(Y_SMALL[0])
    if torch.is_tensor(y):
        assert y.dtype == torch.int64, f"Labels must be int64 for CrossEntropyLoss, not {y.dtype}."


# ------------------------------------------------------------------ loader
def test_batches_have_the_right_shapes_and_dtypes():
    loader = room.make_loader(room.RuneDataset(X_SMALL, Y_SMALL), batch_size=16, shuffle=False, seed=0)
    xb, yb = next(iter(loader))
    assert tuple(xb.shape) == (16, 8, 8) and xb.dtype == torch.float32, f"Batch images: expected (16, 8, 8) float32, got {tuple(xb.shape)} {xb.dtype}."
    assert tuple(yb.shape) == (16,) and yb.dtype == torch.int64, f"Batch labels: expected (16,) int64, got {tuple(yb.shape)} {yb.dtype}."


def test_the_last_batch_is_smaller_not_missing():
    loader = room.make_loader(room.RuneDataset(X_SMALL, Y_SMALL), batch_size=16, shuffle=True, seed=0)
    sizes = [len(yb) for _, yb in loader]
    assert sizes == [16, 16, 16, 2], (
        f"50 tablets in batches of 16 should come out as [16, 16, 16, 2]; you got {sizes}. "
        "Leave drop_last=False; the two leftover tablets are still data."
    )


def test_the_same_seed_walks_the_same_order():
    ds = room.RuneDataset(X_SMALL, Y_SMALL)
    a = _labels_in_order(room.make_loader(ds, 16, shuffle=True, seed=7))
    b = _labels_in_order(room.make_loader(ds, 16, shuffle=True, seed=7))
    assert sorted(a) == sorted(Y_SMALL.tolist()), "Every tablet should appear exactly once per epoch."
    assert a == b, "Two loaders built with the same seed must walk the same order. Pass generator=torch.Generator().manual_seed(seed)."
    assert a != Y_SMALL.tolist(), "shuffle=True, yet the tablets came out in dataset order."


def test_a_different_seed_walks_a_different_order():
    ds = room.RuneDataset(X_SMALL, Y_SMALL)
    a = _labels_in_order(room.make_loader(ds, 16, shuffle=True, seed=7))
    b = _labels_in_order(room.make_loader(ds, 16, shuffle=True, seed=8))
    assert a != b, "Different seeds gave the same order. Is the seed actually reaching the generator?"


def test_shuffle_false_keeps_the_tablets_in_order():
    loader = room.make_loader(room.RuneDataset(X_SMALL, Y_SMALL), 16, shuffle=False, seed=7)
    assert _labels_in_order(loader) == Y_SMALL.tolist(), "shuffle=False must yield the dataset in index order."


def test_the_second_epoch_has_a_new_order_but_the_same_story():
    ds = room.RuneDataset(X_SMALL, Y_SMALL)
    loader = room.make_loader(ds, 16, shuffle=True, seed=3)
    epoch_1, epoch_2 = _labels_in_order(loader), _labels_in_order(loader)
    assert epoch_1 != epoch_2, "The generator advances: epoch 2 should not repeat epoch 1's order."
    twin = room.make_loader(ds, 16, shuffle=True, seed=3)
    assert [_labels_in_order(twin), _labels_in_order(twin)] == [epoch_1, epoch_2], (
        "A twin loader with the same seed should replay both epochs identically."
    )


def test_the_shuffle_does_not_lean_on_the_global_die():
    ds = room.RuneDataset(X_SMALL, Y_SMALL)
    state_before = torch.get_rng_state()
    loader = room.make_loader(ds, 16, shuffle=True, seed=5)
    assert torch.equal(torch.get_rng_state(), state_before), (
        "Building the loader changed the global torch RNG state. Do not torch.manual_seed() inside make_loader; "
        "hand the loader its own torch.Generator()."
    )
    torch.rand(100)  # the rest of the program draws from the global RNG before the epoch starts...
    a = _labels_in_order(loader)
    twin = room.make_loader(ds, 16, shuffle=True, seed=5)
    torch.rand(3)  # ...and a different amount before the twin's epoch
    b = _labels_in_order(twin)
    assert a == b, (
        "Two loaders with the same seed walked different orders once the global RNG was touched in between. "
        "Without its own generator a DataLoader seeds each epoch from the global RNG, so the shuffle depends on "
        "everything else the program did. Pass generator=torch.Generator().manual_seed(seed)."
    )


# ------------------------------------------------------------------ training and evaluation
def _runes():
    X, y = make_runes(n_per_class=100, seed=0)
    return train_val_split(X, y, val_fraction=0.2, seed=0)


def _mean_loss(model, loader, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            total += loss_fn(model(xb), yb).item()
    return total / len(loader)


def test_train_one_epoch_returns_a_float_and_lowers_the_loss():
    torch.manual_seed(0)
    X_train, y_train, _, _ = _runes()
    loader = room.make_loader(room.RuneDataset(X_train, y_train), 32, shuffle=True, seed=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    loss_fn = nn.CrossEntropyLoss()
    before = _mean_loss(model, loader, loss_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    first = room.train_one_epoch(model, loader, optimizer, loss_fn)
    assert isinstance(first, float), (
        f"train_one_epoch returned {type(first).__name__}; return a Python float. Accumulate loss.item(), not the loss tensor."
    )
    assert first < before - 0.1, f"The mean loss over the first epoch was {first:.3f}; before training it was {before:.3f}. Nothing learned: check zero_grad, backward, step."
    second = room.train_one_epoch(model, loader, optimizer, loss_fn)
    assert second < first, f"Epoch 2 ({second:.3f}) should be lower than epoch 1 ({first:.3f})."


def test_train_one_epoch_wakes_the_model_up():
    spy = ModeSpy()
    spy.eval()
    loader = room.make_loader(room.RuneDataset(X_SMALL, Y_SMALL), 16, shuffle=False, seed=0)
    room.train_one_epoch(spy, loader, torch.optim.SGD(spy.parameters(), lr=0.01), nn.CrossEntropyLoss())
    assert spy.seen_training and all(spy.seen_training), (
        "The model was in eval mode during training. Call model.train() first: Dropout and BatchNorm depend on it."
    )


def test_evaluate_measures_in_eval_mode_without_a_graph():
    spy = ModeSpy()
    spy.train()
    loader = room.make_loader(room.RuneDataset(X_SMALL, Y_SMALL), 16, shuffle=False, seed=0)
    acc = room.evaluate(spy, loader)
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0, f"evaluate returns a float in [0, 1], got {acc!r}."
    assert spy.seen_training and not any(spy.seen_training), (
        "evaluate() ran the model in training mode: dropout would be active while you measure. model.eval() first."
    )
    assert not any(spy.seen_grad_enabled), (
        "evaluate() ran with autograd recording. Wrap it in torch.no_grad(): a graph you never backward through is wasted memory."
    )


def test_evaluate_counts_examples_not_batches():
    y = np.zeros(50, dtype=np.int64)
    y[:12] = 3  # 12 of the first 48 are threes ...
    y[48:] = 3  # ... and both leftovers are, so the per-batch mean would be 0.4375, not 0.28
    loader = room.make_loader(room.RuneDataset(X_SMALL, y), 16, shuffle=False, seed=0)
    acc = room.evaluate(AlwaysThree(), loader)
    assert acc == pytest.approx(14 / 50), (
        f"A model that always says 3 is right on 14 of 50 tablets = 0.28; you reported {acc:.4f}. "
        "Count correct examples over all batches and divide by the number of examples, not by the number of batches."
    )


def test_two_epochs_on_the_runes_reach_a_sane_accuracy():
    torch.manual_seed(1)
    X_train, y_train, X_val, y_val = _runes()
    train_loader = room.make_loader(room.RuneDataset(X_train, y_train), 32, shuffle=True, seed=1)
    val_loader = room.make_loader(room.RuneDataset(X_val, y_val), 64, shuffle=False, seed=1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(2):
        room.train_one_epoch(model, train_loader, optimizer, loss_fn)
    acc = room.evaluate(model, val_loader)
    assert acc >= 0.85, f"Two epochs on the runes should give at least 85% validation accuracy; you have {acc:.1%}."
