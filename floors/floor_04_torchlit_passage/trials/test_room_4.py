"""TRIAL 4.4 - THE CURSED TRAINING LOOP

Seven curses, seven symptoms. Each test below names one symptom and points at
the kind of line that causes it. The last test trains the whole loop and
demands that it learn.
"""

import math

import numpy as np
import pytest

from dungeon.scrutiny import names_called_in
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

room = load_room(__file__, "room_4_cursed_training_loop")

nn = torch.nn
F = torch.nn.functional


class ModeSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 10)
        self.seen_training = []
        self.seen_grad_enabled = []

    def forward(self, x):
        self.seen_training.append(self.training)
        self.seen_grad_enabled.append(torch.is_grad_enabled())
        return self.lin(x.flatten(1))


def _two_batches(seed=0, batch=16):
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(2, batch, 8, 8, generator=g)
    ys = torch.randint(0, 10, (2, batch), generator=g)
    return [(xs[0], ys[0]), (xs[1], ys[1])]


# ------------------------------------------------------------------ curse: dtype
def test_labels_are_class_indices_not_floats():
    x = np.random.default_rng(0).random((4, 8, 8), dtype=np.float32)
    y = np.array([0, 3, 9, 1], dtype=np.int64)
    xb, yb = room.prepare_batch(x, y)
    assert xb.dtype == torch.float32, f"Images should be float32, got {xb.dtype}."
    assert yb.dtype == torch.int64, (
        f"Labels came out as {yb.dtype}. CrossEntropyLoss wants class INDICES as int64 (a LongTensor): "
        "'expected target dtype to be Long'. Do not cast labels to float."
    )
    xb, yb = room.prepare_batch(torch.from_numpy(x), torch.from_numpy(y))
    assert yb.dtype == torch.int64 and torch.equal(yb, torch.tensor([0, 3, 9, 1]))


# ------------------------------------------------------------------ curse: applied twice
def test_the_loss_is_fed_logits_not_probabilities():
    logits = torch.full((3, 10), -2.0)
    y = torch.tensor([0, 4, 7])
    logits[torch.arange(3), y] = 8.0  # very confident, very correct
    got = room.compute_loss(logits, y).item()
    expected = F.cross_entropy(logits, y).item()
    assert got == pytest.approx(expected, abs=1e-5), (
        f"On confidently correct logits the cross-entropy is {expected:.5f}; yours is {got:.4f}. "
        "CrossEntropyLoss applies log_softmax itself. Feeding it softmax output squashes every logit into [0, 1] "
        "and the loss can never get below about 1.46 no matter how right the model is."
    )
    torch.manual_seed(0)
    logits = torch.randn(16, 10)
    y = torch.randint(0, 10, (16,))
    assert room.compute_loss(logits, y).item() == pytest.approx(F.cross_entropy(logits, y).item(), abs=1e-5)


# ------------------------------------------------------------------ curse: axis
def test_accuracy_looks_across_classes_not_across_the_batch():
    y = torch.tensor([0, 3, 9, 3, 5, 1])
    logits = torch.full((6, 10), -1.0)
    logits[torch.arange(6), y] = 5.0
    try:
        acc = room.accuracy(logits, y)
    except RuntimeError as exc:
        pytest.fail(
            f"accuracy() crashed: {exc}\nargmax(dim=0) on (B, C) logits gives one winner per CLASS, shape (C,). "
            "You want one winner per EXAMPLE, shape (B,): argmax(dim=1)."
        )
    assert acc == pytest.approx(1.0), f"Every row's top logit is the true class, so accuracy is 1.0; you reported {acc}."
    torch.manual_seed(1)
    logits = torch.randn(10, 10)  # B == C: the wrong axis does not crash, it silently lies
    y = logits.argmax(dim=1)
    acc = room.accuracy(logits, y)
    assert acc == pytest.approx(1.0), (
        f"With B == C the wrong argmax axis raises no error and reports {acc:.2f} instead of 1.0. "
        "This is the version of the bug that ships."
    )
    assert isinstance(acc, float), "Return a Python float (.item())."


# ------------------------------------------------------------------ curse: scale
def test_the_learning_rate_does_not_overshoot_every_minimum_in_the_building():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = room.make_optimizer(model)
    lr = optimizer.param_groups[0]["lr"]
    (xb, yb), _ = _two_batches(seed=2, batch=32)
    first = last = None
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
        last = loss.item()
    assert math.isfinite(last) and last < first, (
        f"Thirty steps on ONE batch took the loss from {first:.3f} to {last:.3f} with lr={lr}. "
        "A step that overshoots the minimum makes the loss worse, not better. For SGD with momentum, 0.01-0.1 is the usual range."
    )


# ------------------------------------------------------------------ curse: never called
def test_gradients_are_not_haunted_by_the_previous_batch():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    loader = _two_batches(seed=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # weights never move; only .grad tells the story
    room.train_epoch(model, loader, optimizer)
    observed = None if model[1].weight.grad is None else model[1].weight.grad.detach().clone()
    if observed is None:
        return  # cleared after stepping: also fine

    def grad_of(batch):
        model.zero_grad()
        x, y = room.prepare_batch(*batch)
        room.compute_loss(model(x), y).backward()
        return model[1].weight.grad.detach().clone()

    g_first, g_last = grad_of(loader[0]), grad_of(loader[1])
    if torch.allclose(observed, g_first + g_last, atol=1e-6):
        pytest.fail(
            "After the epoch, .grad holds the SUM of both batches' gradients. .grad accumulates across backward() calls; "
            "call optimizer.zero_grad() before each loss.backward()."
        )
    assert torch.allclose(observed, g_last, atol=1e-6), "After the epoch, .grad should hold only the last batch's gradient."


# ------------------------------------------------------------------ curse: memory leak
def test_the_epoch_loss_is_a_number_not_a_graph():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    result = room.train_epoch(model, _two_batches(seed=4), torch.optim.SGD(model.parameters(), lr=0.01))
    assert not torch.is_tensor(result), (
        f"train_epoch returned a tensor (grad_fn={getattr(result, 'grad_fn', None)}). You summed loss TENSORS, "
        "which keeps every batch's graph alive until the epoch ends. Sum loss.item() instead."
    )
    assert isinstance(result, float) and math.isfinite(result), f"Return a finite Python float, got {result!r}."
    called = names_called_in(room.train_epoch)
    assert called & {"item", "detach", "float", "tolist"}, (
        "train_epoch never detaches the loss from its graph (.item()). Converting only at the end still holds every graph in memory."
    )


# ------------------------------------------------------------------ curse: mode never switched
def test_the_model_learns_with_dropout_on_and_is_judged_with_dropout_off():
    spy = ModeSpy()
    spy.eval()
    room.train_epoch(spy, _two_batches(seed=5), torch.optim.SGD(spy.parameters(), lr=0.01))
    assert spy.seen_training and all(spy.seen_training), (
        "train_epoch ran the model in eval mode: dropout off, BatchNorm frozen. Call model.train() before the loop."
    )
    spy = ModeSpy()
    spy.train()
    room.evaluate(spy, _two_batches(seed=6))
    assert spy.seen_training and not any(spy.seen_training), (
        "evaluate ran the model in TRAINING mode: dropout was randomly zeroing activations while you measured accuracy. "
        "Call model.eval() first."
    )
    assert not any(spy.seen_grad_enabled), "evaluate should run under torch.no_grad()."


# ------------------------------------------------------------------ the whole loop
def test_the_lifted_loop_learns_the_runes():
    result = room.train_cursed(epochs=3, seed=0)
    losses = result["train_loss"]
    assert all(isinstance(v, float) and math.isfinite(v) for v in losses), f"train_loss should be finite Python floats, got {losses!r}."
    assert losses[-1] < losses[0], f"The loss did not fall over three epochs: {[round(v, 3) for v in losses]}."
    assert result["val_acc"] >= 0.90, (
        f"Validation accuracy after 3 epochs is {result['val_acc']:.1%}; the lifted loop reaches at least 90%. "
        f"Losses: {[round(v, 3) for v in losses]}. A curse remains."
    )
