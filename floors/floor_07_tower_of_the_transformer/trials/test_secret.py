"""SECRET - THE ACCUMULATED VERSE

Gradient accumulation must be *exactly* equivalent to a larger batch: the
accumulated gradient of k micro-batches equals the gradient of one k-times
larger batch, and the optimizer steps once per window, not once per verse.
"""

import copy

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

verse = load_room(__file__, "secret_accumulated_verse")

pytestmark = pytest.mark.secret

CFG = tiny_gpt.GPTConfig(vocab_size=17, block_size=16, n_layer=1, n_head=2, n_embd=32)


class RecordingSGD(torch.optim.SGD):
    """Counts its steps and snapshots the gradients it steps on."""

    def __init__(self, params, lr=0.1):
        super().__init__(params, lr=lr)
        self.steps = 0
        self.grad_snapshots: list[list[torch.Tensor]] = []

    def step(self, *args, **kwargs):
        self.steps += 1
        self.grad_snapshots.append([p.grad.detach().clone() if p.grad is not None else None for g in self.param_groups for p in g["params"]])
        return super().step(*args, **kwargs)


def _model(seed=0):
    torch.manual_seed(seed)
    return tiny_gpt.GPT(CFG)


def _micro_batches(k, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 17, (batch_size, 16), generator=g), torch.randint(0, 17, (batch_size, 16), generator=g)) for _ in range(k)]


def test_accumulated_gradients_equal_the_big_batch_gradient():
    batches = _micro_batches(4)
    small = _model()
    big = copy.deepcopy(small)
    opt = RecordingSGD(small.parameters())
    verse.train_step_accumulated(small, batches, opt, accum_steps=4)
    assert opt.steps == 1, f"Four micro-batches with accum_steps=4 make ONE optimizer step; you made {opt.steps}."
    accumulated = opt.grad_snapshots[0]

    x = torch.cat([b[0] for b in batches])
    y = torch.cat([b[1] for b in batches])
    big.train()
    _, loss = big(x, y)
    loss.backward()
    for (name, p), got in zip(big.named_parameters(), accumulated):
        assert got is not None, f"{name} had no gradient at step time. Did you call backward on every micro-batch?"
        diff = (got - p.grad).abs().max().item()
        scale = p.grad.abs().max().item() + 1e-12
        assert diff < 1e-5, (
            f"Gradient of {name} differs from the big-batch gradient by {diff:.2e}. "
            + (f"It looks about {got.abs().max().item() / scale:.1f}x too large: scale each micro-batch loss by 1/accum_steps before backward()."
               if got.abs().max().item() / scale > 1.5 else
               "Do not zero the gradients between micro-batches: only after optimizer.step().")
        )


def test_the_optimizer_steps_once_per_window():
    model = _model()
    opt = RecordingSGD(model.parameters())
    verse.train_step_accumulated(model, _micro_batches(6), opt, accum_steps=3)
    assert opt.steps == 2, f"Six micro-batches in windows of three make exactly 2 optimizer steps; you made {opt.steps}."
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters()), (
        "After the last window the gradients must be cleared (zero_grad), ready for the next call."
    )


def test_the_second_window_does_not_inherit_the_first():
    batches = _micro_batches(4, seed=3)
    model = _model()
    opt = RecordingSGD(model.parameters(), lr=0.0)  # lr 0: the weights do not move, so window 2 is comparable
    verse.train_step_accumulated(model, batches, opt, accum_steps=2)
    assert opt.steps == 2
    reference = _model()
    reference.train()
    x = torch.cat([batches[2][0], batches[3][0]])
    y = torch.cat([batches[2][1], batches[3][1]])
    _, loss = reference(x, y)
    loss.backward()
    for (name, p), got in zip(reference.named_parameters(), opt.grad_snapshots[1]):
        diff = (got - p.grad).abs().max().item()
        assert diff < 1e-5, (
            f"The second window's gradient of {name} is off by {diff:.2e}: gradients from the first window leaked in. "
            "zero_grad after every optimizer.step()."
        )


def test_the_returned_loss_is_the_mean_over_micro_batches_unscaled():
    batches = _micro_batches(3, seed=5)
    model = _model()
    with torch.no_grad():
        want = sum(model(x, y)[1].item() for x, y in batches) / 3
    got = verse.train_step_accumulated(model, batches, RecordingSGD(model.parameters(), lr=0.0), accum_steps=3)
    assert isinstance(got, float), f"Return a Python float, got {type(got).__name__}."
    assert abs(got - want) < 1e-5, f"The returned loss should be the mean of the UNSCALED micro-batch losses ({want:.5f}); got {got:.5f}."


def test_the_weights_actually_move():
    model = _model()
    before = [p.detach().clone() for p in model.parameters()]
    verse.train_step_accumulated(model, _micro_batches(2), RecordingSGD(model.parameters(), lr=0.5), accum_steps=2)
    moved = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert moved, "No parameter changed. optimizer.step() must be called once the window is full."


def test_uneven_windows_are_refused():
    model = _model()
    with pytest.raises(ValueError):
        verse.train_step_accumulated(model, _micro_batches(5), RecordingSGD(model.parameters()), accum_steps=2)


def test_count_tokens_seen_counts_targets():
    batches = _micro_batches(3, batch_size=4)
    assert verse.count_tokens_seen(batches) == 3 * 4 * 16, f"3 micro-batches of (4, 16) targets are 192 tokens; got {verse.count_tokens_seen(batches)}."
    ragged = [(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long)), (torch.zeros(1, 7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long))]
    assert verse.count_tokens_seen(ragged) == 17
    assert isinstance(verse.count_tokens_seen(ragged), int)
