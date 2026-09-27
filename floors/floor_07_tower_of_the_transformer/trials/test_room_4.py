"""TRIAL 7.4 - THE TRAINING

Batches must be shifted by one, the schedule must warm up and cool down on
the exact steps, gradients must be clipped, and a small tower of your own
must learn the chronicles: from ln(72) to below LOSS_AFTER in STEPS steps.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_4_the_training")
tower = load_room(__file__, "room_3_the_tower")

torch.manual_seed(74)

_, TOKENIZER, _ = tiny_gpt.load_pretrained()
TEXT = tiny_gpt.read_corpus("chronicles")
DATA = torch.tensor(TOKENIZER.encode(TEXT), dtype=torch.long)

SMALL = tiny_gpt.GPTConfig(vocab_size=TOKENIZER.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64)
STEPS, BATCH, LR = 120, 32, 3e-3
LOSS_AFTER = 2.8  # the reference reaches about 2.4 in 120 steps; 2.8 leaves room for an unlucky seed


class SpyAdamW(torch.optim.AdamW):
    """An AdamW that writes down the learning rate and the total gradient norm at every step."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lrs: list[float] = []
        self.grad_norms: list[float] = []
        SpyAdamW.instances.append(self)

    def step(self, *args, **kwargs):
        self.lrs.append(float(self.param_groups[0]["lr"]))
        grads = [p.grad.flatten() for g in self.param_groups for p in g["params"] if p.grad is not None]
        self.grad_norms.append(torch.cat(grads).norm().item() if grads else 0.0)
        return super().step(*args, **kwargs)


@pytest.fixture
def spy(monkeypatch):
    SpyAdamW.instances = []
    monkeypatch.setattr(torch.optim, "AdamW", SpyAdamW)
    yield SpyAdamW
    SpyAdamW.instances = []


def _the_optimizer(spy):
    if not spy.instances:
        pytest.fail(
            "The trial could not see an optimizer. Build it as torch.optim.AdamW(model.parameters(), ...) inside "
            "train(), not with a from-import or a different optimizer, so the trial can inspect its steps."
        )
    return spy.instances[-1]


# --------------------------------------------------------------- encode / batch
def test_encode_corpus_is_one_long_tensor_of_ids():
    data = room.encode_corpus("abcabc", tiny_gpt.CharTokenizer.from_text("abc"))
    assert isinstance(data, torch.Tensor) and data.dtype == torch.long, f"encode_corpus returns an int64 tensor, got {type(data).__name__} {getattr(data, 'dtype', '')}."
    assert data.shape == (6,), f"One id per character: shape (6,), got {tuple(data.shape)}."
    assert data.tolist() == [0, 1, 2, 0, 1, 2]


def test_get_batch_shapes_and_dtype():
    g = torch.Generator().manual_seed(0)
    x, y = room.get_batch(DATA, 32, 5, g)
    assert x.shape == (5, 32) and y.shape == (5, 32), f"x and y are both (batch_size, block_size) = (5, 32); got {tuple(x.shape)} and {tuple(y.shape)}."
    assert x.dtype == torch.long and y.dtype == torch.long, "Token ids are int64 (torch.long); nn.Embedding insists."
    assert x.min() >= 0 and x.max() < TOKENIZER.vocab_size


def test_y_is_x_shifted_one_step_into_the_future():
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(1)
    x, y = room.get_batch(data, 16, 8, g)
    assert torch.equal(y[:, :-1], x[:, 1:]), (
        "y[b, t] must be x[b, t + 1]: the target at every position is the NEXT token. "
        f"Row 0: x = {x[0, :6].tolist()}..., y = {y[0, :6].tolist()}..."
    )
    assert torch.equal(x, x[:, :1] + torch.arange(16)), "Each row of x must be a contiguous window of the data."
    assert torch.equal(y, x + 1), "With data = arange, y should be exactly x + 1: a one-step shift, not a random other window."


def test_get_batch_never_runs_off_the_end_of_the_corpus():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(2)
    x, y = room.get_batch(data, 90, 400, g)
    assert y.max().item() <= 99, "y reached past the last token. A window starting at i needs i + block_size < len(data)."
    assert x.min().item() >= 0
    starts = set(x[:, 0].tolist())
    assert len(starts) > 1, "Every row started at the same index; the windows should be random."


def test_get_batch_obeys_the_generator():
    a = room.get_batch(DATA, 16, 4, torch.Generator().manual_seed(42))
    b = room.get_batch(DATA, 16, 4, torch.Generator().manual_seed(42))
    c = room.get_batch(DATA, 16, 4, torch.Generator().manual_seed(43))
    assert torch.equal(a[0], b[0]), "The same generator seed must give the same batch: pass generator= to torch.randint."
    assert not torch.equal(a[0], c[0]), "Different seeds gave the same batch; the generator is not being used."


# ---------------------------------------------------------------- the schedule
def test_lr_schedule_hits_its_marks_exactly():
    warmup, total, lr_max, lr_min = 10, 110, 1e-3, 1e-4
    s = lambda step: room.lr_schedule(step, warmup, total, lr_max, lr_min)  # noqa: E731
    assert math.isclose(s(0), lr_max / warmup, rel_tol=1e-9), f"Step 0 is the first warmup step: lr_max * 1 / warmup = {lr_max / warmup:.2e}, got {s(0):.3e}."
    assert math.isclose(s(warmup - 1), lr_max, rel_tol=1e-9), f"The last warmup step (step {warmup - 1}) reaches lr_max exactly; got {s(warmup - 1):.3e}."
    assert math.isclose(s(warmup), lr_max, rel_tol=1e-9), f"Cosine decay starts AT lr_max on step {warmup} (cos(0) = 1); got {s(warmup):.3e}."
    mid = warmup + (total - warmup) // 2
    assert math.isclose(s(mid), (lr_max + lr_min) / 2, rel_tol=1e-9), f"Halfway through the decay (step {mid}) the lr is the midpoint {(lr_max + lr_min) / 2:.2e}; got {s(mid):.3e}."
    three_quarters = warmup + 3 * (total - warmup) // 4
    want = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(3 * math.pi / 4))
    assert math.isclose(s(three_quarters), want, rel_tol=1e-9), f"At 3/4 of the decay the lr is {want:.4e}; got {s(three_quarters):.4e}."
    assert math.isclose(s(total), lr_min, rel_tol=1e-9), f"At step total the lr is lr_min = {lr_min:.1e}; got {s(total):.3e}."
    assert math.isclose(s(total + 50), lr_min, rel_tol=1e-9), "After total, the lr stays at lr_min."


def test_lr_schedule_is_a_ramp_and_then_a_cosine():
    warmup, total = 20, 200
    values = [room.lr_schedule(t, warmup, total, 1.0, 0.1) for t in range(total + 1)]
    ramp = values[:warmup]
    assert all(b > a for a, b in zip(ramp, ramp[1:])), "During warmup the learning rate must increase at every step."
    decay = values[warmup:]
    assert all(b <= a for a, b in zip(decay, decay[1:])), "After warmup the learning rate must never increase."
    assert min(values) >= 0.1 * (1 / warmup) - 1e-12 and max(values) <= 1.0 + 1e-12


# ------------------------------------------------------------------- training
def test_the_tower_learns_the_chronicles():
    torch.manual_seed(0)
    model = tower.GPT(SMALL)
    g = torch.Generator().manual_seed(0)
    losses = room.train(model, DATA, steps=STEPS, batch_size=BATCH, lr=LR, generator=g)
    assert isinstance(losses, list) and len(losses) == STEPS, f"train returns one loss per step: a list of {STEPS} floats, got {type(losses).__name__} of length {len(losses) if hasattr(losses, '__len__') else '?'}."
    assert all(isinstance(v, float) for v in losses), "Each recorded loss should be a Python float (loss.item())."
    assert losses[0] > 3.9, f"The first loss should be near ln(72) = 4.28 (a fresh model); got {losses[0]:.3f}."
    tail = sum(losses[-10:]) / 10
    assert tail < LOSS_AFTER, (
        f"After {STEPS} steps the mean loss over the last 10 steps is {tail:.3f}; the room asks for below {LOSS_AFTER}. "
        f"The reference reaches about 2.4. Check: y shifted by one, loss.backward() then optimizer.step(), zero_grad every step, lr={LR}."
    )
    held_out = room.estimate_loss(model, DATA, 4, torch.Generator().manual_seed(9))
    assert held_out < LOSS_AFTER + 0.2, f"estimate_loss after training is {held_out:.3f}, far above the training loss {tail:.3f}."


def test_gradients_are_clipped_before_every_step(spy):
    torch.manual_seed(0)
    model = tower.GPT(SMALL)
    clip = 1e-2  # absurdly small on purpose: every real gradient is far larger, so clipping must bite
    room.train(model, DATA, steps=5, batch_size=8, lr=1e-3, generator=torch.Generator().manual_seed(0), grad_clip=clip)
    opt = _the_optimizer(spy)
    assert len(opt.grad_norms) == 5, f"Expected 5 optimizer steps for 5 training steps, saw {len(opt.grad_norms)}."
    worst = max(opt.grad_norms)
    assert worst <= clip * 1.01, (
        f"The total gradient norm at optimizer.step() reached {worst:.4f} but grad_clip was {clip}. "
        "Call torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) AFTER loss.backward() and BEFORE optimizer.step()."
    )
    assert min(opt.grad_norms) > clip * 0.5, "The gradient norm is far below the clip value; the clip should scale it down to exactly grad_clip, not zero it."


def test_the_learning_rate_warms_up_and_cools_down(spy):
    torch.manual_seed(0)
    model = tower.GPT(SMALL)
    lr = 1e-3
    room.train(model, DATA, steps=20, batch_size=4, lr=lr, generator=torch.Generator().manual_seed(0))
    opt = _the_optimizer(spy)
    lrs = opt.lrs
    assert len(lrs) == 20
    assert len(set(lrs)) > 1, "The learning rate never changed. Set group['lr'] from lr_schedule(step, ...) every step."
    assert lrs[0] < lr, f"The first step should be a warmup step with lr below {lr}; got {lrs[0]:.2e}."
    assert math.isclose(max(lrs), lr, rel_tol=1e-6), f"The schedule should peak at exactly lr={lr}; it peaked at {max(lrs):.4e}."
    assert lrs[-1] < lr, f"The last step should be well into the cosine decay, below {lr}; got {lrs[-1]:.2e}."


def test_estimate_loss_scores_the_batches_it_draws_without_learning():
    torch.manual_seed(0)
    model = tower.GPT(SMALL)
    model.train()
    est = room.estimate_loss(model, DATA, 3, torch.Generator().manual_seed(5))
    assert isinstance(est, float), f"estimate_loss returns a Python float, got {type(est).__name__}."
    model.eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        want = sum(model(*room.get_batch(DATA, SMALL.block_size, 32, g))[1].item() for _ in range(3)) / 3
    assert abs(est - want) < 1e-4, (
        f"estimate_loss gave {est:.5f}; the mean loss of the 3 batches your own get_batch draws from that generator "
        f"(batch_size 32) is {want:.5f}. Use eval() mode: dropout must be off while measuring."
    )
    assert all(p.grad is None for p in model.parameters()), "estimate_loss left gradients behind: wrap it in torch.no_grad()."
