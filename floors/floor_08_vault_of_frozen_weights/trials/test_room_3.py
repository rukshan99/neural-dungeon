"""TRIAL 8.3 - THE ADAPTATION

Teach the Chronicler the goblin ledger through its sigils alone. The ledger loss
must fall from ~6.8 to below LEDGER_AFTER, and the frozen base must not move.
"""

import copy

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

room = load_room(__file__, "room_3_the_adaptation")
sigil = load_room(__file__, "room_2_the_low_rank_sigil")

STEPS = 60
LR = 3e-3
LEDGER_AFTER = 1.5  # the reference lands at 0.77-0.84 across seeds; 6.8 is where it starts


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def corpora():
    return read_corpus("chronicles"), read_corpus("ledger")


_ENCODED: dict[str, "torch.Tensor"] = {}


def _data(tokenizer, text):
    if text not in _ENCODED:
        _ENCODED[text] = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return _ENCODED[text]


@torch.no_grad()
def _loss_on(model, tokenizer, text: str, n_batches: int = 8, block: int = 64, batch: int = 16, seed: int = 0) -> float:
    """Trial-owned evaluation, independent of the learner's evaluate_loss."""
    gen = torch.Generator().manual_seed(seed)
    data = _data(tokenizer, text)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block] for i in ix])
        total += model(x, y)[1].item()
    model.train(was_training)
    return total / n_batches


class _RecordingAdamW(torch.optim.AdamW):
    handed: list = []

    def __init__(self, params, *args, **kwargs):
        params = list(params)
        _RecordingAdamW.handed.append(params)
        super().__init__(params, *args, **kwargs)


@pytest.fixture(scope="module")
def adapted(chronicler, corpora):
    """LoRA r=8 on a frozen copy, fine-tuned on the ledger once for the whole file."""
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    for p in model.parameters():
        p.requires_grad_(False)
    sigil.inject_lora(model, r=8, alpha=16.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.manual_seed(8)
    original_adamw = torch.optim.AdamW
    torch.optim.AdamW = _RecordingAdamW
    try:
        losses = room.finetune(model, corpora[1], tokenizer, steps=STEPS, lr=LR, generator=torch.Generator().manual_seed(8))
    finally:
        torch.optim.AdamW = original_adamw
    return model, before, losses


# ---------------------------------------------------------------- batches
def test_batches_are_shifted_windows_of_the_text(chronicler, corpora):
    tokenizer = chronicler[1]
    stream = room.make_batches(corpora[0], tokenizer, block_size=32, batch_size=6, generator=torch.Generator().manual_seed(1))
    x, y = next(iter(stream))
    assert tuple(x.shape) == (6, 32) and tuple(y.shape) == (6, 32), f"x and y are (batch_size, block_size) = (6, 32); got {tuple(x.shape)} and {tuple(y.shape)}."
    assert x.dtype == torch.long and y.dtype == torch.long, "Token ids are int64 (torch.long); nn.Embedding insists."
    assert torch.equal(x[:, 1:], y[:, :-1]), "y is x shifted left by one: y[:, t] == x[:, t + 1]. The target of a character is the next character."
    for row in range(6):
        window = tokenizer.decode(x[row].tolist()) + tokenizer.decode(y[row, -1:].tolist())
        assert window in corpora[0], f"Row {row} decodes to text that is not in the corpus: {window!r}. Windows must be contiguous slices."


def test_batches_keep_coming_and_differ(chronicler, corpora):
    tokenizer = chronicler[1]
    stream = iter(room.make_batches(corpora[1], tokenizer, block_size=16, batch_size=4, generator=torch.Generator().manual_seed(2)))
    first = [next(stream) for _ in range(5)]
    assert len(first) == 5, "make_batches is an endless stream: next() must keep working."
    assert not torch.equal(first[0][0], first[1][0]), "Consecutive batches should sample different windows."


def test_batches_are_reproducible_from_the_generator(chronicler, corpora):
    tokenizer = chronicler[1]
    a = next(iter(room.make_batches(corpora[0], tokenizer, 24, 3, torch.Generator().manual_seed(9))))
    b = next(iter(room.make_batches(corpora[0], tokenizer, 24, 3, torch.Generator().manual_seed(9))))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (
        "The same generator seed must give the same batches. Draw the window starts with torch.randint(..., generator=generator)."
    )


# ---------------------------------------------------------------- finetune
def test_the_ledger_loss_collapses(adapted, chronicler, corpora):
    model, _before, losses = adapted
    assert isinstance(losses, list) and len(losses) == STEPS, f"finetune() returns one float per step: {STEPS} of them, got {len(losses) if isinstance(losses, list) else type(losses).__name__}."
    assert all(isinstance(v, float) for v in losses), "Return Python floats (loss.item()), not tensors: tensors keep the graph alive."
    assert losses[0] > 5.0, f"The first step's loss should be the pretrained model's ~6.5 on ledger text; yours was {losses[0]:.2f}. Are you training on the right corpus?"
    after = _loss_on(model, chronicler[1], corpora[1])
    assert after < LEDGER_AFTER, (
        f"After {STEPS} LoRA steps at lr {LR} the ledger loss should be below {LEDGER_AFTER} (reference: ~0.8); yours is {after:.2f}. "
        "Check the optimizer step, the learning rate, and that the sigils are actually trainable."
    )
    assert losses[-1] < losses[0] / 3, f"The training loss barely moved: {losses[0]:.2f} -> {losses[-1]:.2f}."


def test_the_frozen_base_did_not_move(adapted):
    model, before, _losses = adapted
    moved = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
    base_moved = [n for n in moved if "lora_" not in n]
    assert not base_moved, (
        f"{len(base_moved)} frozen base tensors changed during fine-tuning, e.g. {base_moved[:3]}. "
        "Only parameters with requires_grad=True may be handed to the optimizer."
    )
    lora_moved = [n for n in moved if "lora_" in n]
    assert len(lora_moved) == 32, f"All 32 LoRA tensors should have moved; only {len(lora_moved)} did. Did lora_B stay at zero because lora_A never trained?"


def test_only_trainable_parameters_were_handed_to_adamw(adapted):
    model, _before, _losses = adapted
    assert _RecordingAdamW.handed, "finetune() must build a torch.optim.AdamW. None was constructed."
    handed = _RecordingAdamW.handed[-1]
    ids = {id(p) for p in handed if isinstance(p, torch.Tensor)}
    if not ids:  # param groups
        ids = {id(p) for g in handed for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids == trainable, (
        f"AdamW received {len(ids)} tensors but the model has {len(trainable)} trainable ones. "
        "Hand the optimizer exactly [p for p in model.parameters() if p.requires_grad]: frozen weights do not belong there."
    )


def test_finetune_restores_eval_mode(adapted):
    model, _before, _losses = adapted
    assert not model.training, "The model was in eval mode before finetune(); put it back when you are done (dropout must be off for evaluation)."


# ---------------------------------------------------------------- evaluate
def test_evaluate_loss_agrees_with_the_trial_on_the_pristine_model(chronicler, corpora):
    model, tokenizer = chronicler
    chron = room.evaluate_loss(model, corpora[0], tokenizer, n_batches=16, block_size=64, generator=torch.Generator().manual_seed(3))
    ledger = room.evaluate_loss(model, corpora[1], tokenizer, n_batches=16, block_size=64, generator=torch.Generator().manual_seed(3))
    assert isinstance(chron, float), "evaluate_loss returns a Python float."
    assert abs(chron - _loss_on(model, tokenizer, corpora[0], n_batches=16)) < 0.15, f"Chronicles loss should be ~0.24; you measured {chron:.3f}."
    assert abs(ledger - _loss_on(model, tokenizer, corpora[1], n_batches=16)) < 0.5, f"Ledger loss should be ~6.8; you measured {ledger:.3f}."


def test_evaluate_loss_leaves_no_trace(chronicler, corpora):
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    room.evaluate_loss(model, corpora[0], tokenizer, n_batches=2, block_size=32, generator=torch.Generator().manual_seed(4))
    assert all(p.grad is None for p in model.parameters()), "Evaluation must run under torch.no_grad(): no gradients should be left on the parameters."
    assert all(torch.equal(p.detach(), before[n]) for n, p in model.named_parameters()), "Evaluation must not change any weight."
    assert not model.training, "Evaluate in eval mode and leave the model as you found it."
