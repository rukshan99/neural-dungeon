"""TRIAL 8.1 - THE VAULT DOOR

The vault opens for someone who can load a checkpoint, freeze exactly what they
mean to, count what is left, and prove afterwards that nothing else moved.
"""

import copy

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

room = load_room(__file__, "room_1_the_vault_door")


@pytest.fixture(scope="module")
def chronicler():
    """Loaded once for the whole file. Tests deepcopy before touching anything."""
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


def _batch(tokenizer, text: str, n: int = 4, block: int = 32):
    ids = torch.tensor(tokenizer.encode(text[: n * (block + 1)]), dtype=torch.long)
    x = torch.stack([ids[i * block : (i + 1) * block] for i in range(n)])
    y = torch.stack([ids[i * block + 1 : (i + 1) * block + 1] for i in range(n)])
    return x, y


# ------------------------------------------------------------------ the door
def test_the_chronicler_is_loaded_with_its_tokenizer():
    model, tokenizer = room.load_chronicler()
    assert isinstance(model, nn.Module), "load_chronicler() must return the GPT module first."
    assert hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"), (
        "load_chronicler() must return the CharTokenizer second. load_pretrained() hands you three things; keep two."
    )
    assert tokenizer.vocab_size == 72, f"The Chronicler's tokenizer has 72 characters, yours has {tokenizer.vocab_size}."
    assert not model.training, "The vault keeps its models in eval mode. load_pretrained() already does this; do not undo it."
    assert all(p.requires_grad for p in model.parameters()), (
        "Fresh from the checkpoint every parameter is trainable. Freezing is a separate, deliberate act."
    )


def test_the_chronicler_still_knows_the_chronicles():
    model, tokenizer = room.load_chronicler()
    x, y = _batch(tokenizer, read_corpus("chronicles")[2000:], n=8, block=64)
    with torch.no_grad():
        _logits, loss = model(x, y)
    assert loss.item() < 1.0, (
        f"The pretrained Chronicler should read its own chronicles at loss ~0.2-0.5; yours scores {loss.item():.2f}. "
        "Did the weights actually load?"
    )


# ---------------------------------------------------------------- freezing
def test_freeze_everything_stops_every_gradient(chronicler):
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    n_tensors = len(list(model.named_parameters()))
    frozen = room.freeze(model)
    assert frozen == n_tensors, (
        f"freeze(model) should report every parameter tensor frozen: {n_tensors} of them (named_parameters counts "
        f"the tied lm_head/wte matrix once). You reported {frozen}."
    )
    assert not any(p.requires_grad for p in model.parameters()), "Some parameter still has requires_grad=True after freeze(model)."
    x, y = _batch(tokenizer, read_corpus("chronicles"))
    _logits, loss = model(x, y)
    assert not loss.requires_grad, (
        "With every parameter frozen the loss has nothing to differentiate with respect to: loss.requires_grad must be False."
    )
    assert all(p.grad is None for p in model.parameters()), "Frozen parameters must never receive a .grad."


def test_freeze_by_pattern_is_a_substring_match(chronicler):
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    frozen = room.freeze(model, patterns=["blocks.0.", "wte"])
    names_expected = {n for n, _p in model.named_parameters() if "blocks.0." in n or "wte" in n}
    assert frozen == len(names_expected), (
        f"Patterns are substrings of parameter names: 'blocks.0.' and 'wte' match {len(names_expected)} tensors; "
        f"you froze {frozen}."
    )
    for name, p in model.named_parameters():
        assert p.requires_grad == (name not in names_expected), f"{name}: requires_grad should be {name not in names_expected}."
    x, y = _batch(tokenizer, read_corpus("chronicles"))
    _logits, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        if name in names_expected:
            assert p.grad is None, f"{name} is frozen but received a gradient. requires_grad=False must be set on the Parameter itself."
        else:
            assert p.grad is not None, f"{name} is trainable but got no gradient. Did you freeze more than the patterns asked for?"


def test_freeze_with_no_match_freezes_nothing(chronicler):
    model = copy.deepcopy(chronicler[0])
    assert room.freeze(model, patterns=["there_is_no_such_layer"]) == 0
    assert all(p.requires_grad for p in model.parameters()), "No pattern matched, so nothing should have been frozen."


# ---------------------------------------------------------------- counting
def test_trainable_parameters_lists_name_and_parameter(chronicler):
    model = copy.deepcopy(chronicler[0])
    room.freeze(model, patterns=["blocks."])
    pairs = room.trainable_parameters(model)
    assert isinstance(pairs, list) and all(isinstance(t, tuple) and len(t) == 2 for t in pairs), (
        "trainable_parameters() returns a list of (name, parameter) tuples."
    )
    names = [n for n, _p in pairs]
    expected = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names == expected, f"Expected the trainable names {expected}, in named_parameters order; got {names}."
    assert all(isinstance(p, nn.Parameter) and p.requires_grad for _n, p in pairs)


def test_count_parameters_does_the_arithmetic(chronicler):
    model = copy.deepcopy(chronicler[0])
    total = sum(p.numel() for p in model.parameters())
    assert room.count_parameters(model) == total, (
        f"The Chronicler has {total:,} weights (tied lm_head counted once, as model.parameters() does). "
        f"You counted {room.count_parameters(model):,}."
    )
    assert room.count_parameters(model, trainable_only=True) == total, "Nothing is frozen yet: trainable == total."
    room.freeze(model, patterns=["blocks."])
    outside_blocks = sum(p.numel() for n, p in model.named_parameters() if "blocks." not in n)
    got = room.count_parameters(model, trainable_only=True)
    assert got == outside_blocks, (
        f"With every block frozen, only the embeddings and final LayerNorm remain trainable: {outside_blocks:,} weights, not {got:,}."
    )
    assert room.count_parameters(model) == total, "trainable_only=False still counts everything, frozen or not."


# ---------------------------------------------------------------- snapshot
def test_snapshot_is_a_detached_clone(chronicler):
    model = copy.deepcopy(chronicler[0])
    snap = room.snapshot(model)
    assert set(snap) == {n for n, _p in model.named_parameters()}, "snapshot() has one entry per named parameter."
    w = model.blocks[0].mlp.fc.weight
    assert not snap["blocks.0.mlp.fc.weight"].requires_grad, "Snapshot tensors must be detached from autograd."
    assert snap["blocks.0.mlp.fc.weight"].data_ptr() != w.data_ptr(), (
        "The snapshot shares memory with the live parameter, so it would change along with it. Clone."
    )
    original = snap["blocks.0.mlp.fc.weight"][0, 0].item()
    with torch.no_grad():
        w[0, 0] += 1.0
    assert snap["blocks.0.mlp.fc.weight"][0, 0].item() == original, "Editing the model must not edit the snapshot."


def test_changed_parameters_catches_a_single_edited_scalar(chronicler):
    model = copy.deepcopy(chronicler[0])
    snap = room.snapshot(model)
    assert room.changed_parameters(snap, model) == [], "Nothing was touched, so nothing should be reported."
    with torch.no_grad():
        model.blocks[1].mlp.fc.weight[3, 7] += 1e-3
    changed = room.changed_parameters(snap, model)
    assert changed == ["blocks.1.mlp.fc.weight"], (
        f"One scalar in blocks.1.mlp.fc.weight moved by 1e-3; with atol=0 that is a change. You reported {changed}."
    )


def test_changed_parameters_respects_atol(chronicler):
    model = copy.deepcopy(chronicler[0])
    snap = room.snapshot(model)
    with torch.no_grad():
        model.wpe.weight[5, 5] += 1e-4
        model.ln_f.weight[0] += 0.5
    assert room.changed_parameters(snap, model, atol=1e-3) == ["ln_f.weight"], (
        "A 1e-4 nudge is inside atol=1e-3 and should be forgiven; the 0.5 jump in ln_f.weight is not."
    )
    assert set(room.changed_parameters(snap, model, atol=0.0)) == {"wpe.weight", "ln_f.weight"}, (
        "With atol=0 both edited tensors are changes."
    )


def test_changed_parameters_notices_a_parameter_that_appeared(chronicler):
    model = copy.deepcopy(chronicler[0])
    snap = room.snapshot(model)
    model.extra_sigil = nn.Parameter(torch.zeros(3))
    changed = room.changed_parameters(snap, model)
    assert "extra_sigil" in changed, "A parameter that is not in the snapshot at all is a change too: report it."
