"""TRIAL 8.2 - THE LOW-RANK SIGIL

A sigil drawn on a frozen door: at first it changes nothing (B is zero), it can
be trained while the door stays frozen, and it can be melted back into the
door (merge) leaving no trace of itself.
"""

import copy

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

room = load_room(__file__, "room_2_the_low_rank_sigil")

TARGETS = ("c_attn", "c_proj", "fc", "proj")
EXPECTED_NAMES = [f"blocks.{i}.{leaf}" for i in range(4) for leaf in ("attn.c_attn", "attn.c_proj", "mlp.fc", "mlp.proj")]


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


def _batch(tokenizer, n: int = 4, block: int = 48):
    text = read_corpus("chronicles")[5000:]
    ids = torch.tensor(tokenizer.encode(text[: n * block]), dtype=torch.long)
    return ids.view(n, block)


def _expected_lora_count(model, r, targets=TARGETS):
    return sum(r * (m.in_features + m.out_features) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets)


# ----------------------------------------------------------- one sigil
def test_a_fresh_sigil_changes_nothing():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(5, 16)
    assert torch.equal(lora(x), base(x)), (
        "At init the LoRA layer must compute exactly base(x). lora_B starts as zeros so B @ A == 0; "
        "if it does not, the fine-tune starts from a different function than the checkpoint."
    )
    assert torch.count_nonzero(lora.lora_B) == 0, "lora_B must be initialised to zeros."
    assert torch.count_nonzero(lora.lora_A) > 0, (
        "lora_A must NOT be zero: if both factors start at zero, both gradients are zero and nothing ever trains."
    )


def test_the_sigil_has_the_right_shapes_and_scaling():
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0)
    assert tuple(lora.lora_A.shape) == (4, 16), f"lora_A is (r, in_features) = (4, 16); yours is {tuple(lora.lora_A.shape)}."
    assert tuple(lora.lora_B.shape) == (24, 4), f"lora_B is (out_features, r) = (24, 4); yours is {tuple(lora.lora_B.shape)}."
    assert lora.scaling == pytest.approx(2.0), f"scaling = alpha / r = 8 / 4 = 2.0, not {lora.scaling}."
    assert lora.base is base, "Keep the wrapped layer as .base; the trials (and merge) look for it there."
    assert not base.weight.requires_grad and not base.bias.requires_grad, "Wrapping must freeze the base layer's weight and bias."
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad, "The two LoRA factors are the only trainable tensors."
    assert getattr(lora, "enabled", None) is True, "A LoRALinear carries an `enabled` flag (True by default); the boss will flip it."


def test_the_sigil_computes_the_low_rank_formula():
    torch.manual_seed(1)
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(7, 16)
    expected = base(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * 2.0
    torch.testing.assert_close(lora(x), expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(lora(x), base(x)), "With a non-zero B the output must differ from the base."


def test_disabling_the_sigil_gives_the_door_back():
    torch.manual_seed(2)
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(3, 16)
    lora.enabled = False
    assert torch.equal(lora(x), base(x)), "With enabled=False the layer must return exactly base(x), whatever A and B hold."
    lora.enabled = True
    assert not torch.equal(lora(x), base(x)), "enabled=True must switch the correction back on."


def test_dropout_falls_on_the_sigil_not_the_door():
    torch.manual_seed(3)
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0, dropout=0.5)
    x = torch.randn(64, 16)
    lora.train()
    assert torch.equal(lora(x), base(x)), (
        "Dropout applies to the input of the LoRA path only. With B == 0 the LoRA path is zero regardless, so the output "
        "in train mode must still be exactly base(x). If it is not, you dropped out the base path."
    )
    with torch.no_grad():
        lora.lora_B.normal_()
    lora.eval()
    expected = base(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * 2.0
    torch.testing.assert_close(lora(x), expected, rtol=1e-5, atol=1e-6)
    lora.train()
    assert not torch.allclose(lora(x), expected), "In train mode with dropout=0.5 the LoRA path must be stochastic."


# ----------------------------------------------------------- merge
def test_merge_folds_the_sigil_into_a_plain_linear():
    torch.manual_seed(4)
    base = nn.Linear(16, 24)
    lora = room.LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_()
    merged = lora.merge()
    assert type(merged) is nn.Linear, f"merge() returns a plain nn.Linear, not {type(merged).__name__}."
    assert merged.in_features == 16 and merged.out_features == 24
    x = torch.randn(9, 16)
    torch.testing.assert_close(merged(x), lora(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged.weight, base.weight + 2.0 * (lora.lora_B @ lora.lora_A), rtol=1e-6, atol=1e-6)
    assert torch.equal(merged.bias, base.bias), "The bias is not part of LoRA; the merged layer keeps the base bias."
    assert torch.equal(base.weight, lora.base.weight), "merge() must not modify the frozen base in place; build a new Linear."


def test_merge_without_a_bias_stays_without_a_bias():
    base = nn.Linear(8, 8, bias=False)
    lora = room.LoRALinear(base, r=2, alpha=2.0)
    assert lora.merge().bias is None, "A bias-free base merges into a bias-free Linear."


# ----------------------------------------------------------- inject
def test_inject_replaces_the_sixteen_targets_in_order(chronicler):
    model = copy.deepcopy(chronicler[0])
    names = room.inject_lora(model, r=4, alpha=8.0)
    assert names == EXPECTED_NAMES, (
        f"Expected these 16 modules replaced, in named_modules order:\n{EXPECTED_NAMES}\ngot:\n{names}"
    )
    for name in EXPECTED_NAMES:
        mod = model.get_submodule(name)
        assert isinstance(mod, room.LoRALinear), f"{name} should now be a LoRALinear, it is {type(mod).__name__}. Use setattr on the parent module."
    assert type(model.lm_head) is nn.Linear, "lm_head is not a target (and its weight is tied to wte): leave it alone."
    again = room.inject_lora(model, r=4, alpha=8.0)
    n_sigils = sum(isinstance(m, room.LoRALinear) for m in model.modules())
    assert again == [] and n_sigils == 16, (
        f"Injecting a second time wrapped {len(again)} more layers ({n_sigils} LoRALinear modules now). A LoRALinear is not an "
        "nn.Linear and its frozen `.base` is not a target: never wrap a sigil in a sigil."
    )


def test_inject_leaves_the_forward_pass_untouched(chronicler):
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    x = _batch(tokenizer)
    with torch.no_grad():
        before, _ = chronicler[0](x)
        room.inject_lora(model, r=8, alpha=16.0)
        after, _ = model(x)
    assert torch.equal(before, after), (
        f"Injecting LoRA must not change a single logit (B is zero). Max difference: {(before - after).abs().max().item():.3e}."
    )


def test_after_injection_only_the_sigils_are_trainable(chronicler):
    model = copy.deepcopy(chronicler[0])
    for p in model.parameters():
        p.requires_grad_(False)
    room.inject_lora(model, r=4, alpha=8.0)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "Nothing is trainable: the LoRA factors must be nn.Parameters with requires_grad=True."
    assert all(n.endswith("lora_A") or n.endswith("lora_B") for n in trainable), (
        f"Only lora_A / lora_B may be trainable after injecting into a frozen model; found {[n for n in trainable if 'lora_' not in n][:4]}."
    )
    assert len(trainable) == 32, f"16 layers x (A, B) = 32 trainable tensors; you have {len(trainable)}."


def test_the_parameter_arithmetic_for_rank_four(chronicler):
    model = copy.deepcopy(chronicler[0])
    expected = _expected_lora_count(chronicler[0], r=4)
    room.inject_lora(model, r=4, alpha=8.0)
    got = room.lora_parameter_count(model)
    assert got == expected, (
        f"r=4 on c_attn (128->384), c_proj (128->128), fc (128->512), proj (512->128) in 4 blocks is "
        f"sum r*(in+out) = {expected:,} weights; you counted {got:,}."
    )
    total = sum(p.numel() for p in chronicler[0].parameters())
    assert got / total < 0.05, "That is under 5% of the Chronicler. This is the whole point of parameter-efficient tuning."
    assert room.lora_parameter_count(chronicler[0]) == 0, "A model without adapters has zero LoRA parameters."


def test_inject_respects_a_narrower_target_list(chronicler):
    model = copy.deepcopy(chronicler[0])
    names = room.inject_lora(model, r=2, alpha=4.0, targets=("c_attn",))
    assert names == [f"blocks.{i}.attn.c_attn" for i in range(4)], f"targets=('c_attn',) should wrap exactly the four fused qkv projections; got {names}."
    assert type(model.blocks[0].mlp.fc) is nn.Linear, "fc was not a target and must stay a plain Linear."


def test_lora_modules_finds_every_sigil(chronicler):
    model = copy.deepcopy(chronicler[0])
    names = room.inject_lora(model, r=2, alpha=4.0)
    found = room.lora_modules(model)
    assert [n for n, _m in found] == names, "lora_modules() returns (name, module) for every LoRALinear, in module order."
    assert all(isinstance(m, room.LoRALinear) for _n, m in found)


def test_the_state_dict_holds_only_sigils(chronicler):
    model = copy.deepcopy(chronicler[0])
    room.inject_lora(model, r=4, alpha=8.0)
    sd = room.lora_state_dict(model)
    assert len(sd) == 32, f"16 layers x (lora_A, lora_B) = 32 tensors; your adapter state dict has {len(sd)}."
    assert all(k.endswith("lora_A") or k.endswith("lora_B") for k in sd), (
        f"Only lora_A / lora_B belong in the adapter state dict; found {[k for k in sd if 'lora_' not in k][:3]}."
    )
    assert tuple(sd["blocks.2.mlp.fc.lora_A"].shape) == (4, 128) and tuple(sd["blocks.2.mlp.fc.lora_B"].shape) == (512, 4)
    live = model.blocks[2].mlp.fc.lora_A
    assert not sd["blocks.2.mlp.fc.lora_A"].requires_grad and sd["blocks.2.mlp.fc.lora_A"].data_ptr() != live.data_ptr(), (
        "The adapter state dict must hold detached CLONES: a view of the live parameter would change under you the moment training resumed."
    )
    total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    full_bytes = sum(p.numel() * p.element_size() for p in chronicler[0].parameters())
    assert total_bytes < full_bytes / 20, "The adapter file should be a small fraction of the full checkpoint. That is why people ship adapters."
