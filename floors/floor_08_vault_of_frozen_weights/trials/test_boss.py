"""BOSS FIGHT - THE CATASTROPHIC FORGETTER

Phase 1: teach the Chronicler the ledger without letting the Forgetter erase the chronicles.
Phase 2: adapters as a detachable part: save, switch off (pristine again), switch on, load elsewhere.
Phase 3: merge the sigils into plain weights and export a checkpoint a fresh GPT accepts.
Phase 4: prophesy which strategies survive the Forgetter's two-loss constraint.
"""

import copy

import pytest

from dungeon.artifacts.tiny_gpt import GPT, GPTConfig, load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

boss = load_room(__file__, "boss_catastrophic_forgetter")
sigil = load_room(__file__, "room_2_the_low_rank_sigil")

pytestmark = pytest.mark.boss

STEPS = 25
SEED = 88
LEDGER_GOAL = 1.6        # ledger loss the adapted model must reach (from ~6.8); LoRA + replay lands at ~1.3
FORGETTING_BUDGET = 0.35  # how much the chronicles loss may rise; LoRA + replay costs ~0.1, full + replay ~0.2
STRATEGIES = ("full_lr_1e-3", "lora_r8_lr_3e-3", "full_lr_1e-3_replay_50", "lora_r8_lr_3e-3_replay_50")


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def corpora():
    return read_corpus("chronicles"), read_corpus("ledger")


_ENCODED: dict[str, "torch.Tensor"] = {}


def _data(tokenizer, text):
    """Encode each corpus once per module; str hashes are cached so the lookup is cheap."""
    if text not in _ENCODED:
        _ENCODED[text] = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return _ENCODED[text]


def _windows(tokenizer, text, n, block, gen):
    data = _data(tokenizer, text)
    ix = torch.randint(0, len(data) - block - 1, (n,), generator=gen)
    x = torch.stack([data[i : i + block] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def _loss_on(model, tokenizer, text, n_batches=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = _windows(tokenizer, text, 16, 64, gen)
        total += model(x, y)[1].item()
    model.train(was_training)
    return total / n_batches


def _probe(tokenizer, text, seed=1):
    return _windows(tokenizer, text, 8, 64, torch.Generator().manual_seed(seed))[0]


def _reference_finetune(model, tokenizer, new_text, old_text, lr, lora_r=None, replay=0.0):
    """The trial's own plain training loop, used to judge the phase-4 prophecy."""
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    m = copy.deepcopy(model)
    if lora_r is not None:
        for p in m.parameters():
            p.requires_grad_(False)
        sigil.inject_lora(m, r=lora_r, alpha=2.0 * lora_r)
    else:
        for p in m.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    n_old = int(round(16 * replay))
    m.train()
    for _ in range(STEPS):
        x, y = _windows(tokenizer, new_text, 16 - n_old, 64, gen)
        if n_old:
            xo, yo = _windows(tokenizer, old_text, n_old, 64, gen)
            x, y = torch.cat([x, xo]), torch.cat([y, yo])
        loss = m(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval()
    return m


# ----------------------------------------------------------------- phase 1
@pytest.fixture(scope="module")
def adapted(chronicler, corpora):
    model, tokenizer = chronicler
    chron, ledger = corpora
    torch.manual_seed(SEED)
    result = boss.adapt_without_forgetting(model, tokenizer, ledger, chron, STEPS, torch.Generator().manual_seed(SEED))
    return result


@pytest.fixture(scope="module")
def base_losses(chronicler, corpora):
    model, tokenizer = chronicler
    return _loss_on(model, tokenizer, corpora[0]), _loss_on(model, tokenizer, corpora[1])


def test_phase_1_the_vault_is_untouched(chronicler, adapted):
    model = chronicler[0]
    pristine, _t, _e = load_pretrained()
    assert adapted is not model, "adapt_without_forgetting() must return a new model, not the one you were handed."
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), pristine.parameters())), (
        "The pristine Chronicler was modified. The vault's weights stay frozen and untouched: deepcopy before you adapt."
    )
    assert not adapted.training, "Return the adapted model in eval mode."


def test_phase_1_the_new_tongue_is_learned(chronicler, corpora, adapted):
    tokenizer = chronicler[1]
    new_loss = _loss_on(adapted, tokenizer, corpora[1])
    assert new_loss <= LEDGER_GOAL, (
        f"The adapted model reads the goblin ledger at loss {new_loss:.2f}; the Forgetter demands <= {LEDGER_GOAL}. "
        f"(The reference reaches ~1.3 in {STEPS} steps.) More learning rate, or more of the batch spent on the ledger."
    )


def test_phase_1_the_old_tongue_is_kept(chronicler, corpora, adapted, base_losses):
    tokenizer = chronicler[1]
    old_loss = _loss_on(adapted, tokenizer, corpora[0])
    assert old_loss <= base_losses[0] + FORGETTING_BUDGET, (
        f"THE FORGETTER LAUGHS. Chronicles loss went {base_losses[0]:.2f} -> {old_loss:.2f}, more than the budget of {FORGETTING_BUDGET}. "
        "Nothing in a plain fine-tune cares about the old text. Give the loss a reason to: replay chronicles rows in every batch, "
        "and move as few weights as you can (LoRA)."
    )


def test_phase_1_the_forgetters_own_way_fails(survivors):
    """The taunt is real: a naive full fine-tune at lr 1e-3 blows the chronicles budget several times over."""
    _survives, _new_loss, rise = survivors["full_lr_1e-3"]
    assert rise > 3 * FORGETTING_BUDGET, f"(trial self-check) naive full fine-tune should forget badly; it rose by only {rise:.2f}."


# ----------------------------------------------------------------- phase 2
def test_phase_2_the_adapter_is_a_small_separate_state(adapted, chronicler):
    state = boss.AdapterSwitch.save_adapter(adapted)
    assert isinstance(state, dict) and "weights" in state, "save_adapter() returns a dict with a 'weights' entry (plus r, alpha, targets)."
    weights = state["weights"]
    assert weights and all(k.endswith("lora_A") or k.endswith("lora_B") for k in weights), (
        "The adapter state holds ONLY lora_A / lora_B tensors. The frozen base ships separately; that is the point."
    )
    n_base = sum(p.numel() for p in chronicler[0].parameters())
    n_adapter = sum(v.numel() for v in weights.values())
    assert n_adapter < n_base / 5, f"The adapter is {n_adapter:,} weights against a {n_base:,}-weight base; it should be a small fraction."
    assert isinstance(state.get("r"), int) and state["r"] > 0, "Record r in the state: whoever loads the adapter must rebuild the same shapes."


def test_phase_2_switching_off_gives_back_the_pristine_chronicler(adapted, chronicler, corpora):
    model, tokenizer = chronicler
    x = _probe(tokenizer, corpora[0])
    boss.AdapterSwitch.disable_adapters(adapted)
    try:
        with torch.no_grad():
            pristine_logits = model(x)[0]
            off_logits = adapted(x)[0]
        diff = (pristine_logits - off_logits).abs().max().item()
        assert diff <= 1e-6, (
            f"With adapters disabled the model should BE the pristine Chronicler; logits differ by up to {diff:.3e}. "
            "Either a base weight moved during adaptation, or disable_adapters() did not reach every LoRALinear."
        )
        ledger_off = _loss_on(adapted, tokenizer, corpora[1])
        assert ledger_off > 5.0, f"Adapters off, the model should have forgotten the ledger entirely (loss ~6.8); it reads it at {ledger_off:.2f}."
    finally:
        boss.AdapterSwitch.enable_adapters(adapted)
    ledger_on = _loss_on(adapted, tokenizer, corpora[1])
    assert ledger_on <= LEDGER_GOAL, f"Adapters back on, the ledger should be readable again (<= {LEDGER_GOAL}); got {ledger_on:.2f}."


def test_phase_2_the_adapter_loads_onto_a_pristine_copy(adapted, chronicler, corpora):
    model, tokenizer = chronicler
    state = boss.AdapterSwitch.save_adapter(adapted)
    fresh = copy.deepcopy(model)
    returned = boss.AdapterSwitch.load_adapter(fresh, state)
    fresh = returned if isinstance(returned, nn.Module) else fresh
    x = _probe(tokenizer, corpora[1])
    with torch.no_grad():
        want = adapted(x)[0]
        got = fresh(x)[0]
    diff = (want - got).abs().max().item()
    assert diff <= 1e-6, (
        f"A pristine model plus the saved adapter should match the adapted model; logits differ by {diff:.3e}. "
        "load_adapter() onto a model without adapters must inject them first (same r, alpha, targets), then copy the weights."
    )
    assert torch.equal(fresh.wte.weight, model.wte.weight), "Loading an adapter must not touch the base weights."


# ----------------------------------------------------------------- phase 3
def test_phase_3_merge_and_export_loads_into_a_fresh_gpt(adapted, chronicler, corpora):
    model, tokenizer = chronicler
    sd = boss.merge_and_export(adapted)
    assert set(sd) == set(model.state_dict()), (
        f"The exported state_dict must have exactly a plain GPT's keys. Extra: {sorted(set(sd) - set(model.state_dict()))[:3]}, "
        f"missing: {sorted(set(model.state_dict()) - set(sd))[:3]}."
    )
    fresh = GPT(GPTConfig(**model.cfg.to_dict()))
    fresh.load_state_dict(sd, strict=True)
    fresh.eval()
    x = _probe(tokenizer, corpora[1])
    with torch.no_grad():
        want = adapted(x)[0]
        got = fresh(x)[0]
    diff = (want - got).abs().max().item()
    assert diff <= 1e-5, (
        f"The merged plain GPT differs from the adapted model by up to {diff:.3e}. Merged weight = W + (alpha/r) * B @ A, same bias."
    )
    ledger = _loss_on(fresh, tokenizer, corpora[1])
    assert ledger <= LEDGER_GOAL, f"The merged model should still read the ledger (<= {LEDGER_GOAL}); got {ledger:.2f}."


def test_phase_3_export_does_not_destroy_the_adapters(adapted):
    n_before = sum(isinstance(m, sigil.LoRALinear) for m in adapted.modules())
    sd = boss.merge_and_export(adapted)
    n_after = sum(isinstance(m, sigil.LoRALinear) for m in adapted.modules())
    assert n_after == n_before == 16, "merge_and_export() must work on a copy: the adapted model keeps its 16 LoRALinear modules."
    live = dict(adapted.named_parameters())["blocks.0.mlp.fc.base.weight"]
    assert sd["blocks.0.mlp.fc.weight"].data_ptr() != live.data_ptr(), "Exported tensors must not alias the live model's parameters."


# ----------------------------------------------------------------- phase 4
@pytest.fixture(scope="module")
def survivors(chronicler, corpora, base_losses):
    model, tokenizer = chronicler
    chron, ledger = corpora
    recipes = {
        "full_lr_1e-3": dict(lr=1e-3),
        "lora_r8_lr_3e-3": dict(lr=3e-3, lora_r=8),
        "full_lr_1e-3_replay_50": dict(lr=1e-3, replay=0.5),
        "lora_r8_lr_3e-3_replay_50": dict(lr=3e-3, lora_r=8, replay=0.5),
    }
    verdicts = {}
    for name, recipe in recipes.items():
        m = _reference_finetune(model, tokenizer, ledger, chron, **recipe)
        new_loss = _loss_on(m, tokenizer, ledger)
        old_rise = _loss_on(m, tokenizer, chron) - base_losses[0]
        verdicts[name] = (new_loss <= LEDGER_GOAL and old_rise <= FORGETTING_BUDGET, new_loss, old_rise)
    return verdicts


def test_phase_4_the_prophecy_is_complete():
    assert set(boss.FORGETTER_PROPHECY) == set(STRATEGIES), f"FORGETTER_PROPHECY names exactly these strategies: {STRATEGIES}."


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_phase_4_who_survives_the_forgetter(survivors, strategy):
    said = boss.FORGETTER_PROPHECY.get(strategy)
    if said is None:
        raise NotImplementedError(f"You have not prophesied whether {strategy!r} survives. Fill in FORGETTER_PROPHECY.")
    survives, new_loss, old_rise = survivors[strategy]
    assert said == survives, (
        f"{strategy} after {STEPS} steps: ledger loss {new_loss:.2f} (goal <= {LEDGER_GOAL}), chronicles rose by {old_rise:.2f} "
        f"(budget {FORGETTING_BUDGET}). It {'survives' if survives else 'is devoured'}; you prophesied {said}."
    )
