"""TRIAL 8.4 - THE FORGETTING

Four ways to teach the Chronicler the ledger, all for the same number of steps
on the same batches. Measure how much of the chronicles each one erases, then
read your prophecy aloud.
"""


import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn

room = load_room(__file__, "room_4_the_forgetting")
sigil = load_room(__file__, "room_2_the_low_rank_sigil")

STEPS = 30
SEED = 8
STRATEGIES = ("full_lr_1e-3", "full_lr_1e-4", "lora_r8_lr_1e-3", "full_lr_1e-3_replay_50")
FORGETTING_THRESHOLD = 0.35  # nats of chronicles loss; reference deltas: 2.2-2.5 / 0.48-0.58 / 1.1-1.2 / 0.17-0.20


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def corpora():
    return read_corpus("chronicles"), read_corpus("ledger")


def _gen(offset: int = 0):
    return torch.Generator().manual_seed(SEED + offset)


@pytest.fixture(scope="module")
def outcomes(chronicler, corpora):
    """Every strategy fine-tuned once on the ledger, plus its forgetting report."""
    model, tokenizer = chronicler
    chron, ledger = corpora
    models = {}
    torch.manual_seed(SEED)
    models["full_lr_1e-3"] = room.full_finetune_copy(model, ledger, tokenizer, STEPS, 1e-3, generator=_gen())
    torch.manual_seed(SEED)
    models["full_lr_1e-4"] = room.full_finetune_copy(model, ledger, tokenizer, STEPS, 1e-4, generator=_gen())
    torch.manual_seed(SEED)
    models["lora_r8_lr_1e-3"] = room.lora_finetune_copy(model, ledger, tokenizer, STEPS, 1e-3, r=8, alpha=16.0, generator=_gen())
    torch.manual_seed(SEED)
    models["full_lr_1e-3_replay_50"] = room.replay_finetune_copy(model, ledger, chron, tokenizer, STEPS, 1e-3, replay_fraction=0.5, generator=_gen())
    reports = {k: room.forgetting_report(model, m, tokenizer, chron, ledger, generator=_gen(100)) for k, m in models.items()}
    return models, reports


# ------------------------------------------------------------ the report
def test_the_report_of_an_unchanged_model_is_all_zeros(chronicler, corpora):
    model, tokenizer = chronicler
    rep = room.forgetting_report(model, model, tokenizer, corpora[0], corpora[1], generator=_gen(5))
    for key in ("old_before", "old_after", "new_before", "new_after", "delta_old", "delta_new"):
        assert key in rep, f"The report is missing {key!r}. Keys: old_before, old_after, new_before, new_after, delta_old, delta_new."
    assert rep["delta_old"] == 0.0 and rep["delta_new"] == 0.0, (
        f"Comparing a model with itself gave delta_old={rep['delta_old']:.4f}, delta_new={rep['delta_new']:.4f}. "
        "Before and after must be measured on the SAME batches, or the noise of sampling different windows masquerades as forgetting."
    )
    assert 0.1 < rep["old_before"] < 0.6 and 5.5 < rep["new_before"] < 8.0, (
        f"old_before should be the chronicles loss (~0.24) and new_before the ledger loss (~6.8); got {rep['old_before']:.2f} and {rep['new_before']:.2f}."
    )


def test_the_report_signs_point_the_right_way(outcomes):
    rep = outcomes[1]["full_lr_1e-3"]
    assert rep["delta_new"] < -4.0, f"Fine-tuning on the ledger should cut the ledger loss by several nats; delta_new = {rep['delta_new']:.2f}."
    assert rep["delta_old"] > 1.0, f"Full fine-tuning at lr 1e-3 should visibly damage the chronicles; delta_old = {rep['delta_old']:.2f}."
    assert rep["delta_old"] == pytest.approx(rep["old_after"] - rep["old_before"]), "delta_old = old_after - old_before."


# ------------------------------------------------------------ the copies
def test_full_finetune_copy_leaves_the_original_in_the_vault(chronicler, outcomes):
    model = chronicler[0]
    pristine, _tok, _extra = load_pretrained()
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), pristine.parameters())), (
        "full_finetune_copy() trained the model you passed in. Deep-copy first: the original must stay pristine."
    )
    tuned = outcomes[0]["full_lr_1e-3"]
    assert tuned is not model
    moved = [n for (n, p), q in zip(tuned.named_parameters(), model.parameters()) if not torch.equal(p, q)]
    assert len(moved) == len(list(model.named_parameters())), (
        f"A FULL fine-tune moves every parameter tensor; only {len(moved)} of {len(list(model.named_parameters()))} moved. Set requires_grad=True on all of them."
    )
    assert not any(isinstance(m, sigil.LoRALinear) for m in tuned.modules()), "A full fine-tune has no adapters."


def test_lora_finetune_copy_moves_only_the_sigils(chronicler, outcomes):
    model = chronicler[0]
    tuned = outcomes[0]["lora_r8_lr_1e-3"]
    loras = [m for m in tuned.modules() if isinstance(m, sigil.LoRALinear)]
    assert len(loras) == 16, f"lora_finetune_copy should inject LoRA into the 16 target Linears; found {len(loras)} LoRALinear modules."
    assert loras[0].r == 8, f"r=8 was requested; the sigils have r={loras[0].r}."
    pristine = dict(model.named_parameters())
    for name, p in tuned.named_parameters():
        if "lora_" in name:
            continue
        original = pristine[name.replace(".base.", ".")]
        assert torch.equal(p, original), f"Base weight {name} changed during a LoRA fine-tune. Freeze everything before injecting."
        assert not p.requires_grad, f"{name} is still trainable inside the LoRA copy."
    assert all(torch.count_nonzero(m.lora_B) > 0 for m in loras), "lora_B is still all zeros: the adapters were not trained."


def test_replay_copy_has_no_sigils_and_remembers(chronicler, outcomes):
    models, reports = outcomes
    tuned = models["full_lr_1e-3_replay_50"]
    assert not any(isinstance(m, sigil.LoRALinear) for m in tuned.modules()), "Replay fine-tuning is a FULL fine-tune on mixed batches; no adapters."
    rep, naive = reports["full_lr_1e-3_replay_50"], reports["full_lr_1e-3"]
    assert rep["new_after"] < 1.5, f"With half of every batch still ledger, the ledger should be learned (loss < 1.5); got {rep['new_after']:.2f}."
    assert rep["delta_old"] < naive["delta_old"] / 4, (
        f"Replaying the chronicles should cut forgetting several-fold: replay delta_old={rep['delta_old']:.2f} vs naive {naive['delta_old']:.2f}. "
        "Are the replayed rows really drawn from old_text, and do they make up replay_fraction of every batch?"
    )


# ------------------------------------------------------------ the prophecy
def _prediction(key):
    prophecy = room.FORGETTING_PROPHECY
    value = prophecy.get(key)
    if value is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in FORGETTING_PROPHECY.")
    return value


def test_the_prophecy_is_complete():
    prophecy = room.FORGETTING_PROPHECY
    assert set(prophecy) == {"forgets_most", "forgets_least", "lora_forgets_less_than_full_at_same_lr", "forgets"}, "Do not add or remove prophecy keys; fill them in."
    assert set(prophecy["forgets"]) == set(STRATEGIES), f"The 'forgets' dict names exactly these strategies: {STRATEGIES}."


def test_prophecy_who_forgets_most(outcomes):
    deltas = {k: r["delta_old"] for k, r in outcomes[1].items()}
    worst = max(deltas, key=deltas.get)
    said = _prediction("forgets_most")
    assert said == worst, f"The measured chronicles damage was {deltas}. The worst offender is {worst!r}; you prophesied {said!r}."


def test_prophecy_who_forgets_least(outcomes):
    deltas = {k: r["delta_old"] for k, r in outcomes[1].items()}
    best = min(deltas, key=deltas.get)
    said = _prediction("forgets_least")
    assert said == best, f"The measured chronicles damage was {deltas}. The gentlest is {best!r}; you prophesied {said!r}."


def test_prophecy_lora_versus_full_at_the_same_learning_rate(outcomes):
    deltas = {k: r["delta_old"] for k, r in outcomes[1].items()}
    truth = deltas["lora_r8_lr_1e-3"] < deltas["full_lr_1e-3"]
    said = _prediction("lora_forgets_less_than_full_at_same_lr")
    assert said == truth, (
        f"At lr 1e-3 for {STEPS} steps, LoRA raised the chronicles loss by {deltas['lora_r8_lr_1e-3']:.2f} and the full fine-tune by "
        f"{deltas['full_lr_1e-3']:.2f}. Moving 8% of the weights along a rank-8 subspace does less damage. You said {said}."
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_prophecy_does_it_forget(outcomes, strategy):
    said = room.FORGETTING_PROPHECY.get("forgets", {}).get(strategy)
    if said is None:
        raise NotImplementedError(f"You have not prophesied whether {strategy!r} forgets. Fill in FORGETTING_PROPHECY['forgets'].")
    delta = outcomes[1][strategy]["delta_old"]
    truth = delta > FORGETTING_THRESHOLD
    assert said == truth, (
        f"{strategy}: the chronicles loss rose by {delta:.2f} nats, which is {'more' if truth else 'less'} than the forgetting threshold "
        f"{FORGETTING_THRESHOLD}. You prophesied forgets={said}."
    )
