"""TRIAL 8.6 - THE PREFERENCE SCALE

Sequence log-probs that score only the answer, a DPO loss that starts at ln 2
and pushes chosen up and rejected down, and thirty steps that tip the scale
toward the true continuation without moving the reference or burning the book.
"""

import copy
import math

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn
F = torch.nn.functional

room = load_room(__file__, "room_6_the_preference_scale")
sigil = load_room(__file__, "room_2_the_low_rank_sigil")

STEPS = 30
BETA = 0.1
LR = 1e-3
SEED = 86
PROMPT_LEN = 32
RESPONSE_LEN = 32
N_TRAIN = 256
N_HELD = 64
MARGIN_BAR = 0.5  # mean implicit reward margin on held-out pairs; 0 at step 0, the reference lands at 1.4-1.9 across seeds
ACCURACY_BAR = 0.65  # fraction of held-out pairs with chosen reward > rejected reward; 0 at step 0, the reference 0.80-0.92
CHRONICLES_RISE_BUDGET = 0.5  # nats; the reference costs 0.06-0.17
CHOSEN_DROP_BUDGET = 20.0  # nats of summed chosen log-prob the policy may lose against the reference; the reference loses 2-11


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def chronicles():
    return read_corpus("chronicles")


@pytest.fixture(scope="module")
def preference_pairs(chronicler, chronicles):
    tokenizer = chronicler[1]
    train = room.make_preference_pairs(chronicles, tokenizer, N_TRAIN, PROMPT_LEN, RESPONSE_LEN, torch.Generator().manual_seed(SEED))
    held = room.make_preference_pairs(chronicles, tokenizer, N_HELD, PROMPT_LEN, RESPONSE_LEN, torch.Generator().manual_seed(SEED + 1))
    return train, held


@torch.no_grad()
def _manual_logprob(model, ids, prompt_len):
    """Trial-owned: an explicit loop over positions, one log_softmax at a time."""
    logits, _ = model(ids)
    out = []
    for b in range(ids.size(0)):
        total = 0.0
        for t in range(prompt_len, ids.size(1)):
            total += F.log_softmax(logits[b, t - 1], dim=-1)[ids[b, t]].item()
        out.append(total)
    return torch.tensor(out)


@torch.no_grad()
def _held_eval(policy, reference, pairs):
    was_training = policy.training
    policy.eval()
    pl = pairs["prompt_len"]
    pc, pr = _manual_logprob(policy, pairs["chosen"], pl), _manual_logprob(policy, pairs["rejected"], pl)
    rc, rr = _manual_logprob(reference, pairs["chosen"], pl), _manual_logprob(reference, pairs["rejected"], pl)
    margin = BETA * ((pc - rc) - (pr - rr))
    policy.train(was_training)
    return {"margin": margin.mean().item(), "accuracy": (margin > 0).float().mean().item(), "policy_chosen": pc.mean().item(), "ref_chosen": rc.mean().item()}


@torch.no_grad()
def _chronicles_loss(model, tokenizer, text, n_batches=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(data) - 65, (16,), generator=gen)
        x = torch.stack([data[i : i + 64] for i in ix])
        y = torch.stack([data[i + 1 : i + 65] for i in ix])
        total += model(x, y)[1].item()
    model.train(was_training)
    return total / n_batches


def _lora_policy(model):
    policy = copy.deepcopy(model)
    for p in policy.parameters():
        p.requires_grad_(False)
    sigil.inject_lora(policy, r=8, alpha=16.0)
    return policy


# ---------------------------------------------------------------- log-probs
def test_sequence_logprob_sums_only_the_response_positions(chronicler, chronicles):
    model, tokenizer = chronicler
    ids = torch.tensor(tokenizer.encode(chronicles[1000 : 1000 + 4 * 24]), dtype=torch.long).view(4, 24)
    got = room.sequence_logprob(model, ids, prompt_len=10)
    assert tuple(got.shape) == (4,), f"sequence_logprob returns one number per row, shape (B,) = (4,); got {tuple(got.shape)}."
    want = _manual_logprob(model, ids, 10)
    torch.testing.assert_close(got.detach(), want, rtol=1e-4, atol=1e-3, msg=lambda m: (
        "sequence_logprob must sum log p(ids[:, t] | ids[:, :t]) over t = prompt_len .. T-1 only. Position t-1 of the logits predicts ids[:, t]; "
        f"gather the log_softmax at the token that actually came next. {m}"
    ))
    assert torch.all(got <= 0), "Log-probabilities are never positive."
    last_only = room.sequence_logprob(model, ids, prompt_len=23)
    torch.testing.assert_close(last_only.detach(), _manual_logprob(model, ids, 23), rtol=1e-4, atol=1e-3, msg="With prompt_len = T-1 only the last token is scored.")
    assert not torch.allclose(got, last_only), "Different prompt_len must score different positions."


# ---------------------------------------------------------------- the loss
def test_dpo_loss_is_ln_2_when_the_scale_is_level():
    pc, rc = torch.tensor([-5.0, -7.0]), torch.tensor([-5.0, -7.0])
    pr, rr = torch.tensor([-9.0, -30.0]), torch.tensor([-9.0, -30.0])
    loss, chosen_reward, rejected_reward = room.dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6), f"Policy == reference means both implicit rewards are 0 and the loss is -log sigmoid(0) = ln 2 = 0.693; yours is {loss.item():.4f}."
    assert torch.all(chosen_reward == 0) and torch.all(rejected_reward == 0), "Implicit rewards are beta * (policy - reference): zero when nothing has moved."
    # Both margins moved by the same amount: still level.
    loss2, _c, _r = room.dpo_loss(pc + 3.0, pr + 3.0, rc, rr, beta=0.1)
    assert loss2.item() == pytest.approx(math.log(2), abs=1e-6), "Only the DIFFERENCE of the two margins matters: raising chosen and rejected equally leaves the loss at ln 2."


def test_dpo_loss_falls_as_the_chosen_margin_grows_and_beta_scales_it():
    rc = rr = torch.tensor([-10.0])
    pr = torch.tensor([-10.0])
    losses = [room.dpo_loss(rc + m, pr, rc, rr, beta=0.1)[0].item() for m in (0.0, 1.0, 2.0, 4.0, 8.0)]
    assert all(a > b for a, b in zip(losses, losses[1:])), f"The loss must fall monotonically as the chosen margin grows; got {[round(v, 4) for v in losses]}."
    assert losses[-1] == pytest.approx(-math.log(torch.sigmoid(torch.tensor(0.8)).item()), abs=1e-5), "loss = -log sigmoid(beta * margin): margin 8 at beta 0.1 gives -log sigmoid(0.8)."
    loss_a, cr_a, rr_a = room.dpo_loss(rc + 1.0, pr, rc, rr, beta=0.2)
    loss_b, cr_b, _ = room.dpo_loss(rc + 2.0, pr, rc, rr, beta=0.1)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6), "beta scales the logits: a margin of 1 at beta 0.2 equals a margin of 2 at beta 0.1."
    assert cr_a.item() == pytest.approx(0.2) and cr_b.item() == pytest.approx(0.2) and rr_a.item() == 0.0, "chosen_reward = beta * (policy_chosen - ref_chosen), rejected likewise."


def test_dpo_gradients_lift_the_chosen_and_sink_the_rejected():
    pc = torch.tensor([-6.0, -8.0], requires_grad=True)
    pr = torch.tensor([-20.0, -9.0], requires_grad=True)
    rc, rr = torch.tensor([-5.0, -8.0]), torch.tensor([-18.0, -9.0])
    loss, _c, _r = room.dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert loss.requires_grad, "The loss must stay attached to the policy's graph (do not detach it or call .item() inside)."
    loss.backward()
    assert torch.all(pc.grad < 0), f"d loss / d policy_chosen must be negative (gradient descent RAISES the chosen log-prob); got {pc.grad.tolist()}."
    assert torch.all(pr.grad > 0), f"d loss / d policy_rejected must be positive (descent LOWERS the rejected log-prob); got {pr.grad.tolist()}."
    margin = 0.1 * ((pc.detach() - rc) - (pr.detach() - rr))
    want = -0.1 * torch.sigmoid(-margin) / 2
    torch.testing.assert_close(pc.grad, want, rtol=1e-5, atol=1e-7, msg="Per pair the gradient is -beta * sigmoid(-beta * margin) / B: pairs the implicit reward gets wrong weigh more.")


# ---------------------------------------------------------------- the pairs
def test_preference_pairs_share_the_question_and_disagree_on_the_answer(chronicler, chronicles, preference_pairs):
    tokenizer = chronicler[1]
    train, _held = preference_pairs
    for key in ("chosen", "rejected", "prompt_len"):
        assert key in train, f"make_preference_pairs returns a dict with keys 'chosen', 'rejected', 'prompt_len'; missing {key!r}."
    chosen, rejected = train["chosen"], train["rejected"]
    length = PROMPT_LEN + RESPONSE_LEN
    assert tuple(chosen.shape) == (N_TRAIN, length) == tuple(rejected.shape), f"chosen and rejected are (n, prompt_len + response_len) = ({N_TRAIN}, {length}); got {tuple(chosen.shape)} and {tuple(rejected.shape)}."
    assert chosen.dtype == torch.long and rejected.dtype == torch.long and train["prompt_len"] == PROMPT_LEN
    assert torch.equal(chosen[:, :PROMPT_LEN], rejected[:, :PROMPT_LEN]), "chosen and rejected share the prompt: the first prompt_len tokens must be identical."
    assert all(not torch.equal(chosen[i, PROMPT_LEN:], rejected[i, PROMPT_LEN:]) for i in range(N_TRAIN)), "Every rejected response must differ from its chosen one (redraw when the second start lands on the true continuation)."
    for i in range(0, N_TRAIN, 37):
        assert tokenizer.decode(chosen[i].tolist()) in chronicles, f"Row {i} of chosen must be a contiguous window of the corpus (prompt + its true continuation)."
        assert tokenizer.decode(rejected[i, PROMPT_LEN:].tolist()) in chronicles, f"Row {i} of rejected must end with a real passage of the corpus, lifted from elsewhere."
    again = room.make_preference_pairs(chronicles, tokenizer, N_TRAIN, PROMPT_LEN, RESPONSE_LEN, torch.Generator().manual_seed(SEED))
    assert torch.equal(again["chosen"], chosen) and torch.equal(again["rejected"], rejected), "The same generator seed must give the same pairs."


def test_the_reference_is_a_frozen_copy_of_the_starting_model(chronicler):
    model = chronicler[0]
    reference = room.frozen_reference(model)
    assert reference is not model, "frozen_reference returns a COPY (copy.deepcopy); the policy will move and the reference must not follow."
    assert not reference.training, "The reference is in eval mode: no dropout noise in the anchor."
    assert all(not p.requires_grad for p in reference.parameters()), "Every reference parameter has requires_grad=False."
    assert all(torch.equal(p, q) and p.data_ptr() != q.data_ptr() for p, q in zip(reference.parameters(), model.parameters())), (
        "Reference tensors must equal the model's but not share its memory."
    )
    assert all(p.requires_grad for p in model.parameters()), "frozen_reference must not freeze the model you passed in."


# ---------------------------------------------------------------- the fight
@pytest.fixture(scope="module")
def scale(chronicler, preference_pairs):
    """A LoRA policy DPO-trained once against a frozen reference, for the whole file."""
    model = chronicler[0]
    torch.manual_seed(SEED)
    policy = _lora_policy(model)
    reference = room.frozen_reference(model)
    ref_before = {n: p.detach().clone() for n, p in reference.named_parameters()}
    history = room.train_dpo(policy, reference, preference_pairs[0], STEPS, BETA, LR, batch_size=8, generator=torch.Generator().manual_seed(SEED))
    return policy, reference, history, ref_before


def test_the_first_step_weighs_nothing(scale):
    _policy, _reference, history, _before = scale
    assert isinstance(history, list) and len(history) == STEPS, f"train_dpo returns one dict per step: {STEPS} of them."
    for key in ("loss", "chosen_reward", "rejected_reward", "margin", "reward_accuracy"):
        assert key in history[0] and isinstance(history[0][key], float), f"Each step's dict carries {key!r} as a Python float."
    assert history[0]["loss"] == pytest.approx(math.log(2), abs=1e-3), (
        f"At step 0 the policy IS the reference (lora_B is zero), so the first loss must be ln 2 = 0.6931; yours was {history[0]['loss']:.4f}. "
        "If it is not: either your sigils did not start at zero, or the policy ran with dropout ON (train mode). DPO trains with dropout off, "
        "because the loss is a difference of log-probs and noise in the policy alone looks like reward."
    )
    assert history[0]["margin"] == pytest.approx(0.0, abs=1e-4) and history[0]["reward_accuracy"] == 0.0, "At step 0 every implicit reward is 0: margin 0, accuracy 0 (ties are not wins)."


def test_the_scale_tips_toward_the_true_continuation(chronicler, preference_pairs, scale):
    policy, reference, history, _before = scale
    held = preference_pairs[1]
    level = _held_eval(chronicler[0], reference, held)
    assert level["margin"] == pytest.approx(0.0, abs=1e-4) and level["accuracy"] == 0.0, "(trial self-check) the pristine model against the reference is level."
    after = _held_eval(policy, reference, held)
    assert after["margin"] > MARGIN_BAR, (
        f"After {STEPS} DPO steps the mean implicit reward margin on {N_HELD} held-out pairs is {after['margin']:.3f}; the scale demands > {MARGIN_BAR} "
        "(reference 1.4-1.9). Check the sign of the loss, that the optimizer sees the sigils, and that the reference log-probs are the frozen model's."
    )
    assert after["accuracy"] > ACCURACY_BAR, (
        f"The policy's implicit reward prefers the true continuation on {after['accuracy']:.0%} of held-out pairs; the scale demands > {ACCURACY_BAR:.0%} (reference 80-92%)."
    )
    assert history[-1]["loss"] < history[0]["loss"], f"The training loss should fall from ln 2; it went {history[0]['loss']:.3f} -> {history[-1]['loss']:.3f}."


def test_the_reference_never_moved(scale):
    _policy, reference, _history, before = scale
    moved = [n for n, p in reference.named_parameters() if not torch.equal(p.detach(), before[n])]
    assert not moved, f"{len(moved)} reference tensors changed during DPO, e.g. {moved[:3]}. pi_ref is frozen: never hand it to the optimizer, never let it share tensors with the policy."
    assert all(p.grad is None for p in reference.parameters()), "The reference must be evaluated under torch.no_grad(): no .grad should exist on it."
    assert not reference.training


def test_the_policy_did_not_burn_the_book(chronicler, chronicles, preference_pairs, scale):
    model, tokenizer = chronicler
    policy, reference, _history, _before = scale
    base = _chronicles_loss(model, tokenizer, chronicles)
    after = _chronicles_loss(policy, tokenizer, chronicles)
    assert after - base <= CHRONICLES_RISE_BUDGET, (
        f"THE BOOK SMOULDERS. Chronicles loss went {base:.3f} -> {after:.3f} (+{after - base:.3f}), over the budget of {CHRONICLES_RISE_BUDGET}. "
        "DPO only trains the margin; pushing rejected text down (it is chronicles text too) can drag the whole distribution with it. Fewer steps, a lower lr or a larger beta."
    )
    held = _held_eval(policy, reference, preference_pairs[1])
    drop = held["ref_chosen"] - held["policy_chosen"]
    assert drop <= CHOSEN_DROP_BUDGET, (
        f"The summed log-prob of the CHOSEN continuations fell by {drop:.1f} nats ({held['ref_chosen']:.1f} -> {held['policy_chosen']:.1f}). A little of this is normal "
        f"(DPO trains only the margin, and the chosen log-prob often falls too); {CHOSEN_DROP_BUDGET} is the budget."
    )


def test_only_the_sigils_moved(chronicler, scale):
    pristine = dict(chronicler[0].named_parameters())
    policy = scale[0]
    for name, p in policy.named_parameters():
        if "lora_" in name:
            continue
        assert torch.equal(p, pristine[name.replace(".base.", ".")]), f"Base weight {name} changed during DPO; only the trainable sigils may move."
    assert any(torch.count_nonzero(m.lora_B) > 0 for m in policy.modules() if isinstance(m, sigil.LoRALinear)), "lora_B is still all zeros: the policy never trained."
    assert not policy.training, "train_dpo must restore the policy's mode when it is done."


def test_dpo_step_reports_its_bookkeeping(chronicler, preference_pairs):
    torch.manual_seed(SEED + 7)
    policy = _lora_policy(chronicler[0])
    reference = room.frozen_reference(chronicler[0])
    train = preference_pairs[0]
    batch = {"chosen": train["chosen"][:4], "rejected": train["rejected"][:4], "prompt_len": train["prompt_len"]}
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
    policy.eval()
    report = room.dpo_step(policy, reference, batch, BETA, opt)
    assert set(report) >= {"loss", "chosen_reward", "rejected_reward", "margin", "reward_accuracy"}, f"dpo_step returns a dict with loss, chosen_reward, rejected_reward, margin, reward_accuracy; got {sorted(report)}."
    assert report["loss"] == pytest.approx(math.log(2), abs=1e-3) and report["margin"] == pytest.approx(0.0, abs=1e-4), "The very first step is measured BEFORE the update: loss ln 2, margin 0."
    assert any(torch.count_nonzero(m.lora_B) > 0 for m in policy.modules() if isinstance(m, sigil.LoRALinear)), "dpo_step must call optimizer.step(): lora_B is still zero."
    second = room.dpo_step(policy, reference, batch, BETA, opt)
    assert second["margin"] > 0 and second["loss"] < report["loss"], f"On the same batch the second step should see a positive margin and a lower loss; got margin {second['margin']:.4f}, loss {second['loss']:.4f}."


def test_evaluate_preferences_agrees_with_the_trial(chronicler, preference_pairs, scale):
    policy, reference, _history, _before = scale
    held = preference_pairs[1]
    mine = room.evaluate_preferences(policy, reference, held, BETA)
    theirs = _held_eval(policy, reference, held)
    for key in ("margin", "reward_accuracy", "policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected"):
        assert key in mine and isinstance(mine[key], float), f"evaluate_preferences returns {key!r} as a Python float."
    assert mine["margin"] == pytest.approx(theirs["margin"], abs=0.02), f"evaluate_preferences margin {mine['margin']:.3f} vs the trial's {theirs['margin']:.3f} on the same pairs."
    assert mine["reward_accuracy"] == pytest.approx(theirs["accuracy"], abs=1e-6), "reward_accuracy is the fraction of pairs with chosen_reward > rejected_reward."
    assert mine["policy_chosen"] == pytest.approx(theirs["policy_chosen"], abs=0.05) and mine["ref_chosen"] == pytest.approx(theirs["ref_chosen"], abs=0.05)
    assert all(p.grad is None for p in reference.parameters()), "Evaluation must not leave gradients on the reference."
