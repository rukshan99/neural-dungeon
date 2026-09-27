"""TRIAL 8.5 - THE SCRIBE'S INSTRUCTION

A chat template the Chronicler's alphabet can spell, labels that are -100
everywhere except the answer, and a scribe that, after forty steps through
its sigils, answers price questions in the ledger's own words.
"""

import copy
import re

import pytest

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus
from dungeon.trials import load_room

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
nn = torch.nn
F = torch.nn.functional

room = load_room(__file__, "room_5_the_scribes_instruction")
sigil = load_room(__file__, "room_2_the_low_rank_sigil")

STEPS = 40
LR = 3e-3
SEED = 85
N_PAIRS = 400
N_TRAIN = 320
RESPONSE_LOSS_BAR = 1.0  # held-out loss per answer token; the pristine model scores ~6.3, the reference 0.55-0.61 across seeds (the floor is ~0.5: a price's digits are unpredictable)
UNITS = ("copper", "silver", "gold")
ITEM_LINE = re.compile(r"^item: (.+?) \| qty: (\d+) \| price: (\d+) (\w+) \| note: (.*)$")


@pytest.fixture(scope="module")
def chronicler():
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


@pytest.fixture(scope="module")
def ledger():
    return read_corpus("ledger")


@pytest.fixture(scope="module")
def pairs(ledger):
    all_pairs = room.make_instruction_pairs(ledger, N_PAIRS)
    return all_pairs[:N_TRAIN], all_pairs[N_TRAIN:]


def _manual_example(tokenizer, prompt, response):
    """Trial-owned rendering + masking, independent of build_sft_example."""
    prompt_ids = tokenizer.encode(room.render_prompt(prompt))
    ids = prompt_ids + tokenizer.encode(response + room.END_TAG)
    labels = [-100] * len(ids)
    for t in range(len(ids) - 1):
        if t + 1 >= len(prompt_ids):
            labels[t] = ids[t + 1]
    return torch.tensor(ids), torch.tensor(labels), len(prompt_ids)


@torch.no_grad()
def _response_loss(model, tokenizer, some_pairs):
    """Mean cross-entropy per answer token over all pairs, computed by the trial."""
    was_training = model.training
    model.eval()
    examples = [_manual_example(tokenizer, p, r)[:2] for p, r in some_pairs]
    longest = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), longest), tokenizer.unk_id, dtype=torch.long)
    y = torch.full((len(examples), longest), -100, dtype=torch.long)
    for row, (ids, lab) in enumerate(examples):
        x[row, : len(ids)] = ids
        y[row, : len(lab)] = lab
    logits, _ = model(x)
    total = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum").item()
    model.train(was_training)
    return total / int((y != -100).sum())


# ---------------------------------------------------------------- template
def test_the_tags_can_be_spelled_by_the_chronicler(chronicler):
    tokenizer = chronicler[1]
    for name in ("USER_TAG", "ASSISTANT_TAG", "END_TAG"):
        tag = getattr(room, name)
        assert isinstance(tag, str) and tag, f"{name} must be a non-empty string."
        missing = sorted({ch for ch in tag if ch not in tokenizer.stoi})
        assert not missing, (
            f"{name} = {tag!r} uses characters the Chronicler cannot spell: {missing}. Its 72-character alphabet has no special tokens "
            f"(and no '#'); every character of a tag must be in tokenizer.stoi or encode() silently turns it into a newline."
        )
    assert room.USER_TAG != room.ASSISTANT_TAG, "The two role tags must differ, or the model cannot tell who is speaking."
    q, a = "Quote the price of salt.", "39 silver"
    rendered = room.render_prompt(q)
    assert rendered.startswith(room.USER_TAG) and q in rendered, f"render_prompt must open with USER_TAG and contain the question; got {rendered!r}."
    assert rendered.endswith(room.ASSISTANT_TAG), (
        f"render_prompt must END with ASSISTANT_TAG so that the model's continuation is the assistant's answer; got {rendered!r}."
    )
    assert room.render_chat([("user", q), ("assistant", a)]) == rendered + a + room.END_TAG, (
        "render_chat([(user, q), (assistant, a)]) must equal render_prompt(q) + a + END_TAG: the same template at training and inference, "
        "character for character."
    )
    assert room.render_chat([]) == "", "An empty conversation renders as the empty string."
    assert tokenizer.decode(tokenizer.encode(rendered)) == rendered, "The rendered template must survive encode/decode unchanged."
    with pytest.raises(ValueError):
        room.render_chat([("goblin", "mine")])


# ---------------------------------------------------------------- masking
def test_labels_are_the_next_token_over_the_answer_and_minus_100_over_the_question(chronicler):
    tokenizer = chronicler[1]
    q, a = "Quote the price of lamp oil.", "61 copper"
    ids, labels = room.build_sft_example(q, a, tokenizer, max_len=128)
    want_ids, want_labels, n_prompt = _manual_example(tokenizer, q, a)
    assert ids.dtype == torch.long and labels.dtype == torch.long, "input_ids and labels are int64 tensors."
    assert ids.shape == labels.shape == want_ids.shape, f"input_ids and labels are both length T = {len(want_ids)}; got {tuple(ids.shape)} and {tuple(labels.shape)}."
    assert torch.equal(ids, want_ids), "input_ids must encode render_prompt(q) + a + END_TAG, nothing more and nothing less."
    assert torch.all(labels[: n_prompt - 1] == -100), (
        f"Positions whose TARGET is still a prompt token must be labelled -100; the first {n_prompt - 1} positions are, and yours has "
        f"{int((labels[: n_prompt - 1] != -100).sum())} unmasked among them."
    )
    assert labels[-1] == -100, "The last position has no next token: its label is -100."
    assert torch.equal(labels[n_prompt - 1 : -1], ids[n_prompt:]), (
        "Over the answer, labels[t] must be input_ids[t + 1] (the next token). Position n_prompt - 1, the last character of the assistant "
        "tag, predicts the FIRST answer character and is supervised."
    )
    targets = tokenizer.decode(labels[labels != -100].tolist())
    assert targets == a + room.END_TAG, f"The supervised labels should decode to the answer plus END_TAG, {a + room.END_TAG!r}; yours decode to {targets!r}."
    assert room.supervised_fraction(labels) == pytest.approx(len(tokenizer.encode(a + room.END_TAG)) / len(ids)), "supervised_fraction = (#labels != -100) / T."


def test_truncation_keeps_the_answer_and_beheads_the_question(chronicler):
    tokenizer = chronicler[1]
    q, a = "Quote the price of rope (128 knots).", "50 gold"
    full_ids, _full_labels, _n = _manual_example(tokenizer, q, a)
    max_len = len(full_ids) - 7
    ids, labels = room.build_sft_example(q, a, tokenizer, max_len=max_len)
    assert len(ids) == max_len == len(labels), f"A too-long example is truncated to max_len={max_len}; yours has {len(ids)} ids and {len(labels)} labels."
    assert torch.equal(ids, full_ids[7:]), "Truncate from the FRONT (the prompt side): the answer and its end tag must survive whole, the question loses its head."
    assert tokenizer.decode(labels[labels != -100].tolist()) == a + room.END_TAG, "After truncation the supervised labels must still decode to the whole answer + END_TAG."
    with pytest.raises(ValueError):
        room.build_sft_example(q, a, tokenizer, max_len=len(tokenizer.encode(a + room.END_TAG)))


def test_collate_pads_on_the_right_and_ignores_the_padding(chronicler):
    tokenizer = chronicler[1]
    short = room.build_sft_example("Salt.", "39 silver", tokenizer, 128)
    long = room.build_sft_example("Quote the price of goblin biscuit.", "27 copper", tokenizer, 128)
    pad_id = tokenizer.stoi["|"]
    x, y = room.collate_sft([short, long], pad_id=pad_id)
    t_max = len(long[0])
    assert tuple(x.shape) == (2, t_max) and tuple(y.shape) == (2, t_max), f"Collated shapes are (B, T_max) = (2, {t_max}); got {tuple(x.shape)} and {tuple(y.shape)}."
    assert x.dtype == torch.long and y.dtype == torch.long
    n_short = len(short[0])
    assert torch.equal(x[0, :n_short], short[0]) and torch.equal(x[1], long[0]), "Real tokens must be copied unchanged, starting at position 0 (right-padding)."
    assert torch.all(x[0, n_short:] == pad_id), f"Padding in input_ids must be pad_id={pad_id}."
    assert torch.equal(y[0, :n_short], short[1]) and torch.all(y[0, n_short:] == -100), "Padding in labels must be -100: a pad is never a target."


def test_sft_loss_scores_only_the_supervised_positions():
    torch.manual_seed(5)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[-100, -100, 3, 1, -100], [-100, 4, 4, -100, -100]])
    loss = room.sft_loss(logits, labels)
    logp = F.log_softmax(logits, dim=-1)
    picked = [-logp[b, t, labels[b, t]] for b in range(2) for t in range(5) if labels[b, t] != -100]
    want = torch.stack(picked).mean()
    torch.testing.assert_close(loss, want, rtol=1e-5, atol=1e-6, msg=lambda m: f"sft_loss must be the MEAN cross-entropy over the {len(picked)} supervised positions (not over all 10). {m}")
    perturbed = logits.clone()
    perturbed[labels == -100] += 100.0
    assert room.sft_loss(perturbed, labels) == pytest.approx(loss.item(), abs=1e-5), "Changing the logits at masked positions must not change the loss at all."
    assert room.supervised_fraction(labels) == pytest.approx(4 / 10), f"4 of 10 positions carry a label; supervised_fraction says {room.supervised_fraction(labels)}."


# ---------------------------------------------------------------- the dataset
def test_the_instruction_pairs_are_read_off_the_ledger(chronicler, ledger, pairs):
    tokenizer = chronicler[1]
    train, held = pairs
    every = train + held
    assert len(every) == N_PAIRS, f"make_instruction_pairs(ledger, {N_PAIRS}) should return {N_PAIRS} pairs; got {len(every)}."
    expected = [m.groups() for m in (ITEM_LINE.match(line) for line in ledger.splitlines()) if m]
    for i, ((prompt, response), (item, _qty, price, unit, _note)) in enumerate(zip(every, expected)):
        assert isinstance(prompt, str) and isinstance(response, str), "Each pair is (prompt: str, response: str)."
        assert item in prompt, f"Pair {i}: the prompt {prompt!r} should ask about item {item!r} (ledger item lines, in order; skip the header and day separators)."
        assert response == f"{price} {unit}", f"Pair {i}: the response should be '{price} {unit}' (PRICE_ANSWER); got {response!r}."
        assert tokenizer.decode(tokenizer.encode(prompt)) == prompt, (
            f"Pair {i}: the prompt {prompt!r} uses a character the Chronicler cannot spell (there is no '?' in its alphabet); encode() would silently turn it into a newline."
        )
    assert room.make_instruction_pairs(ledger, N_PAIRS) == every, "make_instruction_pairs must be deterministic."
    assert len(room.make_instruction_pairs(ledger, 10_000)) == len(expected), f"Asking for more pairs than there are item lines returns all {len(expected)} of them."


# ---------------------------------------------------------------- the scribe
@pytest.fixture(scope="module")
def scribe(chronicler, pairs):
    """A frozen Chronicler with LoRA r=8, instruction-tuned once for the whole file."""
    model, tokenizer = copy.deepcopy(chronicler[0]), chronicler[1]
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(SEED)
    sigil.inject_lora(model, r=8, alpha=16.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    losses = room.finetune_sft(model, pairs[0], tokenizer, steps=STEPS, lr=LR, generator=torch.Generator().manual_seed(SEED))
    return model, losses, before


def test_the_scribe_learns_to_answer_in_forty_steps(chronicler, pairs, scribe):
    model, losses, _before = scribe
    tokenizer = chronicler[1]
    held = pairs[1]
    assert isinstance(losses, list) and len(losses) == STEPS and all(isinstance(v, float) for v in losses), f"finetune_sft returns {STEPS} Python floats, one per step."
    assert losses[0] > 4.0, f"The first step's loss should be the pristine model's ~6.3 on answer tokens; yours was {losses[0]:.2f}. Are the labels masked to the answer?"
    before = _response_loss(chronicler[0], tokenizer, held)
    after = _response_loss(model, tokenizer, held)
    assert before > 4.0, f"(trial self-check) the pristine Chronicler should be lost on held-out answers (~6.3); measured {before:.2f}."
    assert after < RESPONSE_LOSS_BAR, (
        f"After {STEPS} SFT steps the held-out loss per answer token is {after:.2f} (from {before:.2f}); the scribe demands < {RESPONSE_LOSS_BAR} "
        f"(reference ~0.6; the floor is ~0.5 because a price's digits are unpredictable). Check the mask, the optimizer step and the lr."
    )


def test_the_frozen_base_did_not_move_and_the_mode_is_restored(scribe):
    model, _losses, before = scribe
    moved = [n for n, p in model.named_parameters() if n in before and not torch.equal(p.detach(), before[n])]
    assert not moved, f"{len(moved)} frozen base tensors changed during SFT, e.g. {moved[:3]}. Hand AdamW only the parameters with requires_grad=True."
    assert not model.training, "The model was in eval mode before finetune_sft(); restore it afterwards."


def test_evaluate_sft_agrees_with_the_trial(chronicler, pairs, scribe):
    model = scribe[0]
    tokenizer = chronicler[1]
    mine = room.evaluate_sft(model, pairs[1], tokenizer)
    theirs = _response_loss(model, tokenizer, pairs[1])
    assert isinstance(mine, float), "evaluate_sft returns a Python float."
    assert mine == pytest.approx(theirs, abs=0.05), (
        f"evaluate_sft says {mine:.3f}, the trial measures {theirs:.3f} on the same pairs. Sum the token losses over every pair and divide by the number of "
        "supervised labels; do not average over padding or prompt positions."
    )
    pristine = copy.deepcopy(chronicler[0])
    room.evaluate_sft(pristine, pairs[1][:8], tokenizer)
    assert all(p.grad is None for p in pristine.parameters()), "evaluate_sft must run under torch.no_grad() and leave no gradients behind."
    assert not pristine.training, "evaluate_sft must leave the model in the mode it found it."


def test_the_answer_has_the_shape_of_a_price(chronicler, pairs, scribe):
    model = scribe[0]
    tokenizer = chronicler[1]
    for prompt, _response in pairs[0][:3]:
        text = room.answer(model, tokenizer, prompt, max_new_tokens=24)
        assert isinstance(text, str), "answer() returns the generated text as a str."
        assert room.END_TAG not in text and room.USER_TAG not in text, f"answer() must stop at END_TAG and return the text before it; got {text!r}."
        assert len(text) < 24, f"Generation should stop at END_TAG well before the budget of 24 tokens; got {len(text)} characters: {text!r}."
        assert re.search(r"\d", text) and any(unit in text for unit in UNITS), (
            f"To {prompt!r} the scribe answered {text!r}. After SFT a greedy answer should look like a price: digits and one of {UNITS}. "
            "Check that answer() continues render_prompt(prompt) with the SAME template used in training."
        )
    assert not model.training, "answer() must leave the model in the mode it found it."
