"""BOSS FIGHT - THE STUTTERING SOVEREIGN

Phase 1: the instruments - distinct n-gram ratio, repetition penalty,
         no-repeat-n-gram blocking - on hand-made sequences and on the
         Sovereign's own greedy stutter.
Phase 2: mean_nll - the fluency judge - checked against cross-entropy.
Phase 3: speak_without_stuttering must satisfy BOTH judges on three seeds.
Phase 4: the prophecy - four decoding strategies, predicted then measured.
"""

import math
from functools import lru_cache

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch.nn.functional as F  # noqa: E402

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

boss = load_room(__file__, "boss_stuttering_sovereign")

pytestmark = pytest.mark.boss

torch.manual_seed(77)

MODEL, TOK, _ = tiny_gpt.load_pretrained()
PROMPT = "\n"
PROMPT_IDS = torch.tensor([TOK.encode(PROMPT)], dtype=torch.long)

# The two judges. Greedy decoding scores about 0.23 on diversity; sensible
# sampling scores 0.85-0.95; uniform noise scores 1.0 on diversity but about 11
# on NLL; temperature 2.0 lands at 1.2-2.0 NLL. The lines are drawn between.
DISTINCT_4_MIN = 0.6
NLL_MAX = 1.0

PROPHECY_KEYS = ["greedy", "temperature_0.7_top_k_40", "temperature_2.0", "greedy_no_repeat_4gram"]
PROPHECY_TOKENS = 200
PROPHECY_SEEDS = (1, 2, 3)


# ------------------------------------------------------- the trial's own judges
def _distinct(ids, n):
    ids = [int(t) for t in ids]
    total = len(ids) - n + 1
    if total <= 0:
        return 0.0
    return len({tuple(ids[i : i + n]) for i in range(total)}) / total


@torch.no_grad()
def _nll(ids):
    ids = torch.as_tensor(ids, dtype=torch.long).flatten()
    block = MODEL.cfg.block_size
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, block):
        y = ids[start + 1 : start + 1 + block]
        x = ids[start : start + len(y)]
        logits, _ = MODEL(x[None])
        total += F.cross_entropy(logits[0], y, reduction="sum").item()
        count += len(y)
    return total / count


def _block(logits, generated, n):
    out = logits.clone()
    seq = generated[0].tolist()
    if len(seq) >= n:
        prefix = tuple(seq[len(seq) - (n - 1) :]) if n > 1 else ()
        banned = {seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix}
        if banned:
            out[0, list(banned)] = float("-inf")
    return out


@torch.no_grad()
def _decode(n_tokens, seed=0, temperature=0.0, top_k=None, no_repeat=None):
    """The trial's own decoder: greedy, or temperature + top-k sampling, optionally with n-gram blocking."""
    g = torch.Generator().manual_seed(seed)
    idx = PROMPT_IDS
    for _ in range(n_tokens):
        logits, _ = MODEL(idx[:, -MODEL.cfg.block_size :])
        logits = logits[:, -1, :]
        if no_repeat:
            logits = _block(logits, idx, no_repeat)
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g).squeeze(-1)
        idx = torch.cat([idx, nxt[:, None]], 1)
    return idx[0]


@lru_cache(maxsize=None)
def _greedy_300():
    return _decode(300)


def _verdict(distinct, nll):
    if distinct < DISTINCT_4_MIN:
        return "stutters"
    if nll > NLL_MAX:
        return "babbles"
    return "speaks"


# ------------------------------------------------------------------- phase 1
def test_phase_1_distinct_ngram_ratio_on_sequences_you_can_count_by_hand():
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert math.isclose(boss.distinct_ngram_ratio(seq, 2), 3 / 8), (
        f"{seq} has 8 bigrams of which 3 are distinct: 0.375. You said {boss.distinct_ngram_ratio(seq, 2)}."
    )
    assert math.isclose(boss.distinct_ngram_ratio(seq, 3), 3 / 7), "7 trigrams, 3 distinct: 3/7."
    assert math.isclose(boss.distinct_ngram_ratio(list(range(10)), 4), 1.0), "A sequence with no repeats scores 1.0."
    assert boss.distinct_ngram_ratio([1, 2], 4) == 0.0, "A sequence shorter than n has no n-grams: return 0.0, not a division by zero."
    assert math.isclose(boss.distinct_ngram_ratio(torch.tensor(seq), 2), 3 / 8), "Accept a 1-D tensor as well as a list."


def test_phase_1_the_ratio_exposes_the_sovereigns_stutter():
    ids = _greedy_300()
    ratio = boss.distinct_ngram_ratio(ids, 4)
    assert math.isclose(ratio, _distinct(ids, 4), abs_tol=1e-9), f"Your ratio {ratio:.4f} disagrees with the trial's {_distinct(ids, 4):.4f} on the same sequence."
    assert ratio < DISTINCT_4_MIN, (
        f"Dungeon self-check: greedy decoding should stutter (ratio {ratio:.3f} >= {DISTINCT_4_MIN}). "
        "If this fails, the checkpoint has changed; re-calibrate the boss."
    )


def test_phase_1_repetition_penalty_divides_the_positive_and_multiplies_the_negative():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0], [2.0, -1.0, 0.5, 3.0]])
    history = torch.tensor([[0, 1, 1], [3, 3, 3]])
    out = boss.repetition_penalty(logits, history, 2.0)
    assert out.shape == logits.shape
    assert torch.allclose(out[0], torch.tensor([1.0, -2.0, 0.5, 3.0])), (
        f"Row 0 has seen tokens 0 and 1: logit 2.0 -> 1.0 (divided), -1.0 -> -2.0 (multiplied), the rest untouched. Got {out[0].tolist()}."
    )
    assert torch.allclose(out[1], torch.tensor([2.0, -1.0, 0.5, 1.5])), f"Row 1 has only seen token 3: 3.0 -> 1.5. Got {out[1].tolist()}."
    assert torch.equal(logits, torch.tensor([[2.0, -1.0, 0.5, 3.0], [2.0, -1.0, 0.5, 3.0]])), "Do not modify the logits in place; return a new tensor."
    assert torch.allclose(boss.repetition_penalty(logits, history, 1.0), logits), "penalty=1.0 changes nothing."


def test_phase_1_block_repeated_ngrams_bans_exactly_the_completions_seen_before():
    logits = torch.zeros(3, 6)
    history = torch.tensor([[1, 2, 3, 4, 1, 2, 3], [0, 5, 0, 5, 0, 5, 0], [1, 2, 3, 4, 5, 0, 1]])
    out = boss.block_repeated_ngrams(logits, history, 4)
    assert torch.isinf(out[0, 4]) and out[0, 4] < 0, "Row 0 ends in (1, 2, 3), which was followed by 4 before: token 4 must be -inf."
    assert torch.all(out[0, [0, 1, 2, 3, 5]] == 0), f"Row 0 should ban ONLY token 4; got {out[0].tolist()}."
    assert torch.isinf(out[1, 5]) and torch.all(out[1, [0, 1, 2, 3, 4]] == 0), f"Row 1 ends in (0, 5, 0), which was followed by 5 both times it appeared earlier: only token 5 is banned. Got {out[1].tolist()}."
    assert torch.all(out[2] == 0), f"Row 2's last 3 tokens (5, 0, 1) never appeared before: nothing is banned. Got {out[2].tolist()}."
    assert torch.all(logits == 0), "Do not modify the logits in place."
    short = boss.block_repeated_ngrams(torch.zeros(1, 6), torch.tensor([[1, 2]]), 4)
    assert torch.all(short == 0), "A history shorter than n cannot contain a repeated n-gram: nothing is banned."
    unigram = boss.block_repeated_ngrams(torch.zeros(1, 6), torch.tensor([[1, 3, 3]]), 1)
    assert torch.isinf(unigram[0, 1]) and torch.isinf(unigram[0, 3]) and torch.all(unigram[0, [0, 2, 4, 5]] == 0), "n=1 bans every token already generated."


def test_phase_1_blocking_makes_a_loop_impossible():
    idx = PROMPT_IDS
    with torch.no_grad():
        for _ in range(150):
            logits, _ = MODEL(idx[:, -MODEL.cfg.block_size :])
            logits = boss.block_repeated_ngrams(logits[:, -1, :], idx, 4)
            idx = torch.cat([idx, logits.argmax(-1)[:, None]], 1)
    ratio = _distinct(idx[0], 4)
    assert ratio == 1.0, (
        f"With repeated 4-grams blocked, every 4-gram in the output must be new, so the distinct-4-gram ratio is exactly 1.0; got {ratio:.3f}. "
        "The prefix is the LAST n-1 generated tokens; ban every token that followed that prefix earlier."
    )


# ------------------------------------------------------------------- phase 2
def test_phase_2_mean_nll_is_cross_entropy_under_teacher_forcing():
    ids = torch.tensor(TOK.encode(tiny_gpt.read_corpus("chronicles")[1000:1060]))
    got = boss.mean_nll(MODEL, ids)
    with torch.no_grad():
        logits, _ = MODEL(ids[None, :-1])
        want = F.cross_entropy(logits[0], ids[1:]).item()
    assert isinstance(got, float), f"mean_nll returns a Python float, got {type(got).__name__}."
    assert abs(got - want) < 1e-4, (
        f"mean_nll gave {got:.5f}; feeding ids[:-1] and scoring ids[1:] with cross-entropy gives {want:.5f}. "
        "Every token except the first is a target; the first is only ever context."
    )


def test_phase_2_mean_nll_walks_a_long_text_in_windows():
    ids = torch.tensor(TOK.encode(tiny_gpt.read_corpus("chronicles")[5000:5301]))
    got = boss.mean_nll(MODEL, ids)
    want = _nll(ids)
    assert abs(got - want) < 1e-4, (
        f"On a 301-token text (longer than block_size=128) mean_nll gave {got:.5f}; the windowed contract gives {want:.5f}. "
        "Windows are consecutive and non-overlapping: tokens [wB, wB+B) predict tokens [wB+1, wB+B+1). Divide by len(ids) - 1."
    )
    assert got < 0.6, f"Dungeon self-check: the Chronicler should find its own chronicles fluent (NLL {got:.3f})."
    noise = torch.randint(0, TOK.vocab_size, (301,), generator=torch.Generator().manual_seed(0))
    assert boss.mean_nll(MODEL, noise) > 5.0, "Uniform noise should be very unlikely under the Chronicler."


def test_phase_2_mean_nll_leaves_no_gradients_behind():
    MODEL.zero_grad(set_to_none=True)
    boss.mean_nll(MODEL, torch.tensor(TOK.encode("The chronicle records that")))
    assert all(p.grad is None for p in MODEL.parameters()), "mean_nll must run under torch.no_grad(): scoring is not training."


# ------------------------------------------------------------------- phase 3
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phase_3_the_sovereign_speaks_without_stuttering(seed):
    text = boss.speak_without_stuttering(MODEL, TOK, PROMPT, n_tokens=300, seed=seed)
    assert isinstance(text, str), f"speak_without_stuttering returns the decoded string, got {type(text).__name__}."
    assert text.startswith(PROMPT), "The returned text starts with the prompt."
    assert len(text) == len(PROMPT) + 300, f"Prompt plus exactly 300 generated characters: expected length {len(PROMPT) + 300}, got {len(text)}."
    ids = TOK.encode(text)
    distinct = _distinct(ids, 4)
    nll = _nll(ids)
    assert distinct >= DISTINCT_4_MIN, (
        f"[seed {seed}] The Sovereign STUTTERS: distinct-4-gram ratio {distinct:.3f} < {DISTINCT_4_MIN}.\n{text[:200]!r}\n"
        "Greedy decoding on a confident model falls into loops. Sample, or block repeated n-grams, or penalise repeats."
    )
    assert nll <= NLL_MAX, (
        f"[seed {seed}] The Sovereign BABBLES: mean NLL {nll:.3f} > {NLL_MAX} under its own model.\n{text[:200]!r}\n"
        "Diversity is cheap if you stop making sense. Lower the temperature, or keep only the nucleus (top-p / top-k)."
    )


def test_phase_3_the_judges_cannot_be_fooled_by_noise():
    noise = torch.randint(0, TOK.vocab_size, (301,), generator=torch.Generator().manual_seed(1))
    assert _distinct(noise, 4) >= DISTINCT_4_MIN, "Dungeon self-check: noise is diverse."
    assert _verdict(_distinct(noise, 4), _nll(noise)) == "babbles", "Dungeon self-check: noise must fail the fluency judge."


# ------------------------------------------------------------------- phase 4
def test_phase_4_the_prophecy_names_every_strategy():
    assert set(boss.SOVEREIGN_PROPHECY) == set(PROPHECY_KEYS), "Do not add or remove strategies from SOVEREIGN_PROPHECY; predict them."


def _measure(strategy):
    if strategy == "greedy":
        runs = [_decode(PROPHECY_TOKENS)]
    elif strategy == "temperature_0.7_top_k_40":
        runs = [_decode(PROPHECY_TOKENS, seed=s, temperature=0.7, top_k=40) for s in PROPHECY_SEEDS]
    elif strategy == "temperature_2.0":
        runs = [_decode(PROPHECY_TOKENS, seed=s, temperature=2.0) for s in PROPHECY_SEEDS]
    elif strategy == "greedy_no_repeat_4gram":
        runs = [_decode(PROPHECY_TOKENS, no_repeat=4)]
    else:  # pragma: no cover - keys are fixed above
        raise KeyError(strategy)
    distinct = sum(_distinct(r, 4) for r in runs) / len(runs)
    nll = sum(_nll(r) for r in runs) / len(runs)
    return distinct, nll, TOK.decode(runs[0][:80].tolist())


@pytest.mark.parametrize("strategy", PROPHECY_KEYS)
def test_phase_4_the_prophecy_is_measured(strategy):
    prediction = boss.SOVEREIGN_PROPHECY.get(strategy)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied how '{strategy}' fares. Fill in SOVEREIGN_PROPHECY.")
    assert prediction in ("stutters", "babbles", "speaks"), f"Predictions are one of 'stutters', 'babbles', 'speaks'; got {prediction!r}."
    distinct, nll, sample = _measure(strategy)
    verdict = _verdict(distinct, nll)
    assert prediction == verdict, (
        f"You prophesied that '{strategy}' {prediction}; measured over {PROPHECY_TOKENS} tokens it {verdict}: "
        f"distinct-4-gram ratio {distinct:.3f} (needs >= {DISTINCT_4_MIN}), mean NLL {nll:.3f} (needs <= {NLL_MAX}).\n"
        f"It began: {sample!r}"
    )
