"""TRIAL 7.5 - THE VOICE

sample_next is judged on hand-made logits where the right answer is known:
greedy is the argmax, top-k keeps k, top-p keeps exactly the nucleus, the
generator makes it reproducible. generate is judged on cropping, on passing
the options through, and on reproducing the Chronicler's greedy line.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts import tiny_gpt  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_5_the_voice")

torch.manual_seed(75)

# A known distribution, deliberately out of order so that top-p has to sort:
# token 2 has p=0.5, token 0 has 0.3, token 3 has 0.15, token 1 has 0.05.
PROBS = torch.tensor([0.30, 0.05, 0.50, 0.15])
LOGITS = PROBS.log()


def _sampled_set(logits_row, rows=600, seed=0, **kw):
    logits = logits_row.expand(rows, -1).clone()
    out = room.sample_next(logits, generator=torch.Generator().manual_seed(seed), **kw)
    assert out.shape == (rows,) and out.dtype == torch.long, f"sample_next returns (B,) int64 ids; got {tuple(out.shape)} {out.dtype}."
    return set(out.tolist())


# ---------------------------------------------------------------- sample_next
def test_temperature_zero_is_the_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, -1.0], [0.0, 0.0, 0.5]])
    out = room.sample_next(logits, temperature=0.0)
    assert out.shape == (3,) and out.dtype == torch.long, f"Expected (3,) int64, got {tuple(out.shape)} {out.dtype}."
    assert out.tolist() == [1, 0, 2], f"Greedy decoding picks the largest logit in every row: [1, 0, 2], got {out.tolist()}."
    assert room.sample_next(logits, temperature=-1.0).tolist() == [1, 0, 2], "Any temperature <= 0 means greedy."


def test_top_k_one_is_greedy_no_matter_the_dice():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.9]])
    for seed in range(5):
        out = room.sample_next(logits, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(seed))
        assert out.tolist() == [1], f"With top_k=1 only the argmax survives; seed {seed} produced {out.tolist()}."


def test_top_k_keeps_exactly_the_k_largest():
    logits = torch.tensor([5.0, 5.0, 5.0, 0.0, -1.0, -2.0])
    assert _sampled_set(logits, top_k=3) == {0, 1, 2}, "top_k=3 must keep tokens 0, 1, 2 (the three largest logits) and nothing else."
    flat = torch.zeros(6)
    assert _sampled_set(flat, top_k=100) == {0, 1, 2, 3, 4, 5}, "top_k larger than the vocabulary keeps everything: use min(top_k, V)."


@pytest.mark.parametrize(
    "p,expected",
    [(0.40, {2}), (0.70, {2, 0}), (0.85, {2, 0, 3}), (0.96, {2, 0, 3, 1})],
    ids=["p=0.40", "p=0.70", "p=0.85", "p=0.96"],
)
def test_top_p_keeps_the_smallest_nucleus_that_reaches_p(p, expected):
    got = _sampled_set(LOGITS, top_p=p)
    assert got == expected, (
        f"Probabilities sorted: token 2 (0.50), token 0 (0.30), token 3 (0.15), token 1 (0.05). "
        f"With top_p={p} the nucleus is {sorted(expected)} (keep tokens until the cumulative mass reaches {p}), "
        f"but you sampled from {sorted(got)}."
    )


def test_top_p_always_keeps_at_least_the_best_token():
    assert _sampled_set(LOGITS, top_p=0.01) == {2}, "Even with a tiny p the most probable token must survive, or nothing can be sampled."
    assert _sampled_set(LOGITS, top_p=1.0) == {0, 1, 2, 3}, "top_p=1.0 keeps everything."


def test_temperature_reshapes_the_distribution():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    rows = 3000
    cold = room.sample_next(logits.expand(rows, -1).clone(), temperature=0.3, generator=torch.Generator().manual_seed(1))
    hot = room.sample_next(logits.expand(rows, -1).clone(), temperature=5.0, generator=torch.Generator().manual_seed(1))
    cold_share = (cold == 0).float().mean().item()
    hot_share = (hot == 0).float().mean().item()
    assert cold_share > 0.9, f"At temperature 0.3 the top token should win about 96% of draws; it won {cold_share:.1%}. Divide the logits by the temperature."
    assert hot_share < 0.45, f"At temperature 5.0 the top token should win only about 33% of draws; it won {hot_share:.1%}. Higher temperature = flatter distribution."


def test_the_dice_are_reproducible_with_a_generator():
    logits = torch.randn(64, 20)
    a = room.sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(3))
    b = room.sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(3))
    c = room.sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b), "The same generator seed must give the same tokens: pass generator= to torch.multinomial."
    assert not torch.equal(a, c), "Different seeds gave identical tokens; the generator is not being used."


def test_minus_infinity_is_never_drawn():
    logits = torch.tensor([[0.0, float("-inf"), 0.0, float("-inf")]])
    got = _sampled_set(logits[0], temperature=1.0)
    assert got == {0, 2}, f"Tokens with logit -inf have probability exactly 0 and must never appear; sampled {sorted(got)}."


# ------------------------------------------------------------------- generate
def _tiny_model(block_size=8, seed=0):
    torch.manual_seed(seed)
    cfg = tiny_gpt.GPTConfig(vocab_size=13, block_size=block_size, n_layer=1, n_head=2, n_embd=16)
    return tiny_gpt.GPT(cfg).eval()


def test_generate_appends_exactly_max_new_tokens_and_keeps_the_prompt():
    model = _tiny_model()
    prompt = torch.randint(0, 13, (2, 3))
    out = room.generate(model, prompt, 5, temperature=0.0)
    assert out.shape == (2, 8), f"(B, T0) -> (B, T0 + max_new_tokens) = (2, 8); got {tuple(out.shape)}."
    assert out.dtype == torch.long
    assert torch.equal(out[:, :3], prompt), "The prompt must be preserved at the front; new tokens are appended after it."


def test_generate_crops_the_context_to_block_size():
    model = _tiny_model(block_size=8)
    prompt = torch.randint(0, 13, (1, 4))
    try:
        out = room.generate(model, prompt, 20, temperature=0.0)
    except ValueError as exc:
        pytest.fail(f"generate fed the model more than block_size tokens: {exc}. Feed only idx[:, -block_size:] each step.")
    assert out.shape == (1, 24)
    # Same thing by hand, with the crop, greedy: must agree token for token.
    idx = prompt.clone()
    with torch.no_grad():
        for _ in range(20):
            logits, _ = model(idx[:, -8:])
            idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, idx), (
        f"Greedy generation differs from the hand-rolled loop: {out[0].tolist()} vs {idx[0].tolist()}. "
        "Use the logits of the LAST position only, and crop to the last block_size tokens (not the first)."
    )


def test_generate_passes_the_sampling_options_through():
    model = _tiny_model(block_size=8, seed=1)
    prompt = torch.randint(0, 13, (2, 2))
    out = room.generate(model, prompt, 12, temperature=0.8, top_k=5, generator=torch.Generator().manual_seed(11))
    idx = prompt.clone()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _ in range(12):
            logits, _ = model(idx[:, -8:])
            nxt = room.sample_next(logits[:, -1, :], temperature=0.8, top_k=5, generator=g)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
    assert torch.equal(out, idx), (
        "generate(..., temperature=0.8, top_k=5, generator=g) should produce exactly what one sample_next call per step "
        "produces with the same options and generator. Pass **sampling straight through to sample_next."
    )


def test_the_chronicler_speaks_its_fixed_greedy_line():
    model, tok, _ = tiny_gpt.load_pretrained()
    prompt = torch.tensor([tok.encode("\n")], dtype=torch.long)
    n = 200
    expected = model.generate(prompt, n, temperature=0.0)
    got = room.generate(model, prompt, n, temperature=0.0)
    assert got.shape == expected.shape, f"Expected shape {tuple(expected.shape)}, got {tuple(got.shape)}."
    if not torch.equal(got, expected):
        first = int((got[0] != expected[0]).nonzero()[0])
        pytest.fail(
            f"Greedy decoding diverged from the Chronicler's own at token {first}.\n"
            f"  expected: {tok.decode(expected[0, :first + 20].tolist())!r}\n"
            f"  yours:    {tok.decode(got[0, :first + 20].tolist())!r}\n"
            "Greedy has no randomness: any difference is a bug in which logits you pick or how you crop."
        )
    text = tok.decode(got[0].tolist())
    assert math.isfinite(len(text)) and "chronicle" in text.lower() or "the" in text.lower(), "Dungeon self-check: the greedy line should be readable text."
