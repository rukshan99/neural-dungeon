"""TRIAL 12.1 - THE CACHE OF KEYS

The pipes are connected. Every key computed once, every token exact, and the
recomputing reference left behind on the same machine.
"""

import time

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_1_cache_of_keys")

torch.manual_seed(12)
torch.set_num_threads(1)  # a tiny model on one core: the fairest clock for a ratio
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
TEXT = read_corpus()
SPEEDUP = 1.5  # measured 3-10x on a laptop core; the bar is deliberately low

PROMPTS = [
    "Ruel carried a mirror that showed the validation set. ",
    '"Count the axes," Okonkwo told Tadeo. ',
    "The Learning-Rate Lich",
]


def _enc(text: str) -> torch.Tensor:
    return torch.tensor([TOK.encode(text)], dtype=torch.long)


def _best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ------------------------------------------------------------------ attention_with_cache
def test_the_cache_holds_keys_and_values_per_head_and_grows_by_the_new_tokens():
    block = MODEL.blocks[0]
    H, hd = CFG.n_head, CFG.n_embd // CFG.n_head
    x = torch.randn(2, 5, CFG.n_embd)
    with torch.no_grad():
        y, cache = room.attention_with_cache(block, x, None)
    assert y.shape == (2, 5, CFG.n_embd), f"y should be (B, T_new, C) = (2, 5, {CFG.n_embd}), got {tuple(y.shape)}"
    assert isinstance(cache, (tuple, list)) and len(cache) == 2, "The cache is a pair (k, v)."
    k, v = cache
    assert k.shape == (2, H, 5, hd) and v.shape == (2, H, 5, hd), (
        f"Cached k and v should be (B, H, T, hd) = (2, {H}, 5, {hd}); got {tuple(k.shape)} and {tuple(v.shape)}."
    )
    with torch.no_grad():
        y2, (k2, v2) = room.attention_with_cache(block, torch.randn(2, 3, CFG.n_embd), cache)
    assert y2.shape == (2, 3, CFG.n_embd), f"Three new tokens in, three out: got {tuple(y2.shape)}"
    assert k2.shape == (2, H, 8, hd), f"5 cached + 3 new keys = 8 along the time axis; got {tuple(k2.shape)}"
    assert torch.equal(k2[:, :, :5], k), "The first five cached keys must be carried over unchanged."


def test_cached_attention_matches_the_reference_attention_when_the_prefix_is_split():
    block = MODEL.blocks[1]
    x = torch.randn(2, 9, CFG.n_embd)
    with torch.no_grad():
        ref = block.attn(x)
        y_all, _ = room.attention_with_cache(block, x, None)
        y1, c1 = room.attention_with_cache(block, x[:, :6], None)
        y2, c2 = room.attention_with_cache(block, x[:, 6:], c1)
    assert torch.allclose(y_all, ref, atol=1e-5), (
        f"With no cache your attention should equal block.attn(x); max diff {(y_all - ref).abs().max():.2e}. "
        "Check the 1/sqrt(hd) scaling and the causal mask."
    )
    assert torch.allclose(y1, ref[:, :6], atol=1e-5), "The first 6 positions do not match: prefill is wrong."
    assert torch.allclose(y2, ref[:, 6:], atol=1e-5), (
        f"Positions 6..8 attending over 6 cached + 3 new keys differ from the reference by "
        f"{(y2 - ref[:, 6:]).abs().max():.2e}. New query i sits at position T_past + i and may see keys 0..T_past + i."
    )
    assert c2[0].shape[2] == 9, "After the second call the cache should cover all 9 positions."


def test_a_new_token_may_not_peek_at_a_newer_one():
    """Two new tokens at once: the first must be blind to the second."""
    block = MODEL.blocks[2]
    x = torch.randn(1, 7, CFG.n_embd)
    with torch.no_grad():
        _, cache = room.attention_with_cache(block, x[:, :5], None)
        y_a, _ = room.attention_with_cache(block, x[:, 5:], cache)
        x_alt = x.clone()
        x_alt[:, 6] += 3.0  # change only the LAST new token
        y_b, _ = room.attention_with_cache(block, x_alt[:, 5:], cache)
    assert torch.allclose(y_a[:, 0], y_b[:, 0], atol=1e-5), (
        "Changing new token 1 changed the output at new token 0. The causal mask for the new positions is missing."
    )
    assert not torch.allclose(y_a[:, 1], y_b[:, 1], atol=1e-5), "Changing a token should change its own output."


# ------------------------------------------------------------------ forward_with_cache
def test_prefill_logits_equal_the_plain_forward_and_fill_every_layer():
    idx = _enc(PROMPTS[0])
    with torch.no_grad():
        ref, _ = MODEL(idx)
        logits, caches = room.forward_with_cache(MODEL, idx, 0, None)
    assert logits.shape == ref.shape, f"logits should be {tuple(ref.shape)}, got {tuple(logits.shape)}"
    assert torch.allclose(logits, ref, atol=1e-4), (
        f"Prefill logits differ from model(idx) by up to {(logits - ref).abs().max():.2e}. "
        "Positions 0..T-1, ln_1 before attention, ln_2 before the MLP, ln_f before lm_head."
    )
    assert len(caches) == CFG.n_layer, f"One cache per layer: expected {CFG.n_layer}, got {len(caches)}"
    T = idx.shape[1]
    for i, (k, v) in enumerate(caches):
        assert k.shape == (1, CFG.n_head, T, CFG.n_embd // CFG.n_head), f"layer {i} k cache is {tuple(k.shape)}"


def test_one_decode_step_equals_recomputing_the_whole_prefix():
    idx = _enc(PROMPTS[1])
    T = idx.shape[1]
    with torch.no_grad():
        ref, _ = MODEL(idx)
        _, caches = room.forward_with_cache(MODEL, idx[:, :-1], 0, None)
        logits, caches = room.forward_with_cache(MODEL, idx[:, -1:], T - 1, caches)
    assert logits.shape == (1, 1, CFG.vocab_size), f"One token in, one row of logits out; got {tuple(logits.shape)}"
    assert torch.allclose(logits[:, -1], ref[:, -1], atol=1e-4), (
        f"Decoding the last token with a cache differs from the full forward by {(logits[:, -1] - ref[:, -1]).abs().max():.2e}. "
        "Is the position id past_len (not 0)? Are k and v concatenated behind the cache?"
    )
    assert caches[0][0].shape[2] == T, "The cache should now cover the whole sequence."


def test_the_position_table_has_an_end():
    idx = _enc(TEXT[:10])
    with pytest.raises(ValueError):
        room.forward_with_cache(MODEL, idx, CFG.block_size - 5, None)


# ------------------------------------------------------------------ generate_with_cache
@pytest.mark.parametrize("prompt", PROMPTS, ids=["ruel", "okonkwo", "lich"])
def test_generation_is_token_exact_with_the_recomputing_reference(prompt):
    idx = _enc(prompt)
    with torch.no_grad():
        ref = MODEL.generate(idx, 60, temperature=0)
        got = room.generate_with_cache(MODEL, idx, 60)
    assert got.shape == ref.shape, f"Expected {tuple(ref.shape)} (prompt + 60 tokens), got {tuple(got.shape)}"
    if not torch.equal(got, ref):
        first = int((got[0] != ref[0]).nonzero()[0])
        pytest.fail(
            f"Token {first - idx.shape[1]} of the generation differs: reference {TOK.decode(ref[0, idx.shape[1]:])!r} "
            f"vs yours {TOK.decode(got[0, idx.shape[1]:])!r}. Greedy decoding with a correct cache is exact."
        )


def test_every_token_is_embedded_exactly_once():
    idx = _enc(TEXT[500:540])
    P, N = idx.shape[1], 30
    rows = []
    handle = MODEL.wte.register_forward_hook(lambda mod, inp, out: rows.append(inp[0].numel()))
    try:
        with torch.no_grad():
            room.generate_with_cache(MODEL, idx, N)
    finally:
        handle.remove()
    total = sum(rows)
    assert total > 0, "The meter saw no lookups through model.wte. Call the module (model.wte(idx)) so the trial can count."
    assert P + N - 1 <= total <= P + N, (
        f"{total} tokens went through the embedding for a {P}-token prompt and {N} new tokens. "
        f"A cache embeds each token once: {P} at prefill plus one per decode step ({P + N - 1}). "
        f"Without a cache it would be about {sum(range(P, P + N))}."
    )


def test_the_cache_knows_where_the_corridor_ends():
    idx = _enc(TEXT[3000:3000 + CFG.block_size - 8])
    with torch.no_grad():
        got = room.generate_with_cache(MODEL, idx, 20)
        ref = MODEL.generate(idx, 8, temperature=0)
    assert got.shape[1] <= CFG.block_size, (
        f"The output is {got.shape[1]} tokens long but the position table has {CFG.block_size} rows."
    )
    assert torch.equal(got[:, :CFG.block_size], ref[:, :CFG.block_size]), (
        "The 8 tokens that fit before block_size should equal the reference's first 8."
    )
    full = _enc(TEXT[4000:4000 + CFG.block_size])
    with torch.no_grad():
        out = room.generate_with_cache(MODEL, full, 3)
    assert out.shape[1] <= CFG.block_size, "A prompt that already fills block_size must not overflow it."


def test_the_cache_outruns_the_recomputing_reference():
    idx = _enc(TEXT[1000:1060])
    N = 60
    with torch.no_grad():
        MODEL.generate(idx, 4, temperature=0)  # warm-up
        room.generate_with_cache(MODEL, idx, 4)
        t_ref = _best_of(lambda: MODEL.generate(idx, N, temperature=0), 3)
        t_cache = _best_of(lambda: room.generate_with_cache(MODEL, idx, N), 3)
    ratio = t_ref / max(t_cache, 1e-9)
    assert ratio >= SPEEDUP, (
        f"60 prompt tokens + 60 new: the reference took {t_ref * 1e3:.0f} ms, your cache {t_cache * 1e3:.0f} ms "
        f"({ratio:.2f}x). The Engine Room demands {SPEEDUP}x. Are you feeding only the newest token at each decode step?"
    )
