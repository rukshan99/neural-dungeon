"""BOSS FIGHT - THE LATENCY LEVIATHAN

Phase 1: serve eight petitions batched AND cached; token-exact with the naive
         baseline and several times faster on this machine.
Phase 2: an honest report from a real meter.
Phase 3: the prophecy, measured.
"""

import time

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.artifacts.tiny_gpt import load_pretrained, read_corpus  # noqa: E402
from dungeon.trials import load_room  # noqa: E402

boss = load_room(__file__, "boss_latency_leviathan")

pytestmark = pytest.mark.boss

torch.manual_seed(1212)
torch.set_num_threads(1)
MODEL, TOK, _ = load_pretrained()
CFG = MODEL.cfg
TEXT = read_corpus()
SPEEDUP = 3.0  # measured 7-10x naive/serve on a laptop core

PROMPTS = [
    "Ruel carried a mirror that showed the validation set. ",
    '"Count the axes," Okonkwo told Tadeo. ',
    "The Learning-Rate Lich",
    "Below them, the Scriptorium of Tokens. Above them, the Forge of Layers.",
    "the",
    "Zamira rested, then climbed, then measured, ",
    "Kasimir retreated. Kasimir waited. The dungeon did neither.",
    "Long past midnight, Yusuf found a ledger",
]
EIGHT = [TOK.encode(p) for p in PROMPTS]
N_EXACT = 32
N_SPEED = 24

_MEASURED: dict[str, float] = {}


def _reference(prompt_ids, n):
    with torch.no_grad():
        return MODEL.generate(torch.tensor([prompt_ids]), n, temperature=0)[0, len(prompt_ids):].tolist()


def _best_of(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _measure_naive_and_serve():
    if "no cache" not in _MEASURED:
        with torch.no_grad():
            boss.serve(MODEL, EIGHT, 3)  # warm-up
            boss.naive_serve(MODEL, EIGHT, 3)
            _MEASURED["no cache"] = _best_of(lambda: boss.naive_serve(MODEL, EIGHT, N_SPEED), 2)
            _MEASURED["cache+batch"] = _best_of(lambda: boss.serve(MODEL, EIGHT, N_SPEED), 2)
    return _MEASURED


def _measure_all_four():
    _measure_naive_and_serve()
    if "cache" not in _MEASURED:
        room_1 = load_room(__file__, "room_1_cache_of_keys")
        room_3 = load_room(__file__, "room_3_the_batcher")

        def cache_only():
            for p in EIGHT:
                room_1.generate_with_cache(MODEL, torch.tensor([p]), N_SPEED)

        with torch.no_grad():
            cache_only()
            _MEASURED["cache"] = _best_of(cache_only, 2)
            _MEASURED["batch"] = _best_of(lambda: room_3.batched_greedy_generate(MODEL, EIGHT, N_SPEED), 2)
    return _MEASURED


# --------------------------------------------------------------------- phase 1
def test_phase_1_the_naive_baseline_feeds_the_leviathan_honestly():
    got = boss.naive_serve(MODEL, EIGHT, 12)
    assert [list(g) for g in got] == [_reference(p, 12) for p in EIGHT], (
        "naive_serve must return exactly the reference greedy tokens (generated tokens only), one prompt at a time."
    )


def test_phase_1_serve_is_token_exact_for_eight_petitions_at_once():
    with torch.no_grad():
        got = boss.serve(MODEL, EIGHT, N_EXACT)
    assert isinstance(got, list) and len(got) == 8, "serve returns one token list per prompt."
    for b, p in enumerate(EIGHT):
        ref = _reference(p, N_EXACT)
        out = list(got[b])
        assert len(out) == N_EXACT, f"Row {b}: expected {N_EXACT} generated tokens, got {len(out)}."
        if out != ref:
            first = next(i for i in range(N_EXACT) if out[i] != ref[i])
            pytest.fail(
                f"Row {b} (prompt length {len(p)}) diverges at token {first}: reference {TOK.decode(ref)!r}, "
                f"served {TOK.decode(out)!r}. Check the cached mask: pads are keys in the cache too, and the "
                f"decode position must be the row's own last position + 1."
            )


def test_phase_1_the_cached_masked_attention_agrees_with_the_reference_block():
    block = MODEL.blocks[0]
    x = torch.randn(2, 8, CFG.n_embd)
    key_mask = torch.tensor([[True] * 8, [False, False, False] + [True] * 5])
    with torch.no_grad():
        y1, cache = boss.cached_masked_attention(block, x[:, :6], None, key_mask[:, :6])
        y2, cache = boss.cached_masked_attention(block, x[:, 6:], cache, key_mask)
        ref0 = block.attn(x[:1])
        ref1 = block.attn(x[1:, 3:])
    y = torch.cat([y1, y2], dim=1)
    assert torch.isfinite(y).all(), "Fully padded query rows produced NaN; nan_to_num after the softmax."
    assert torch.allclose(y[0], ref0[0], atol=1e-5), f"Unpadded row differs by {(y[0] - ref0[0]).abs().max():.2e}."
    assert torch.allclose(y[1, 3:], ref1[0], atol=1e-5), (
        f"Padded row's real positions differ by {(y[1, 3:] - ref1[0]).abs().max():.2e}: the pads must be masked in the cache too."
    )
    assert cache[0].shape == (2, CFG.n_head, 8, CFG.n_embd // CFG.n_head), f"cache k shape {tuple(cache[0].shape)}"


def test_phase_1_serve_outruns_the_naive_baseline():
    m = _measure_naive_and_serve()
    ratio = m["no cache"] / max(m["cache+batch"], 1e-9)
    assert ratio >= SPEEDUP, (
        f"8 prompts x {N_SPEED} tokens: naive {m['no cache'] * 1e3:.0f} ms, serve {m['cache+batch'] * 1e3:.0f} ms "
        f"({ratio:.1f}x). The Leviathan demands {SPEEDUP:.0f}x. One prefill, then (B, 1) decode steps."
    )


# --------------------------------------------------------------------- phase 2
def test_phase_2_the_report_has_every_gauge_and_they_agree_with_each_other():
    long_prompts = [TOK.encode(TEXT[o:o + 40 + 4 * i]) for i, o in enumerate(range(5000, 5800, 100))]
    with torch.no_grad():
        report = boss.leviathan_report(MODEL, long_prompts, 32)
    keys = {"requests", "prefill_ms", "decode_ms_per_token", "p50_ms", "p95_ms", "tokens", "tokens_per_second"}
    assert set(report) >= keys, f"report is missing {sorted(keys - set(report))}"
    assert report["requests"] == 8
    assert report["tokens"] == 8 * 32, f"8 requests x 32 tokens = 256 tokens; the report says {report['tokens']}."
    assert report["p50_ms"] <= report["p95_ms"], f"p50 {report['p50_ms']:.1f} > p95 {report['p95_ms']:.1f}: impossible."
    assert 0 < report["decode_ms_per_token"] < report["p50_ms"], "One decode step is a fraction of a whole request."
    assert 0 < report["prefill_ms"] < report["p50_ms"], "Prefill is one forward; the request is that plus 31 decode steps."
    assert report["prefill_ms"] > report["decode_ms_per_token"], (
        f"Prefilling 40-70 tokens ({report['prefill_ms']:.2f} ms) should cost more than one cached decode step "
        f"({report['decode_ms_per_token']:.2f} ms). Is the decode step feeding only one token?"
    )
    expected_tps = report["tokens"] / (report["p50_ms"] / 1000.0 * 8)
    assert 0.2 * expected_tps < report["tokens_per_second"] < 5 * expected_tps, (
        f"tokens_per_second {report['tokens_per_second']:.0f} is not in the same world as tokens / total time."
    )


def test_phase_2_the_report_uses_the_meter_you_were_given():
    room_4 = load_room(__file__, "room_4_the_meter")
    meter = room_4.LatencyMeter()
    with torch.no_grad():
        boss.leviathan_report(MODEL, EIGHT[:3], 8, meter=meter)
    assert len(meter.durations) == 3, f"Three requests should leave three lines in the meter; it holds {len(meter.durations)}."
    assert meter.tokens == [8, 8, 8], f"Eight tokens each: {meter.tokens}"
    assert len(meter.ttfts) == 3 and all(t < d for t, d in zip(meter.ttfts, meter.durations)), (
        "Call req.first_token() right after the prefill: TTFT must be recorded, before the request ends."
    )


# --------------------------------------------------------------------- phase 3
PROPHECY_KEYS = {"fastest_of_four", "cache_alone_vs_no_cache", "dominant_phase_60_60",
                 "forward_passes_with_cache", "token_rows_with_cache"}


def _prediction(key):
    value = boss.LEVIATHAN_PROPHECY.get(key)
    if value is None:
        raise NotImplementedError(f"LEVIATHAN_PROPHECY[{key!r}] is still None. Predict, then fight.")
    return value


def test_phase_3_the_prophecy_is_complete():
    assert set(boss.LEVIATHAN_PROPHECY) == PROPHECY_KEYS, "Do not add or remove prophecy lines; answer them."


def test_phase_3_the_fastest_of_four():
    m = _measure_all_four()
    fastest = min(m, key=m.get)
    table = ", ".join(f"{k}: {v * 1e3:.0f} ms" for k, v in sorted(m.items(), key=lambda kv: kv[1]))
    assert _prediction("fastest_of_four") == fastest, f"Measured ({table}); the fastest was {fastest!r}."


def test_phase_3_cache_alone_versus_no_cache():
    m = _measure_all_four()
    truth = "faster" if m["cache"] < m["no cache"] else "slower"
    assert _prediction("cache_alone_vs_no_cache") == truth, (
        f"Cache alone {m['cache'] * 1e3:.0f} ms vs no cache {m['no cache'] * 1e3:.0f} ms -> {truth!r}."
    )


def test_phase_3_which_phase_dominates_a_sixty_sixty_request():
    prompt = TOK.encode(TEXT[2000:2060])
    with torch.no_grad():
        boss.leviathan_report(MODEL, [prompt], 4)  # warm-up
        report = boss.leviathan_report(MODEL, [prompt], 60)
    decode_total = report["decode_ms_per_token"] * 59
    truth = "prefill" if report["prefill_ms"] > decode_total else "decode"
    assert _prediction("dominant_phase_60_60") == truth, (
        f"Prefill of 60 tokens: {report['prefill_ms']:.2f} ms. 59 decode steps: {decode_total:.2f} ms. -> {truth!r}."
    )


def test_phase_3_what_the_cache_changes():
    room_1 = load_room(__file__, "room_1_cache_of_keys")
    idx = torch.tensor([TOK.encode(TEXT[600:640])])
    N = 30

    def count(fn):
        calls, rows = [], []

        def hook(mod, inp, out):  # returns None so the embedding output is left alone
            calls.append(1)
            rows.append(inp[0].numel())

        handle = MODEL.wte.register_forward_hook(hook)
        try:
            with torch.no_grad():
                fn()
        finally:
            handle.remove()
        return len(calls), sum(rows)

    passes_plain, rows_plain = count(lambda: MODEL.generate(idx, N, temperature=0))
    passes_cache, rows_cache = count(lambda: room_1.generate_with_cache(MODEL, idx, N))
    assert passes_cache > 0, "The meter saw no lookups through model.wte in the cached path; call the module."
    passes_truth = "same" if abs(passes_cache - passes_plain) <= 1 else ("fewer" if passes_cache < passes_plain else "more")
    rows_truth = "fewer" if rows_cache < rows_plain else ("same" if rows_cache == rows_plain else "more")
    assert _prediction("forward_passes_with_cache") == passes_truth, (
        f"Forward passes for {N} tokens: {passes_plain} without a cache, {passes_cache} with -> {passes_truth!r}."
    )
    assert _prediction("token_rows_with_cache") == rows_truth, (
        f"Token rows through the embedding: {rows_plain} without a cache, {rows_cache} with -> {rows_truth!r}."
    )
