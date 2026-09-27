"""TRIAL 12.4 - THE METER

Known samples with known percentiles, a needle that must move before the
petition is finished, and a verdict that names what it failed.
"""

import time

import pytest

torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")

from dungeon.trials import load_room  # noqa: E402

room = load_room(__file__, "room_4_the_meter")


# ------------------------------------------------------------------ percentile
def test_nearest_rank_on_one_to_a_hundred():
    samples = [float(i) for i in range(100, 0, -1)]  # unsorted on purpose
    assert room.percentile(samples, 50) == 50.0, f"p50 of 1..100 is 50 by nearest rank; got {room.percentile(samples, 50)}"
    assert room.percentile(samples, 95) == 95.0, f"p95 of 1..100 is 95; got {room.percentile(samples, 95)}"
    assert room.percentile(samples, 99) == 99.0, f"p99 of 1..100 is 99; got {room.percentile(samples, 99)}"
    assert room.percentile(samples, 0) == 1.0 and room.percentile(samples, 100) == 100.0, "p0 is the min, p100 the max."


def test_nearest_rank_never_invents_a_latency():
    assert room.percentile([1.0, 100.0], 50) == 1.0, (
        "Nearest rank: rank = ceil(0.5 * 2) = 1 -> the first sorted element. 50.5 would be interpolation; no request took 50.5."
    )
    five = [30.0, 10.0, 50.0, 20.0, 40.0]
    assert room.percentile(five, 50) == 30.0, "ceil(2.5) = 3 -> third smallest = 30"
    assert room.percentile(five, 95) == 50.0, "ceil(4.75) = 5 -> the maximum"
    assert room.percentile(five, 20) == 10.0, "ceil(1.0) = 1 -> the minimum"
    assert isinstance(room.percentile(five, 50), float), "Return a Python float."


def test_the_meter_refuses_to_read_an_empty_gauge():
    with pytest.raises(ValueError):
        room.percentile([], 50)
    with pytest.raises(ValueError):
        room.percentile([1.0], 101)


def test_summary_keys_and_the_needles_never_cross():
    g = torch.Generator().manual_seed(4)
    samples = (torch.rand(500, generator=g) * 100).tolist() + [900.0, 1500.0]  # a long tail
    s = room.latency_summary(samples)
    assert set(s) == {"p50", "p95", "p99", "mean", "max"}, f"summary keys: {sorted(s)}"
    assert s["p50"] <= s["p95"] <= s["p99"] <= s["max"], f"Percentiles must be monotonic: {s}"
    assert s["max"] == 1500.0
    assert abs(s["mean"] - sum(samples) / len(samples)) < 1e-9
    assert s["p99"] < s["max"], "With 502 samples p99 is the 497th: the two outliers are above it."


def test_throughput_is_tokens_over_seconds():
    assert room.throughput(100, 2.0) == 50.0
    assert room.throughput(0, 1.0) == 0.0
    with pytest.raises(ValueError):
        room.throughput(10, 0.0)


# ------------------------------------------------------------------ LatencyMeter
def test_record_and_summary():
    meter = room.LatencyMeter()
    for d, n in [(0.010, 5), (0.020, 10), (0.030, 15), (0.040, 20)]:
        meter.record(d, n)
    s = meter.summary()
    assert s["requests"] == 4
    assert s["tokens"] == 50
    assert abs(s["tokens_per_second"] - 50 / 0.1) < 1e-6, f"50 tokens in 0.1 s is 500 tok/s; got {s['tokens_per_second']}"
    assert abs(s["latency_ms"]["p50"] - 20.0) < 1e-9, f"durations are recorded in seconds and summarised in ms; p50 {s['latency_ms']['p50']}"
    assert abs(s["latency_ms"]["max"] - 40.0) < 1e-9
    assert "ttft_ms" not in s, "No TTFT was recorded, so ttft_ms should be absent."
    meter.record(0.050, 1, ttft_s=0.005)
    assert abs(meter.summary()["ttft_ms"]["max"] - 5.0) < 1e-9


def test_the_context_manager_times_the_block_and_the_first_token_arrives_early():
    meter = room.LatencyMeter()
    with meter.request() as req:
        time.sleep(0.03)
        req.first_token()
        time.sleep(0.04)
        req.first_token()  # a second call must not move the needle
        req.tokens = 7
    assert len(meter.durations) == 1 and meter.tokens == [7], "One request, seven tokens, recorded on exit."
    duration, ttft = meter.durations[0], meter.ttfts[0]
    assert duration >= 0.07 * 0.8, f"The block slept ~70 ms; the meter read {duration * 1e3:.1f} ms."
    assert 0.03 * 0.8 <= ttft < duration, (
        f"TTFT {ttft * 1e3:.1f} ms should be ~30 ms and strictly less than the full duration {duration * 1e3:.1f} ms; "
        "the first call to first_token() wins."
    )


def test_run_load_drives_a_token_generator_and_meters_each_petition():
    meter = room.LatencyMeter()

    def slow_scribe(request):
        time.sleep(0.02)
        yield "a"
        for _ in range(request - 1):
            time.sleep(0.002)
            yield "b"

    outputs = room.run_load(slow_scribe, [3, 5, 1], meter)
    assert outputs == [["a", "b", "b"], ["a", "b", "b", "b", "b"], ["a"]], f"run_load should return the yielded tokens: {outputs}"
    assert meter.tokens == [3, 5, 1], f"token counts per request: {meter.tokens}"
    assert len(meter.durations) == 3 and len(meter.ttfts) == 3, "Every request has a duration and a TTFT."
    for d, t in zip(meter.durations, meter.ttfts):
        assert 0.02 * 0.8 <= t <= d, f"TTFT {t * 1e3:.1f} ms must fall before the request's end {d * 1e3:.1f} ms."


# ------------------------------------------------------------------ slo_report
def _meter_with(durations_ms, tokens_each):
    meter = room.LatencyMeter()
    for d in durations_ms:
        meter.record(d / 1000.0, tokens_each)
    return meter


def test_the_verdict_passes_when_both_targets_hold():
    meter = _meter_with([10.0] * 19 + [50.0], tokens_each=10)  # p95 = 10 ms (20th sample is the 50), 1000 tok/s
    report = room.slo_report(meter, p95_target_ms=20.0, tps_target=500.0)
    assert report["pass"] is True, f"p95 10 ms <= 20 and 1000 tok/s >= 500 should pass: {report}"
    assert report["violations"] == []
    assert set(report) >= {"pass", "p95_ms", "p95_target_ms", "tokens_per_second", "tps_target", "requests", "violations"}


def test_the_verdict_fails_on_the_tail_even_when_the_mean_looks_fine():
    meter = _meter_with([10.0] * 18 + [400.0, 400.0], tokens_each=10)  # mean 49 ms, p95 = 400 ms
    report = room.slo_report(meter, p95_target_ms=50.0, tps_target=1.0)
    assert report["pass"] is False, "Two slow requests in twenty put p95 at 400 ms. The mean does not save you."
    assert len(report["violations"]) == 1 and "p95" in report["violations"][0], f"Name the broken target: {report['violations']}"


def test_the_verdict_fails_on_throughput_alone():
    meter = _meter_with([10.0] * 20, tokens_each=1)  # 100 tok/s
    report = room.slo_report(meter, p95_target_ms=50.0, tps_target=500.0)
    assert report["pass"] is False
    assert len(report["violations"]) == 1 and "throughput" in report["violations"][0].lower(), report["violations"]
    both = room.slo_report(meter, p95_target_ms=5.0, tps_target=500.0)
    assert len(both["violations"]) == 2, "Both targets broken: two violations."
