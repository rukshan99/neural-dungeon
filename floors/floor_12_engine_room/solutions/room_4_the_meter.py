"""ROOM 12.4 - THE METER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Percentiles by nearest rank, a meter that records one line per request, a
sequential load harness, and an SLO verdict that says pass or fail and why.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile: sort, take element number ceil(p/100 * n) (1-based).

    p=0 gives the minimum, p=100 the maximum. Never interpolates: the answer is
    always a value that actually happened. Raises ValueError on an empty list.
    """
    if not samples:
        raise ValueError("percentile of nothing: record some requests first")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(samples)
    rank = math.ceil(p / 100 * len(ordered))
    return float(ordered[max(rank, 1) - 1])


def latency_summary(samples: list[float]) -> dict[str, float]:
    """{p50, p95, p99, mean, max} of ``samples`` (whatever unit they came in)."""
    return {
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "mean": float(sum(samples) / len(samples)),
        "max": float(max(samples)),
    }


def throughput(tokens: int, seconds: float) -> float:
    """Tokens per second. Raises ValueError if seconds is not positive."""
    if seconds <= 0:
        raise ValueError(f"throughput needs a positive duration, got {seconds}")
    return tokens / seconds


class RequestTimer:
    """Times one request. Call ``first_token()`` when the first token appears; set ``tokens``."""

    def __init__(self, meter: LatencyMeter):
        self.meter = meter
        self.tokens = 0
        self.t_start = 0.0
        self.ttft: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> RequestTimer:
        self.t_start = time.perf_counter()
        return self

    def first_token(self) -> None:
        """Record time-to-first-token. Idempotent: only the first call counts."""
        if self.ttft is None:
            self.ttft = time.perf_counter() - self.t_start

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration = time.perf_counter() - self.t_start
        self.meter.record(self.duration, self.tokens, self.ttft)


class LatencyMeter:
    """Collects one (duration_s, tokens, ttft_s) line per request."""

    def __init__(self) -> None:
        self.durations: list[float] = []
        self.tokens: list[int] = []
        self.ttfts: list[float] = []

    def record(self, duration_s: float, tokens: int, ttft_s: float | None = None) -> None:
        self.durations.append(float(duration_s))
        self.tokens.append(int(tokens))
        if ttft_s is not None:
            self.ttfts.append(float(ttft_s))

    def request(self) -> RequestTimer:
        """``with meter.request() as req:`` times the block and records it on exit."""
        return RequestTimer(self)

    @property
    def count(self) -> int:
        return len(self.durations)

    def total_tokens(self) -> int:
        return sum(self.tokens)

    def total_seconds(self) -> float:
        return sum(self.durations)

    def summary(self) -> dict:
        """Latency percentiles in ms, TTFT percentiles in ms (if any), totals and tokens/s."""
        ms = [d * 1000.0 for d in self.durations]
        out = {
            "requests": self.count,
            "latency_ms": latency_summary(ms),
            "tokens": self.total_tokens(),
            "tokens_per_second": throughput(self.total_tokens(), self.total_seconds()),
        }
        if self.ttfts:
            out["ttft_ms"] = latency_summary([t * 1000.0 for t in self.ttfts])
        return out


def run_load(fn: Callable[[object], Iterable], requests: list, meter: LatencyMeter) -> list[list]:
    """Sequential harness: for each request, drive the token generator ``fn(request)`` to
    exhaustion, recording TTFT on the first token and the token count at the end.

    Returns the list of token lists, one per request.
    """
    outputs: list[list] = []
    for request in requests:
        tokens: list = []
        with meter.request() as req:
            for tok in fn(request):
                req.first_token()
                tokens.append(tok)
            req.tokens = len(tokens)
        outputs.append(tokens)
    return outputs


def slo_report(meter: LatencyMeter, p95_target_ms: float, tps_target: float) -> dict:
    """Pass iff p95 latency <= target AND throughput >= target. Lists every violation in words."""
    summary = meter.summary()
    p95_ms = summary["latency_ms"]["p95"]
    tps = summary["tokens_per_second"]
    violations = []
    if p95_ms > p95_target_ms:
        violations.append(f"p95 latency {p95_ms:.1f} ms exceeds target {p95_target_ms:.1f} ms")
    if tps < tps_target:
        violations.append(f"throughput {tps:.1f} tok/s is below target {tps_target:.1f} tok/s")
    return {
        "pass": not violations,
        "p95_ms": p95_ms,
        "p95_target_ms": float(p95_target_ms),
        "tokens_per_second": tps,
        "tps_target": float(tps_target),
        "requests": summary["requests"],
        "violations": violations,
    }
