"""ROOM 12.4 - THE METER

    A wall of brass gauges, one per pipe. The needle you look at first is the
    average. The needle that matters is the one on the far right, the one that
    twitches once in a hundred petitions and sends someone home in tears.

A mean hides the tail. If 99 requests take 20 ms and one takes 4 seconds, the
mean says 60 ms and the one user who waited 4 seconds does not care. Serving
systems are judged by percentiles:

    p50  the median; what a typical request sees
    p95  one in twenty is slower than this
    p99  one in a hundred; the number in the contract (the SLO)

Why the tail matters more than it looks: a page that makes 10 backend calls
hits at least one p99-slow call on 1 - 0.99^10 ~ 10% of loads. Tails add up.

Nearest-rank percentile (the one this room uses): sort the samples, take the
element at 1-based rank ceil(p/100 * n). It never interpolates, so the answer
is always a latency that actually happened. p0 is the minimum, p100 the maximum.

Two other gauges: throughput (tokens per second, the number the finance
department reads) and time-to-first-token (TTFT, the number a human waiting on
a stream reads). Latency and throughput pull against each other - batching
raises throughput and every request's latency at once - which is why an SLO
names both, and why you measure per request, after a warm-up, and report the
percentiles rather than the mean.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile of ``samples`` (any unit) for ``p`` in [0, 100].

    Sort; take the element at 1-based rank ``ceil(p / 100 * n)`` (rank 0 becomes 1).
    Return a Python float. Raise ValueError for an empty list or p outside [0, 100].
    """
    raise NotImplementedError("percentile() is unwritten")


def latency_summary(samples: list[float]) -> dict[str, float]:
    """``{"p50", "p95", "p99", "mean", "max"}`` of ``samples`` in whatever unit they came in."""
    raise NotImplementedError("latency_summary() is unwritten")


def throughput(tokens: int, seconds: float) -> float:
    """Tokens per second. Raise ValueError if ``seconds`` is not positive."""
    raise NotImplementedError("throughput() is unwritten")


class RequestTimer:
    """Times one request. Use as ``with meter.request() as req:``.

    ``__enter__`` starts the clock (``time.perf_counter``) and returns self.
    ``first_token()`` records time-to-first-token; only the first call counts.
    Set ``req.tokens`` to the number of tokens produced before the block ends.
    ``__exit__`` computes the duration and calls ``meter.record(duration, tokens, ttft)``.
    """

    def __init__(self, meter: LatencyMeter):
        raise NotImplementedError("RequestTimer.__init__() is unwritten")

    def __enter__(self) -> RequestTimer:
        raise NotImplementedError("RequestTimer.__enter__() is unwritten")

    def first_token(self) -> None:
        raise NotImplementedError("RequestTimer.first_token() is unwritten")

    def __exit__(self, exc_type, exc, tb) -> None:
        raise NotImplementedError("RequestTimer.__exit__() is unwritten")


class LatencyMeter:
    """Collects one line per request: duration in seconds, tokens produced, TTFT in seconds.

    Attributes: ``durations: list[float]``, ``tokens: list[int]``, ``ttfts: list[float]``
    (ttfts only holds requests that reported one).
    """

    def __init__(self) -> None:
        raise NotImplementedError("LatencyMeter.__init__() is unwritten")

    def record(self, duration_s: float, tokens: int, ttft_s: float | None = None) -> None:
        """Append one request."""
        raise NotImplementedError("LatencyMeter.record() is unwritten")

    def request(self) -> RequestTimer:
        """A context manager that records the block it wraps."""
        raise NotImplementedError("LatencyMeter.request() is unwritten")

    def summary(self) -> dict:
        """``{"requests": n, "latency_ms": latency_summary(durations in ms), "tokens": total,
        "tokens_per_second": total tokens / total seconds, "ttft_ms": latency_summary(...)}``.
        Include ``ttft_ms`` only when at least one TTFT was recorded.
        """
        raise NotImplementedError("LatencyMeter.summary() is unwritten")


def run_load(fn: Callable[[object], Iterable], requests: list, meter: LatencyMeter) -> list[list]:
    """Sequential load harness.

    For each request: open ``meter.request()``, iterate the token generator ``fn(request)``
    to exhaustion, call ``req.first_token()`` when the first token arrives, set
    ``req.tokens`` to the count. Return the list of token lists (one per request).
    """
    raise NotImplementedError("run_load() is unwritten")


def slo_report(meter: LatencyMeter, p95_target_ms: float, tps_target: float) -> dict:
    """The verdict. ``pass`` is True iff p95 latency <= target AND tokens/s >= target.

    Return ``{"pass": bool, "p95_ms", "p95_target_ms", "tokens_per_second", "tps_target",
    "requests", "violations": [human-readable strings, one per broken target]}``.
    """
    raise NotImplementedError("slo_report() is unwritten")
