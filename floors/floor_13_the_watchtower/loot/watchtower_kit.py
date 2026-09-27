"""Watchtower Kit - loot from Floor 13, The Watchtower.

A standalone copy of the tower's tools: numpy and the standard library only,
no dungeon imports. Copy this file into a project and use it as is.

    from watchtower_kit import Tracer, DriftMonitor, assign_arm, compare_arms, release_decision, AlertManager

    tracer = Tracer()                                     # time.perf_counter by default
    with tracer.span("agent.run", "chain", request_id=rid):
        ...

    monitor = DriftMonitor(reference_features, thresholds={"psi": 0.1, "ks": 0.15})
    report = monitor.observe(todays_features)             # report["drifted_features"]

    arm = assign_arm(request_id, canary_fraction=0.05, salt="release-42")
    decision = release_decision(compare_arms(baseline_scores, canary_scores), guardrail_check(metrics, slo))

    alerts = AlertManager(clock=time.time, cooldown=600)
    alerts.add_rule("p99_latency_s", Threshold(fire_above=2.0, clear_below=1.5))
    for event in alerts.observe("p99_latency_s", window.percentile(99)):
        page(event)

Run this file directly for a small demonstration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

# ==================================================================== tracing

REDACTED = "[REDACTED]"
FIELD_ORDER = ("name", "kind", "start", "end", "status", "attributes", "children")


@dataclass
class Span:
    name: str
    kind: str
    start: float
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            raise ValueError(f"span {self.name!r} is still open")
        return (self.end - self.start) * 1000.0


def walk(span: Span, depth: int = 1) -> Iterator[tuple[Span, int]]:
    yield span, depth
    for child in span.children:
        yield from walk(child, depth + 1)


class Tracer:
    """Spans nest by a stack; the clock is injectable so tests can be exact."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.traces: list[Span] = []
        self._stack: list[Span] = []

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attributes):
        span = Span(name, kind, float(self.clock()), attributes=dict(attributes))
        (self._stack[-1].children if self._stack else self.traces).append(span)
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.attributes["error.type"] = type(exc).__name__
            span.attributes["error.message"] = str(exc)
            raise
        finally:
            span.end = float(self.clock())
            self._stack.pop()


def record_call(tracer: Tracer, name: str, kind: str, fn: Callable, **arguments):
    """Run ``fn(**arguments)`` inside a span of ``kind``; the result is returned, errors propagate."""
    with tracer.span(name, kind, arguments=dict(arguments)) as span:
        result = fn(**arguments)
        span.attributes["result_chars"] = len(str(result))
    return result


def summarize(trace: Span) -> dict:
    out = {"total_ms": trace.duration_ms, "llm_ms": 0.0, "tool_ms": 0.0, "tokens": 0, "cost": 0.0,
           "n_llm_calls": 0, "n_tool_calls": 0, "n_errors": 0, "depth": 0}
    for span, depth in walk(trace):
        out["depth"] = max(out["depth"], depth)
        out["n_errors"] += span.status == "error"
        if span.kind == "llm":
            out["n_llm_calls"] += 1
            out["llm_ms"] += span.duration_ms
            out["tokens"] += int(span.attributes.get("input_tokens", 0)) + int(span.attributes.get("output_tokens", 0))
            out["cost"] += float(span.attributes.get("cost_usd", 0.0))
        elif span.kind == "tool":
            out["n_tool_calls"] += 1
            out["tool_ms"] += span.duration_ms
    return out


def span_to_dict(span: Span) -> dict:
    return {"name": span.name, "kind": span.kind, "start": span.start, "end": span.end, "status": span.status,
            "attributes": dict(span.attributes), "children": [span_to_dict(c) for c in span.children]}


def to_json(trace: Span, indent: int | None = None) -> str:
    return json.dumps(span_to_dict(trace), indent=indent, default=str)


def redact_attributes(trace: Span, keys: Iterable[str], placeholder: str = REDACTED) -> Span:
    wanted = {k.lower() for k in keys}
    clone = copy.deepcopy(trace)
    for span, _ in walk(clone):
        for key in list(span.attributes):
            if key.lower() in wanted:
                span.attributes[key] = placeholder
    return clone


# ====================================================================== drift


def histogram_bins(reference, n_bins: int = 10) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    inner = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_fractions(values, edges) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).ravel()
    idx = np.searchsorted(np.asarray(edges)[1:-1], x, side="right")
    return np.bincount(idx, minlength=len(edges) - 1) / x.size


def psi(reference, current, edges=None, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. < 0.1 stable, 0.1-0.25 moderate, >= 0.25 major."""
    edges = histogram_bins(reference, n_bins) if edges is None else edges
    r = np.clip(bin_fractions(reference, edges), eps, None)
    c = np.clip(bin_fractions(current, edges), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_statistic(a, b) -> float:
    a, b = np.sort(np.asarray(a, dtype=np.float64).ravel()), np.sort(np.asarray(b, dtype=np.float64).ravel())
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size - np.searchsorted(b, grid, side="right") / b.size)))


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence in nats of two categorical distributions (probabilities or counts)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def embedding_drift(ref_vectors, cur_vectors) -> float:
    """(1 - cosine of centroids) + relative change in mean norm."""
    ref, cur = np.asarray(ref_vectors, dtype=np.float64), np.asarray(cur_vectors, dtype=np.float64)
    cr, cc = ref.mean(axis=0), cur.mean(axis=0)
    denom = np.linalg.norm(cr) * np.linalg.norm(cc)
    cosine_distance = max(0.0, 1.0 - float(cr @ cc) / denom) if denom > 0 else 0.0
    nr, nc = np.linalg.norm(ref, axis=1).mean(), np.linalg.norm(cur, axis=1).mean()
    return float(cosine_distance + (abs(nc - nr) / nr if nr > 0 else 0.0))


class DriftMonitor:
    """Per-feature PSI and KS of the last ``window`` batches against a fixed reference."""

    def __init__(self, reference, window: int = 1, n_bins: int = 10, thresholds: dict | None = None) -> None:
        ref = np.asarray(reference, dtype=np.float64)
        self.reference = ref[:, None] if ref.ndim == 1 else ref
        self.thresholds = {"psi": 0.25, "ks": 0.15, **(thresholds or {})}
        self.edges = [histogram_bins(self.reference[:, j], n_bins) for j in range(self.reference.shape[1])]
        self._recent: deque[np.ndarray] = deque(maxlen=window)
        self.history: list[dict] = []

    def observe(self, batch) -> dict:
        x = np.asarray(batch, dtype=np.float64)
        x = x[:, None] if x.ndim == 1 else x
        self._recent.append(x)
        current = np.concatenate(list(self._recent))
        d = self.reference.shape[1]
        psi_values = [psi(self.reference[:, j], current[:, j], self.edges[j]) for j in range(d)]
        ks_values = [ks_statistic(self.reference[:, j], current[:, j]) for j in range(d)]
        drifted = [j for j in range(d) if psi_values[j] > self.thresholds["psi"] or ks_values[j] > self.thresholds["ks"]]
        report = {"n_current": int(current.shape[0]), "psi": psi_values, "ks": ks_values, "max_psi": max(psi_values),
                  "max_ks": max(ks_values), "drifted_features": drifted, "drifted": bool(drifted)}
        self.history.append(report)
        return report


# ==================================================================== canary

DEFAULT_RAMP = (0.01, 0.05, 0.25, 0.5, 1.0)


def assign_arm(request_id: str, canary_fraction: float, salt: str = "") -> str:
    """Deterministic, stateless, monotone under ramps: sha256(salt:id) -> uniform -> compare."""
    if not 0.0 <= canary_fraction <= 1.0:
        raise ValueError("canary_fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{salt}:{request_id}".encode("utf-8")).digest()
    return "canary" if int.from_bytes(digest[:8], "big") / 2**64 < canary_fraction else "baseline"


def compare_arms(baseline_scores, canary_scores, n_boot: int = 2000, alpha: float = 0.05, rng=None) -> dict:
    """Unpaired bootstrap of mean(canary) - mean(baseline): the arms saw different requests."""
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(baseline_scores, dtype=np.float64).ravel()
    b = np.asarray(canary_scores, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("both arms need at least one score")
    deltas = np.empty(n_boot)
    rows = max(1, int(2_000_000 // max(a.size, b.size)))
    for start in range(0, n_boot, rows):
        k = min(rows, n_boot - start)
        deltas[start:start + k] = (b[rng.integers(0, b.size, size=(k, b.size))].mean(axis=1)
                                   - a[rng.integers(0, a.size, size=(k, a.size))].mean(axis=1))
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return {"delta": float(b.mean() - a.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": p,
            "n_baseline": int(a.size), "n_canary": int(b.size)}


def guardrail_check(canary_metrics: dict, slo: dict) -> dict:
    """Every SLO key is a maximum; an unmeasured metric is a violation."""
    violations = [name for name, limit in slo.items()
                  if canary_metrics.get(name) is None or float(canary_metrics[name]) > float(limit)]
    return {"passed": not violations, "violations": violations}


def release_decision(comparison: dict, guardrails: dict, min_effect: float = 0.0) -> str:
    if not guardrails["passed"] or comparison["ci_high"] < 0.0:
        return "rollback"
    if comparison["ci_low"] > 0.0 and comparison["delta"] >= min_effect:
        return "promote"
    return "extend"


def advance_ramp(fraction: float, decision: str, steps: Sequence[float] = DEFAULT_RAMP) -> float:
    if decision == "rollback":
        return 0.0
    if decision == "extend":
        return float(fraction)
    return next((float(s) for s in steps if s > fraction), 1.0)


# ==================================================================== alerts


class RollingWindow:
    def __init__(self, size: int) -> None:
        self._values: deque[float] = deque(maxlen=size)

    def push(self, value: float) -> None:
        self._values.append(float(value))

    def __len__(self) -> int:
        return len(self._values)

    def percentile(self, p: float) -> float:
        """Nearest rank: sorted[ceil(p/100 * n) - 1]. Always a value that actually happened."""
        if not self._values:
            raise ValueError("the window is empty")
        ordered = sorted(self._values)
        return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]

    def mean(self) -> float:
        return sum(self._values) / len(self._values)

    def rate(self, predicate: Callable[[float], bool]) -> float:
        return sum(1 for v in self._values if predicate(v)) / len(self._values)


class Threshold:
    """Fires above one line, clears below a lower one: a hovering metric pages once."""

    def __init__(self, fire_above: float, clear_below: float) -> None:
        if clear_below > fire_above:
            raise ValueError("clear_below must not exceed fire_above")
        self.fire_above, self.clear_below, self.active = float(fire_above), float(clear_below), False

    def update(self, value: float) -> str | None:
        if not self.active and value > self.fire_above:
            self.active = True
            return "fired"
        if self.active and value < self.clear_below:
            self.active = False
            return "resolved"
        return None


@dataclass(frozen=True)
class Alert:
    name: str
    severity: str
    value: float
    fired_at: float
    kind: str = "fire"  # "fire" | "resolved"


class AlertManager:
    def __init__(self, clock: Callable[[], float] = time.time, cooldown: float = 0.0) -> None:
        self.clock, self.cooldown = clock, float(cooldown)
        self.rules: dict[str, tuple[Threshold, str]] = {}
        self.events: list[Alert] = []
        self.last_fired: dict[str, float] = {}
        self.suppressed = 0
        self._announced: set[str] = set()

    def add_rule(self, name: str, threshold: Threshold, severity: str = "page") -> None:
        self.rules[name] = (threshold, severity)

    def observe(self, name: str, value: float) -> list[Alert]:
        threshold, severity = self.rules[name]
        now, edge, emitted = float(self.clock()), None, []
        edge = threshold.update(value)
        if edge == "fired":
            if name in self.last_fired and now - self.last_fired[name] < self.cooldown:
                self.suppressed += 1
            else:
                emitted.append(Alert(name, severity, float(value), now, "fire"))
                self.last_fired[name] = now
                self._announced.add(name)
        elif edge == "resolved" and name in self._announced:
            emitted.append(Alert(name, severity, float(value), now, "resolved"))
            self._announced.discard(name)
        self.events.extend(emitted)
        return emitted


def error_budget(slo_target: float, window_requests: int, failures: int) -> dict:
    allowed = (1.0 - slo_target) * window_requests
    return {"allowed_failures": allowed, "remaining_fraction": (allowed - failures) / allowed,
            "burn_rate": (failures / window_requests) / (1.0 - slo_target)}


# ================================================================ sequential


class SequentialSentinel:
    """Wald's SPRT on discordant pairs: peek every day and keep alpha. See Floor 13's secret room."""

    def __init__(self, p_a: float, min_effect: float, alpha: float = 0.05, beta: float = 0.2) -> None:
        p_b = p_a + min_effect
        b_only, a_only = p_b * (1 - p_a), p_a * (1 - p_b)
        self.q1 = b_only / (b_only + a_only)
        self.lower, self.upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
        self.llr, self.n_pairs, self.decision = 0.0, 0, None

    def update(self, a_success, b_success) -> str | None:
        if self.decision is not None:
            return self.decision
        self.n_pairs += 1
        if bool(a_success) != bool(b_success):
            self.llr += math.log(self.q1 / 0.5) if b_success else math.log((1 - self.q1) / 0.5)
            if self.llr >= self.upper:
                self.decision = "b_better"
            elif self.llr <= self.lower:
                self.decision = "no_difference"
        return self.decision


# ====================================================================== demo

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(5000, 3))
    monitor = DriftMonitor(reference, thresholds={"psi": 0.1, "ks": 0.15})
    for day in range(6):
        batch = rng.normal(size=(500, 3))
        batch[:, 2] += 0.15 * day
        report = monitor.observe(batch)
        print(f"day {day}: max PSI {report['max_psi']:.3f}  max KS {report['max_ks']:.3f}  drifted {report['drifted_features']}")

    baseline = (rng.random(4500) < 0.88).astype(float)
    canary = (rng.random(500) < 0.91).astype(float)
    comparison = compare_arms(baseline, canary, rng=rng)
    guardrails = guardrail_check({"p95_latency_ms": 640.0, "error_rate": 0.004}, {"p95_latency_ms": 1000.0, "error_rate": 0.02})
    print("canary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in comparison.items()}, "->", release_decision(comparison, guardrails))

    alerts = AlertManager(clock=lambda: 0.0, cooldown=0.0)
    alerts.add_rule("error_rate", Threshold(0.20, 0.15))
    kinds = [e.kind for v in (0.10, 0.21, 0.19, 0.22, 0.18, 0.10) for e in alerts.observe("error_rate", v)]
    print("hovering metric ->", kinds)
    print("error budget:", error_budget(0.999, 100_000, 30))
