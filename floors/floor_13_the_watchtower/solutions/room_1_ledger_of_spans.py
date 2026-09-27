"""ROOM 13.1 - THE LEDGER OF SPANS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A trace is a tree of spans. The tracer keeps a stack: the span on top is the
parent of whatever opens next, so nesting falls out of a list. Exceptions mark
the span as an error, are re-raised, and still close the span (``finally``).
Everything the summary needs is an attribute on an ``llm`` or ``tool`` span, so
the summary is a tree walk.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from dungeon.artifacts.llm import LLM, Completion, Message

INPUT_PRICE_PER_MILLION = 3.0  # USD per million input tokens (illustrative)
OUTPUT_PRICE_PER_MILLION = 15.0  # USD per million output tokens (illustrative)
REDACTED = "[REDACTED]"
FIELD_ORDER = ("name", "kind", "start", "end", "status", "attributes", "children")


@dataclass
class Span:
    """One timed operation. ``kind`` is "llm", "tool", or anything else ("chain", "internal")."""

    name: str
    kind: str
    start: float
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    status: str = "ok"  # "ok" | "error"

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            raise ValueError(f"span {self.name!r} is still open")
        return (self.end - self.start) * 1000.0


def walk(span: Span, depth: int = 1) -> Iterator[tuple[Span, int]]:
    """Pre-order walk yielding (span, depth); the root has depth 1."""
    yield span, depth
    for child in span.children:
        yield from walk(child, depth + 1)


class Tracer:
    """Records spans from an injectable clock. Root spans land in ``traces``."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.traces: list[Span] = []
        self._stack: list[Span] = []

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attributes):
        span = Span(name=name, kind=kind, start=float(self.clock()), attributes=dict(attributes))
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.traces.append(span)
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


# ------------------------------------------------------------------ recording


def record_llm_call(
    tracer: Tracer,
    llm: LLM,
    messages: Sequence[Message],
    *,
    model: str = "mock",
    input_price_per_million: float = INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = OUTPUT_PRICE_PER_MILLION,
) -> Completion:
    """Call ``llm.complete(messages)`` inside an ``llm`` span that records usage and cost."""
    with tracer.span("llm.complete", "llm", model=model, n_messages=len(messages)) as span:
        completion = llm.complete(messages)
        usage = completion.usage
        span.attributes["input_tokens"] = int(usage.input_tokens)
        span.attributes["output_tokens"] = int(usage.output_tokens)
        span.attributes["cost_usd"] = (
            usage.input_tokens * input_price_per_million / 1e6
            + usage.output_tokens * output_price_per_million / 1e6
        )
        span.attributes["stop_reason"] = completion.stop_reason
    return completion


def record_tool_call(tracer: Tracer, name: str, fn: Callable, **arguments):
    """Call ``fn(**arguments)`` inside a ``tool`` span. Errors mark the span and propagate."""
    with tracer.span(name, "tool", arguments=dict(arguments)) as span:
        result = fn(**arguments)
        span.attributes["result_chars"] = len(str(result))
    return result


# -------------------------------------------------------------------- summary


def summarize(trace: Span) -> dict:
    """Totals over the whole tree: time, tokens, cost, call counts, errors and depth."""
    out = {
        "total_ms": trace.duration_ms,
        "llm_ms": 0.0,
        "tool_ms": 0.0,
        "tokens": 0,
        "cost": 0.0,
        "n_llm_calls": 0,
        "n_tool_calls": 0,
        "n_errors": 0,
        "depth": 0,
    }
    for span, depth in walk(trace):
        out["depth"] = max(out["depth"], depth)
        if span.status == "error":
            out["n_errors"] += 1
        if span.kind == "llm":
            out["n_llm_calls"] += 1
            out["llm_ms"] += span.duration_ms
            out["tokens"] += int(span.attributes.get("input_tokens", 0)) + int(span.attributes.get("output_tokens", 0))
            out["cost"] += float(span.attributes.get("cost_usd", 0.0))
        elif span.kind == "tool":
            out["n_tool_calls"] += 1
            out["tool_ms"] += span.duration_ms
    return out


# ----------------------------------------------------------------------- json


def span_to_dict(span: Span) -> dict:
    """A JSON-ready dict with keys in FIELD_ORDER, children recursively."""
    return {
        "name": span.name,
        "kind": span.kind,
        "start": span.start,
        "end": span.end,
        "status": span.status,
        "attributes": dict(span.attributes),
        "children": [span_to_dict(child) for child in span.children],
    }


def span_from_dict(data: dict) -> Span:
    return Span(
        name=data["name"],
        kind=data["kind"],
        start=data["start"],
        end=data.get("end"),
        attributes=dict(data.get("attributes", {})),
        children=[span_from_dict(child) for child in data.get("children", [])],
        status=data.get("status", "ok"),
    )


def to_json(trace: Span, indent: int | None = None) -> str:
    """Serialize a trace with a stable field order (FIELD_ORDER) at every level."""
    return json.dumps(span_to_dict(trace), indent=indent, default=str)


def from_json(text: str) -> Span:
    return span_from_dict(json.loads(text))


# ------------------------------------------------------------------ redaction


def redact_attributes(trace: Span, keys: Iterable[str], placeholder: str = REDACTED) -> Span:
    """A deep copy of ``trace`` with every attribute named in ``keys`` (case-insensitive) replaced."""
    wanted = {key.lower() for key in keys}
    clone = copy.deepcopy(trace)
    for span, _depth in walk(clone):
        for key in list(span.attributes):
            if key.lower() in wanted:
                span.attributes[key] = placeholder
    return clone
