"""ROOM 13.1 - THE LEDGER OF SPANS

    The lift doors open onto a round stone room lined with ledgers. Every
    request the dungeon has ever served is written down here: when it began,
    what it called, what that called, how long each part took, what it cost.
    The keeper does not look up. "Nothing is a mystery," she says, "if you
    wrote it down while it was happening."

A TRACE is a tree of SPANS. A span is one timed operation: a name, a kind
("llm", "tool", "chain", ...), a start and end time from a clock, a dict of
attributes, its children, and a status ("ok" or "error"). An agent run is one
root span; each model call and each tool call is a child; a tool that calls a
model has a grandchild. Almost every production question ("why was this slow?",
"what did that cost?", "which tool failed?") is a query over this tree.

Three engineering rules that the trial enforces:

1. NESTING COMES FROM A STACK. ``Tracer.span`` is a context manager. On entry it
   creates the span, attaches it to the span currently on top of the stack (or
   makes it a root trace when the stack is empty), and pushes it. On exit it
   records the end time and pops. That is all nesting needs.
2. THE CLOCK IS INJECTED. ``Tracer(clock)`` takes any callable returning
   seconds. Real code passes ``time.perf_counter``; the trial passes a fake it
   can advance by hand, so every duration is exact. A tracer that reads the
   wall clock directly cannot be tested and will flake.
3. ERRORS ARE RECORDED AND RE-RAISED. An exception inside a span sets
   ``status = "error"``, stores ``error.type`` and ``error.message`` as
   attributes, still closes the span (``finally``), still pops the stack, and
   propagates. Swallowing the exception would turn every failure into a quiet
   success; forgetting to pop would make every later span a child of the corpse.

Model calls are the expensive spans. ``record_llm_call`` wraps
``llm.complete(messages)`` and records the usage the completion reports plus a
cost in USD at a per-million-token price. Tool calls record their arguments and
whether they worked. ``summarize`` walks the tree once and adds it all up.
``to_json`` writes the tree with the SAME field order at every level, so two
traces of the same run produce byte-identical text and log queries do not
break when a field moves. ``redact_attributes`` exists because traces get
shipped to third-party dashboards and the user's email address should not.

The mock models from ``dungeon/artifacts/llm.py`` report usage the same way a
real adapter does (``completion.usage.input_tokens`` / ``output_tokens``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from dungeon.artifacts.llm import LLM, Completion, Message  # noqa: F401 - Message for type hints

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
    """Pre-order walk yielding (span, depth); the root has depth 1. Provided: use it in summarize/redact."""
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
        """The innermost open span, or None when nothing is open."""
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attributes):
        """Open a span, yield it, close it.

        Entry: ``Span(name, kind, start=self.clock(), attributes=dict(attributes))``; append it to the
        parent's ``children`` if the stack is non-empty, else to ``self.traces``; push it.
        Exit: on an exception set ``status="error"``, ``attributes["error.type"]`` (the class name) and
        ``attributes["error.message"]``, then re-raise. Always (``finally``) set ``end = self.clock()``
        and pop the stack.
        """
        raise NotImplementedError("Tracer.span() is unwritten")


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
    """Call ``llm.complete(messages)`` inside a span named "llm.complete" of kind "llm"; return the Completion.

    Record on the span: ``model``, ``input_tokens`` and ``output_tokens`` from ``completion.usage``,
    ``cost_usd = input_tokens * input_price / 1e6 + output_tokens * output_price / 1e6``, and
    ``stop_reason``. If the model raises, the span's context manager marks the error; let it propagate.
    """
    raise NotImplementedError("record_llm_call() is unwritten")


def record_tool_call(tracer: Tracer, name: str, fn: Callable, **arguments):
    """Call ``fn(**arguments)`` inside a span named ``name`` of kind "tool"; return the result.

    Record ``arguments`` (a dict) as an attribute. A tool that raises leaves an "error" span behind
    and the exception propagates.
    """
    raise NotImplementedError("record_tool_call() is unwritten")


# -------------------------------------------------------------------- summary


def summarize(trace: Span) -> dict:
    """Totals over the whole tree, as Python numbers:

    {"total_ms": root duration, "llm_ms": sum of llm span durations, "tool_ms": same for tools,
     "tokens": sum of input_tokens + output_tokens over llm spans, "cost": sum of cost_usd,
     "n_llm_calls", "n_tool_calls", "n_errors": spans with status "error", "depth": deepest level (root = 1)}
    """
    raise NotImplementedError("summarize() is unwritten")


# ----------------------------------------------------------------------- json


def span_to_dict(span: Span) -> dict:
    """A JSON-ready dict whose keys are exactly FIELD_ORDER, in that order, children converted recursively."""
    raise NotImplementedError("span_to_dict() is unwritten")


def span_from_dict(data: dict) -> Span:
    """Inverse of span_to_dict."""
    raise NotImplementedError("span_from_dict() is unwritten")


def to_json(trace: Span, indent: int | None = None) -> str:
    """``json.dumps(span_to_dict(trace), indent=indent)``. Same trace, same text, every time."""
    raise NotImplementedError("to_json() is unwritten")


def from_json(text: str) -> Span:
    """Inverse of to_json: ``from_json(to_json(t)) == t``."""
    raise NotImplementedError("from_json() is unwritten")


# ------------------------------------------------------------------ redaction


def redact_attributes(trace: Span, keys: Iterable[str], placeholder: str = REDACTED) -> Span:
    """A DEEP COPY of ``trace`` in which every attribute whose name matches one of ``keys``
    (case-insensitively), at any depth, has its value replaced by ``placeholder``.

    The original trace must not change: ``copy.deepcopy`` first, then walk the copy.
    """
    raise NotImplementedError("redact_attributes() is unwritten")
