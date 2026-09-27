"""TRIAL 13.1 - THE LEDGER OF SPANS

Spans nest by a stack, durations come from the clock you were handed, errors
are recorded and re-raised, secrets are redacted, and the ledger survives a
trip through JSON with its columns in the same order.
"""

import json
import math

import pytest

from dungeon.artifacts.llm import Completion, Message, ScriptedLLM
from dungeon.trials import load_room

room = load_room(__file__, "room_1_ledger_of_spans")


class FakeClock:
    """A clock that only moves when the trial says so. Timing assertions become exact."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


def _ticking_reply(clock, seconds, text):
    """A scripted LLM reply that takes ``seconds`` of fake time to arrive."""

    def respond(_messages):
        clock.tick(seconds)
        return text

    return respond


def _agent_trace():
    """One agent run: 10 ms of thinking, a 200 ms LLM call, a 50 ms tool, a 100 ms LLM call, 1 ms of tidying.

    Root:  371 ms total. llm_ms = 300, tool_ms = 50, depth 2.
    """
    clock = FakeClock()
    tracer = room.Tracer(clock)
    llm = ScriptedLLM([_ticking_reply(clock, 0.200, "Look it up."), _ticking_reply(clock, 0.100, "Done.")])

    def lookup(query):
        clock.tick(0.050)
        return f"{query}: Floor 13"

    with tracer.span("agent.run", "chain", user_email="ada@example.test", api_key="sk-very-secret", request_id="req-1"):
        clock.tick(0.010)
        room.record_llm_call(tracer, llm, [Message.user("Where is the watchtower?")])
        clock.tick(0.005)
        room.record_tool_call(tracer, "lookup", lookup, query="watchtower")
        clock.tick(0.005)
        room.record_llm_call(tracer, llm, [Message.user("Answer.")])
        clock.tick(0.001)
    return tracer, tracer.traces[0]


# ---------------------------------------------------------------------- spans
def test_a_span_reads_its_times_from_the_clock_it_was_given():
    clock = FakeClock(start=5.0)
    tracer = room.Tracer(clock)
    with tracer.span("think", "internal", note="hmm") as span:
        clock.tick(0.25)
    assert isinstance(span, room.Span)
    assert span.name == "think" and span.kind == "internal"
    assert span.start == 5.0 and span.end == 5.25, f"start/end must come from the injected clock; got {span.start}/{span.end}."
    assert math.isclose(span.duration_ms, 250.0), f"5.0 -> 5.25 s is 250 ms; you report {span.duration_ms}."
    assert span.attributes.get("note") == "hmm", "Keyword arguments to span() become attributes."
    assert span.status == "ok"
    assert tracer.traces == [span], "A span opened with nothing on the stack is a root trace."
    assert tracer.current is None, "After the block, the stack is empty again."


def test_spans_nest_by_the_stack():
    tracer = room.Tracer(FakeClock())
    with tracer.span("root", "chain") as root:
        assert tracer.current is root
        with tracer.span("child_a", "tool") as a:
            assert tracer.current is a, "The innermost open span is the current one."
            with tracer.span("grandchild", "llm") as g:
                pass
        with tracer.span("child_b", "llm") as b:
            pass
    assert tracer.traces == [root], f"Only the root is a trace; children hang off their parent. traces has {len(tracer.traces)} entries."
    assert root.children == [a, b], "Children are recorded on the parent, in the order they opened."
    assert a.children == [g], "The grandchild belongs to child_a, which was on top of the stack when it opened."
    assert b.children == []


def test_an_exception_marks_the_span_and_still_escapes():
    clock = FakeClock()
    tracer = room.Tracer(clock)
    with pytest.raises(ZeroDivisionError):
        with tracer.span("divide", "tool") as span:
            clock.tick(0.010)
            1 / 0  # noqa: B018 - the point is the exception
    assert span.status == "error", f"A span that raised has status 'error', not {span.status!r}."
    assert span.attributes.get("error.type") == "ZeroDivisionError", f"Record the exception type under 'error.type'; got {span.attributes}."
    assert span.attributes.get("error.message") == "division by zero", (
        f"Record str(exc) under 'error.message' so the ledger says what went wrong, not just that it did; got {span.attributes}."
    )
    assert span.end == 0.010, "The span must still be closed (end set) when the exception passes through: use finally."
    assert tracer.current is None, "The stack must be popped even on error, or every later span becomes a child of the corpse."
    assert tracer.traces == [span], "The failed trace is still recorded; failures are what you will be reading at 2 a.m."
    with tracer.span("next", "internal") as nxt:
        pass
    assert tracer.traces == [span, nxt], "After the error, a new span is a new root, not a child of the failed one."


# ------------------------------------------------------------------- recording
def test_record_llm_call_records_usage_and_cost():
    tracer = room.Tracer(FakeClock())
    llm = ScriptedLLM(["Twelve gold."])
    messages = [Message.system("Be brief."), Message.user("What does a potion cost?")]
    completion = room.record_llm_call(tracer, llm, messages, model="scribe-v1")
    assert isinstance(completion, Completion) and completion.text == "Twelve gold.", "Return the Completion; callers still need the answer."
    assert len(llm.calls) == 1, "Exactly one call to the model."
    span = tracer.traces[0]
    assert span.kind == "llm", f"An LLM call is a span of kind 'llm', not {span.kind!r}."
    attrs = span.attributes
    usage = completion.usage
    assert attrs.get("input_tokens") == usage.input_tokens and attrs.get("output_tokens") == usage.output_tokens, (
        f"Record completion.usage on the span: expected {usage.input_tokens}/{usage.output_tokens}, got {attrs}."
    )
    expected_cost = usage.input_tokens * room.INPUT_PRICE_PER_MILLION / 1e6 + usage.output_tokens * room.OUTPUT_PRICE_PER_MILLION / 1e6
    assert math.isclose(attrs.get("cost_usd", -1.0), expected_cost), (
        f"cost_usd = input * {room.INPUT_PRICE_PER_MILLION}/1e6 + output * {room.OUTPUT_PRICE_PER_MILLION}/1e6 = {expected_cost:.8f}; got {attrs.get('cost_usd')}."
    )
    assert attrs.get("model") == "scribe-v1", "Record which model answered; you will compare models later."
    assert attrs.get("stop_reason") == completion.stop_reason, (
        f"Record completion.stop_reason ({completion.stop_reason!r}): a model that keeps stopping on max_tokens is truncating answers, "
        f"a quality regression you can see without labels. Got {attrs.get('stop_reason')!r}."
    )


def test_record_tool_call_returns_the_result_and_marks_failures():
    tracer = room.Tracer(FakeClock())

    def add(a, b):
        return a + b

    assert room.record_tool_call(tracer, "add", add, a=2, b=3) == 5, "Return the tool's result."
    span = tracer.traces[0]
    assert span.kind == "tool" and span.name == "add" and span.status == "ok"
    assert span.attributes.get("arguments") == {"a": 2, "b": 3}, f"Record the arguments; got {span.attributes}."

    def explode(**_kwargs):
        raise RuntimeError("the tool is on fire")

    with pytest.raises(RuntimeError):
        room.record_tool_call(tracer, "explode", explode, x=1)
    failed = tracer.traces[1]
    assert failed.status == "error" and failed.attributes.get("error.type") == "RuntimeError"


# --------------------------------------------------------------------- summary
def test_summarize_adds_up_the_ledger():
    _tracer, trace = _agent_trace()
    summary = room.summarize(trace)
    for key in ("total_ms", "llm_ms", "tool_ms", "tokens", "cost", "n_llm_calls", "n_tool_calls", "depth"):
        assert key in summary, f"summarize() is missing {key!r}."
    assert math.isclose(summary["total_ms"], 371.0), f"10 + 200 + 5 + 50 + 5 + 100 + 1 = 371 ms in total; you report {summary['total_ms']}."
    assert math.isclose(summary["llm_ms"], 300.0), f"Two LLM calls of 200 and 100 ms: llm_ms = 300; you report {summary['llm_ms']}."
    assert math.isclose(summary["tool_ms"], 50.0), f"One 50 ms tool call; you report tool_ms = {summary['tool_ms']}."
    assert summary["n_llm_calls"] == 2 and summary["n_tool_calls"] == 1
    assert summary["depth"] == 2, f"Root plus one level of children is depth 2; you report {summary['depth']}."
    llm_spans = [c for c in trace.children if c.kind == "llm"]
    expected_tokens = sum(c.attributes["input_tokens"] + c.attributes["output_tokens"] for c in llm_spans)
    expected_cost = sum(c.attributes["cost_usd"] for c in llm_spans)
    assert summary["tokens"] == expected_tokens, f"tokens is the sum of input and output tokens over llm spans: {expected_tokens}, not {summary['tokens']}."
    assert math.isclose(summary["cost"], expected_cost), "cost is the sum of cost_usd over llm spans."
    assert summary.get("n_errors", 0) == 0


def test_summarize_reports_depth_and_errors():
    tracer = room.Tracer(FakeClock())
    with tracer.span("root", "chain"):
        with tracer.span("step", "chain"):
            with pytest.raises(ValueError):
                with tracer.span("deep", "tool"):
                    raise ValueError("bad input")
    summary = room.summarize(tracer.traces[0])
    assert summary["depth"] == 3, f"root -> step -> deep is depth 3; you report {summary['depth']}."
    assert summary["n_errors"] == 1, f"One span failed; n_errors should be 1, not {summary['n_errors']}."


# ------------------------------------------------------------------------ json
def test_to_json_keeps_a_stable_field_order_and_round_trips():
    _tracer, trace = _agent_trace()
    text = room.to_json(trace)
    data = json.loads(text)
    assert list(data.keys()) == list(room.FIELD_ORDER), (
        f"Top-level keys must follow FIELD_ORDER {list(room.FIELD_ORDER)} so diffs and log queries are stable; got {list(data.keys())}."
    )
    assert list(data["children"][0].keys()) == list(room.FIELD_ORDER), "Children use the same field order."
    back = room.from_json(text)
    assert back == trace, "from_json(to_json(trace)) must equal the original (dataclass equality)."
    assert room.to_json(back) == text, "Serializing the round-tripped trace gives the identical text."


# ------------------------------------------------------------------- redaction
def test_redaction_scrubs_named_attributes_everywhere_and_leaves_the_original_alone():
    tracer = room.Tracer(FakeClock())
    with tracer.span("root", "chain", user_email="ada@example.test", API_KEY="sk-1", note="keep"):
        with tracer.span("child", "tool", user_email="ada@example.test", arguments={"q": 1}):
            pass
    original = tracer.traces[0]
    clean = room.redact_attributes(original, ["user_email", "api_key"])
    assert clean.attributes["user_email"] == room.REDACTED and clean.attributes["API_KEY"] == room.REDACTED, (
        f"Both named attributes must be replaced with {room.REDACTED!r} (matching is case-insensitive); got {clean.attributes}."
    )
    assert clean.attributes["note"] == "keep", "Only the named attributes are redacted."
    assert clean.children[0].attributes["user_email"] == room.REDACTED, "Redaction must reach every span in the tree."
    assert clean.children[0].attributes["arguments"] == {"q": 1}
    assert original.attributes["user_email"] == "ada@example.test" and original.children[0].attributes["user_email"] == "ada@example.test", (
        "redact_attributes returns a copy; the original ledger must be untouched."
    )
    assert "sk-1" not in room.to_json(clean), "The secret must not survive into the serialized trace."
