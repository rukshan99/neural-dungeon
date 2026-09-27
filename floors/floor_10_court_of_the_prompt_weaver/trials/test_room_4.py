"""TRIAL 10.4 - THE LEDGER

The Ledger has a fixed number of pages. Tear out the oldest, keep the rules
and the latest petition, and never separate a herald's report from its summons.
"""

import pytest

from dungeon.artifacts.llm import (
    Completion,
    Message,
    ScriptedLLM,
    ToolCall,
    Usage,
    approx_messages_tokens,
)
from dungeon.trials import load_room

room = load_room(__file__, "room_4_the_ledger")

SYSTEM = Message.system("You are the court's ledger-keeper. Answer briefly.")


def make_history(turns):
    history = [SYSTEM]
    for i in range(turns):
        history.append(Message.user(f"Petition number {i}: may I have a torch?"))
        history.append(Message.assistant(f"Reply number {i}: one torch, granted."))
    return history


def tokens(messages):
    return approx_messages_tokens(messages)


# --------------------------------------------------------------------- budget
def test_available_is_the_window_minus_the_reply():
    budget = room.ContextBudget(max_tokens=1000, reserve_for_output=200)
    assert budget.available == 800, f"max_tokens 1000 minus reserve 200 leaves 800 for the prompt, you said {budget.available}."
    small = [Message.user("hi")]
    assert budget.fits(small) is True
    assert room.ContextBudget(max_tokens=4, reserve_for_output=0).fits(small) is False, (
        f"'hi' costs {tokens(small)} approx tokens; it does not fit in 4."
    )


def test_when_everything_fits_nothing_is_dropped():
    history = make_history(3)
    kept = room.fit_messages(history, room.ContextBudget(max_tokens=100_000))
    assert kept == history, "Under budget, fit_messages returns the whole history in order."
    assert all(a is b for a, b in zip(kept, history)), "Return the original Message objects, not copies."


def test_the_oldest_turns_fall_first_and_the_rules_stay():
    history = make_history(6)
    target = [SYSTEM] + history[-4:]
    budget = room.ContextBudget(max_tokens=tokens(target), reserve_for_output=0)
    kept = room.fit_messages(history, budget)
    assert kept[0] is SYSTEM, f"The system message must survive; the first kept message has role {kept[0].role!r}."
    assert kept[-1] is history[-1], "The newest message must survive."
    assert budget.fits(kept), f"Result uses {tokens(kept)} tokens but only {budget.available} are available."
    assert kept == target, (
        f"Expected the system message plus the last 4 messages ({len(target)} total), got {len(kept)}: "
        f"{[m.content[:24] for m in kept]}. Drop the OLDEST non-system messages first."
    )
    positions = [history.index(m) for m in kept]
    assert positions == sorted(positions), "Keep the original order."


def test_the_latest_petition_is_never_dropped():
    history = make_history(4)
    budget = room.ContextBudget(max_tokens=tokens([SYSTEM]) + 1, reserve_for_output=0)
    kept = room.fit_messages(history, budget)
    assert kept == [SYSTEM, history[-2]], (
        "Even when the budget is hopeless, keep the system message and the latest user message: "
        f"the question is why you are calling the model at all. Got roles {[m.role for m in kept]}"
    )


def test_a_heralds_report_never_survives_without_its_summons():
    c1 = ToolCall("c1", "search_archive", {"query": "torch"})
    c2 = ToolCall("c2", "read_document", {"doc_id": "torches"})
    history = [
        SYSTEM,
        Message.user("Where are the torches kept?"),
        Message.assistant("Let me look.", [c1]),
        Message.tool("c1", "Matching documents: torches, lanterns, the long inventory of the east wing " * 4, name="search_archive"),
        Message.assistant("They are in the east wing."),
        Message.user("And the key to the east wing?"),
        Message.assistant("Reading.", [c2]),
        Message.tool("c2", "The key hangs by the kitchen door.", name="read_document"),
        Message.assistant("By the kitchen door."),
        Message.user("Thank you. Summarise all of that."),
    ]
    # A budget that a naive per-message dropper would satisfy by removing only the first user
    # message and the first assistant message, leaving the tool result c1 orphaned.
    budget = room.ContextBudget(
        max_tokens=tokens(history) - tokens([history[1], history[2]]),
        reserve_for_output=0,
    )
    kept = room.fit_messages(history, budget)
    assert budget.fits(kept), f"Result uses {tokens(kept)} tokens, available {budget.available}."
    kept_ids = {m.tool_call_id for m in kept if m.role == "tool"}
    requested_ids = {c.id for m in kept if m.role == "assistant" for c in m.tool_calls}
    assert kept_ids <= requested_ids, (
        f"Orphaned tool result(s) {sorted(kept_ids - requested_ids)}: a tool message survived without the assistant "
        "message that requested it. Drop them as one unit."
    )
    assert requested_ids <= kept_ids, (
        f"Unanswered tool call(s) {sorted(requested_ids - kept_ids)}: an assistant message with tool_calls survived "
        "without its tool results. Drop them as one unit."
    )
    assert kept[0] is SYSTEM and kept[-1] is history[-1]
    assert history[3] not in kept, "The first exchange (c1) should be gone: it is the oldest unit big enough to make room."


# ---------------------------------------------------------------------- cost
def test_the_meter_adds_up_exactly():
    meter = room.CostMeter(price_in_per_million=3.0, price_out_per_million=15.0)
    meter.record(Usage(1_000_000, 0))
    assert meter.cost == pytest.approx(3.0), f"A million input tokens at $3/M is $3.00, you billed {meter.cost}"
    meter.record(Usage(0, 500_000))
    assert meter.cost == pytest.approx(10.5), f"Plus half a million output tokens at $15/M is $10.50, you billed {meter.cost}"
    meter.record(Usage(250_000, 100_000))
    assert meter.cost == pytest.approx(12.75), f"Plus $0.75 + $1.50 is $12.75, you billed {meter.cost}"
    assert meter.usage.total_tokens == 1_850_000, f"Total tokens should be 1,850,000, got {meter.usage.total_tokens}"
    assert meter.calls == 3


def test_the_meter_accepts_a_completion():
    meter = room.CostMeter(1.0, 2.0)
    meter.record(Completion(Message.assistant("x"), usage=Usage(10, 20)))
    assert (meter.usage.input_tokens, meter.usage.output_tokens) == (10, 20)
    assert meter.cost == pytest.approx((10 * 1.0 + 20 * 2.0) / 1e6)


# ------------------------------------------------------------------- summary
def test_summarize_history_returns_a_system_note_built_from_the_transcript():
    old = make_history(3)[1:]
    llm = ScriptedLLM(["Three petitions for torches, all granted."])
    note = room.summarize_history(llm, old)
    assert isinstance(note, Message) and note.role == "system", f"The summary is a system-role note, got role {getattr(note, 'role', None)!r}"
    assert note.content.startswith("Summary of earlier conversation: "), f"Prefix the note with 'Summary of earlier conversation: '. Got {note.content!r}"
    assert "Three petitions for torches" in note.content, "The model's summary text belongs in the note."
    assert len(llm.calls) == 1
    sent = "\n".join(m.content for m in llm.calls[0]["messages"])
    assert "Petition number 2" in sent, "The summariser must be shown the transcript it is summarising."


def test_compaction_keeps_the_recent_turns_verbatim_and_remembers_the_rest():
    history = make_history(8)
    budget = room.ContextBudget(max_tokens=tokens([SYSTEM] + history[-4:]) + 40, reserve_for_output=0)
    llm = ScriptedLLM(["Eight petitions for torches were granted earlier."])
    out = room.compact(history, budget, llm)
    assert out[-4:] == history[-4:], "The most recent turns must be kept verbatim; only the dropped prefix is summarised."
    assert out[0] is SYSTEM, "The original system prompt stays first."
    summaries = [m for m in out if m.role == "system" and m.content.startswith("Summary of earlier conversation: ")]
    assert len(summaries) == 1, f"Expected exactly one summary note, found {len(summaries)}."
    assert "Eight petitions" in summaries[0].content
    first_non_system = next(i for i, m in enumerate(out) if m.role != "system")
    assert out.index(summaries[0]) < first_non_system, "The summary precedes the kept conversation."
    assert budget.fits(out), f"After compaction the history must fit: {tokens(out)} > {budget.available}."
    assert len(llm.calls) == 1, "Summarise once, not once per dropped message."
    sent = "\n".join(m.content for m in llm.calls[0]["messages"])
    assert "Petition number 0" in sent, "The dropped turns are what the summariser must see."


def test_compaction_does_nothing_when_the_ledger_has_room():
    history = make_history(2)
    llm = ScriptedLLM([])  # any call would raise ScriptExhausted
    out = room.compact(history, room.ContextBudget(max_tokens=100_000), llm)
    assert out == history, "Nothing to drop, nothing to summarise, nothing to pay for."
    assert llm.calls == [], "Do not call the model when the history already fits."
