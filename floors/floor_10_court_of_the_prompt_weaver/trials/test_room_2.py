"""TRIAL 10.2 - THE SUMMONING LOOP

The Weaver names heralds; the heralds report; the transcript must stay in
order. Nothing a herald does may crash the court.
"""

import pytest

from dungeon.artifacts.llm import Message, ScriptedLLM, ToolSpec
from dungeon.trials import load_room

room = load_room(__file__, "room_2_the_summoning_loop")

NO_PARAMS = {"type": "object", "properties": {}}
ITEM_PARAMS = {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}


def make_court(log):
    def look_around():
        log.append("look")
        return "A torch flickers on the wall."

    def take(item):
        log.append(f"take:{item}")
        return f"You take the {item}."

    def count_coins():
        log.append("count")
        return {"coins": 42}

    def cursed_mirror():
        raise RuntimeError("the mirror shatters")

    registry = room.ToolRegistry()
    registry.register(room.Tool("look_around", "Describe the room.", NO_PARAMS, look_around))
    registry.register(room.Tool("take", "Pick something up.", ITEM_PARAMS, take))
    registry.register(room.Tool("count_coins", "Count the coins in the purse.", NO_PARAMS, count_coins))
    registry.register(room.Tool("cursed_mirror", "Do not look.", NO_PARAMS, cursed_mirror))
    return registry


def call(name, arguments=None, id=None):
    return {"id": id or f"call_{name}", "name": name, "arguments": arguments or {}}


# ------------------------------------------------------------------- registry
def test_specs_describe_every_herald_for_the_model():
    registry = make_court([])
    specs = registry.specs()
    assert all(isinstance(s, ToolSpec) for s in specs), "specs() must return dungeon.artifacts.llm.ToolSpec objects."
    assert [s.name for s in specs] == ["look_around", "take", "count_coins", "cursed_mirror"], (
        f"One spec per registered tool, in registration order. Got {[s.name for s in specs]}"
    )
    take = next(s for s in specs if s.name == "take")
    assert take.parameters == ITEM_PARAMS, "The ToolSpec's parameters must be the tool's JSON schema, untouched."
    assert take.description == "Pick something up."


def test_a_herald_returning_data_is_serialised_for_the_model():
    registry = make_court([])
    assert registry.call("count_coins", {}) == '{"coins": 42}', "Non-string results are json.dumps'ed so the model reads text."
    assert registry.call("take", {"item": "torch"}) == "You take the torch."
    assert registry.call("look_around", None) == "A torch flickers on the wall.", "arguments may be None; treat it as {}."


def test_an_unknown_herald_is_reported_not_raised():
    registry = make_court([])
    reply = registry.call("summon_dragon", {})
    assert isinstance(reply, str) and reply.startswith(room.TOOL_ERROR_PREFIX), (
        f"An unknown tool is a string starting with {room.TOOL_ERROR_PREFIX!r} for the model to read, not an exception. Got {reply!r}"
    )
    assert "summon_dragon" in reply, f"Name the tool that was not found so the model can correct itself: {reply!r}"
    assert "look_around" in reply, f"List the available tools in the error so the model can pick one: {reply!r}"


def test_a_herald_that_explodes_becomes_a_message():
    registry = make_court([])
    reply = registry.call("cursed_mirror", {})
    assert reply.startswith(room.TOOL_ERROR_PREFIX), f"A tool exception becomes a ToolError string, never propagates. Got {reply!r}"
    assert "RuntimeError" in reply and "shatters" in reply, f"Include the exception type and message: {reply!r}"


def test_wrong_arguments_are_a_tool_error_too():
    registry = make_court([])
    reply = registry.call("take", {"weapon": "sword"})
    assert reply.startswith(room.TOOL_ERROR_PREFIX), (
        f"The model passed a wrong argument name; that TypeError must go back to it as text, not crash the loop. Got {reply!r}"
    )


# ----------------------------------------------------------------------- loop
def test_two_rounds_of_summoning_then_an_answer():
    log = []
    registry = make_court(log)
    llm = ScriptedLLM([
        {"tool_calls": [call("look_around", id="c1")]},
        {"tool_calls": [call("take", {"item": "torch"}, id="c2")]},
        "You now hold a torch.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Get me a light.")])
    assert isinstance(result, room.AgentResult)
    assert result.text == "You now hold a torch.", f"The final text is the model's last reply; got {result.text!r}"
    assert result.steps == 3, f"Three model calls were made; you counted steps={result.steps}."
    assert result.tool_calls_made == 2, f"Two tools were executed; you counted {result.tool_calls_made}."
    assert log == ["look", "take:torch"], f"Tools must run in the order requested, with their arguments. Log: {log}"


def test_the_transcript_has_exactly_the_right_shape():
    registry = make_court([])
    llm = ScriptedLLM([
        {"tool_calls": [call("look_around", id="c1")]},
        {"tool_calls": [call("take", {"item": "torch"}, id="c2")]},
        "Done.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Get me a light.")])
    roles = [m.role for m in result.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"], (
        f"Expected user, assistant(tool_calls), tool, assistant(tool_calls), tool, assistant. Got {roles}. "
        "Append the completion's own message (it carries the tool_calls), then one tool message per call."
    )
    assert result.messages[1].tool_calls and result.messages[1].tool_calls[0].id == "c1", (
        "The assistant message you append must be the completion's message, tool_calls included."
    )
    assert result.messages[2].tool_call_id == "c1", f"The tool message must carry the id of the call it answers; got {result.messages[2].tool_call_id!r}"
    assert result.messages[2].name == "look_around", f"Set name= on the tool message; got {result.messages[2].name!r}"
    assert result.messages[2].content == "A torch flickers on the wall."
    assert result.messages[4].tool_call_id == "c2" and result.messages[4].name == "take"


def test_parallel_summons_are_all_answered_in_order():
    log = []
    registry = make_court(log)
    llm = ScriptedLLM([
        {"tool_calls": [call("look_around", id="a"), call("count_coins", id="b")]},
        "A torch and 42 coins.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Look, then count.")])
    assert result.steps == 2 and result.tool_calls_made == 2, (
        f"Two calls in ONE completion are executed in one step: steps={result.steps}, tool_calls_made={result.tool_calls_made}"
    )
    assert log == ["look", "count"], f"Execute every call in the completion, in order. Log: {log}"
    ids = [m.tool_call_id for m in result.messages if m.role == "tool"]
    assert ids == ["a", "b"], f"One tool message per call, in the same order as the calls. Got ids {ids}"
    assert result.messages[3].content == '{"coins": 42}'


def test_the_model_is_shown_the_heralds():
    registry = make_court([])
    llm = ScriptedLLM(["Nothing to do."])
    room.run_tool_loop(llm, registry, [Message.user("Hello.")])
    offered = [t.name for t in llm.calls[0]["tools"]]
    assert offered == registry.names(), f"Pass registry.specs() to llm.complete on every call. The model was offered {offered}"


def test_an_unknown_summons_does_not_end_the_court():
    registry = make_court([])
    llm = ScriptedLLM([
        {"tool_calls": [call("summon_dragon", id="d1")]},
        "No dragons today.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Dragon, please.")])
    assert result.text == "No dragons today."
    tool_msg = result.messages[2]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == "d1"
    assert tool_msg.content.startswith(room.TOOL_ERROR_PREFIX), (
        f"The unknown-tool error must be delivered to the model as the tool result: {tool_msg.content!r}"
    )


def test_an_exploding_herald_is_reported_and_the_loop_continues():
    registry = make_court([])
    llm = ScriptedLLM([
        {"tool_calls": [call("cursed_mirror", id="m1")]},
        "The mirror is broken; I looked away.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Look in the mirror.")])
    assert result.text == "The mirror is broken; I looked away."
    assert "shatters" in result.messages[2].content, (
        f"The exception's message must reach the model as the tool result: {result.messages[2].content!r}"
    )
    assert result.tool_calls_made == 1, "A failed tool call still counts as a call made."


def test_the_step_budget_ends_an_endless_summoning():
    registry = make_court([])

    def forever(messages):
        return {"tool_calls": [call("look_around")]}

    llm = ScriptedLLM([forever] * 50)
    with pytest.raises(room.StepBudgetExceeded):
        room.run_tool_loop(llm, registry, [Message.user("Keep looking.")], max_steps=5)
    assert len(llm.calls) == 5, (
        f"max_steps=5 means at most five model calls; you made {len(llm.calls)}. "
        "Raise when the loop would make call number max_steps + 1."
    )


def test_a_reply_cut_at_max_tokens_is_final_not_another_turn():
    registry = make_court([])
    llm = ScriptedLLM([
        {"tool_calls": [call("look_around", id="c1")]},
        {"text": "The torch is on the", "stop_reason": "max_tokens"},
        "This reply must never be requested.",
    ])
    result = room.run_tool_loop(llm, registry, [Message.user("Describe the room at length.")])
    assert result.text == "The torch is on the", f"A reply cut at max_tokens is still the final text; got {result.text!r}"
    assert len(llm.calls) == 2 and result.steps == 2, (
        f"Only stop_reason 'tool_use' asks for another turn. 'max_tokens' means the model was cut off, not that it "
        f"wants tools; return what it said. You made {len(llm.calls)} model call(s)."
    )
    assert result.messages[-1].role == "assistant" and result.messages[-1].content == "The torch is on the"


def test_the_callers_scroll_is_not_scribbled_on():
    registry = make_court([])
    original = [Message.system("Court rules."), Message.user("Look.")]
    snapshot = list(original)
    llm = ScriptedLLM([{"tool_calls": [call("look_around")]}, "Done."])
    result = room.run_tool_loop(llm, registry, original)
    assert original == snapshot, f"run_tool_loop must work on a copy; the caller's list grew to {len(original)} messages."
    assert len(result.messages) == 5, "The returned transcript, on the other hand, has everything."
