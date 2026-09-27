"""TRIAL 10.1 - THE CONTRACT

Petitions arrive as prose, in fences and with the age written as a word. The
clerks must weave each into a valid contract or say exactly which clause broke.
"""

import json

import pytest

from dungeon.artifacts.llm import Message, ScriptedLLM
from dungeon.trials import load_room

room = load_room(__file__, "room_1_the_contract")

PETITION = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "role": {"type": "string", "enum": ["knight", "herald", "imp"]},
        "gear": {"type": "array", "items": {"type": "string"}},
        "address": {
            "type": "object",
            "properties": {"town": {"type": "string"}, "houses": {"type": "integer"}},
            "required": ["town"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "age", "role"],
    "additionalProperties": False,
}

GOOD = {"name": "Wren", "age": 31, "role": "herald", "gear": ["lantern", "ledger"]}
GOOD_JSON = json.dumps(GOOD)
BAD_AGE = json.dumps({**GOOD, "age": "thirty-one"})


# ------------------------------------------------------------------ extraction
EXTRACTION_CASES = [
    ("```json\n" + GOOD_JSON + "\n```", GOOD),
    ("```\n" + GOOD_JSON + "\n```", GOOD),
    ("Certainly! Here is the contract:\n" + GOOD_JSON + "\nLet me know if you need anything else.", GOOD),
    ("The heralds, in order: [1, 2, {\"name\": \"Wren\"}] and that is all.", [1, 2, {"name": "Wren"}]),
    ('{\n  "outer": {"inner": [1, {"deep": true}]},\n  "n": null\n}', {"outer": {"inner": [1, {"deep": True}]}, "n": None}),
]


@pytest.mark.parametrize(
    "reply,expected",
    EXTRACTION_CASES,
    ids=["json-fence", "bare-fence", "prose-wrapped", "array-in-prose", "nested-multiline"],
)
def test_the_clerk_finds_the_json_under_the_flourishes(reply, expected):
    got = room.extract_json(reply)
    assert isinstance(got, str), f"extract_json returns the JSON *text*; you returned {type(got).__name__}."
    assert json.loads(got) == expected, (
        f"extract_json returned {got!r}, which does not parse to the JSON hidden in the reply."
    )


def test_braces_inside_strings_do_not_close_the_contract():
    text = 'Here: {"motto": "we } never { yield", "n": 1} - thanks!'
    got = json.loads(room.extract_json(text))
    assert got == {"motto": "we } never { yield", "n": 1}, (
        f"A brace inside a string literal is just a character. You extracted {got!r}. "
        "Track whether you are inside quotes while you scan."
    )


def test_escaped_quotes_inside_strings_are_still_strings():
    text = r'Reply: {"quote": "he said \"}\" and left", "ok": true}'
    got = json.loads(room.extract_json(text))
    assert got == {"quote": 'he said "}" and left', "ok": True}, (
        f"A backslash escapes the next character, so \\\" does not end the string. Got {got!r}."
    )


def test_a_balanced_but_unparseable_block_is_skipped():
    text = 'Set the {flags} first, then reply: {"ok": true}'
    got = json.loads(room.extract_json(text))
    assert got == {"ok": True}, (
        f"'{{flags}}' is balanced but it is not JSON. Keep scanning for the next block. Got {got!r}."
    )


def test_a_reply_with_no_json_raises_value_error():
    with pytest.raises(ValueError):
        room.extract_json("The petition was eaten by a goat. There is no contract.")


# ------------------------------------------------------------------ validation
def test_a_valid_petition_has_no_errors():
    assert room.validate(GOOD, PETITION) == [], f"A valid petition should produce no errors, got {room.validate(GOOD, PETITION)}"


def test_the_error_path_names_the_clause():
    errors = room.validate({"age": "42"}, {"type": "object", "properties": {"age": {"type": "integer"}}})
    assert errors == ["$.age: expected integer, got str"], (
        f"Expected exactly ['$.age: expected integer, got str'], got {errors}. "
        "The path says where, the message says what; the type names are JSON's for the expectation "
        "and Python's for what arrived."
    )


def test_integer_is_not_number_and_bool_is_not_integer():
    assert room.validate(3, {"type": "integer"}) == []
    assert room.validate(3.0, {"type": "integer"}) != [], "3.0 is a number, not an integer. isinstance(3.0, int) is False; check it."
    assert room.validate(True, {"type": "integer"}) != [], (
        "True is not an integer in JSON, even though isinstance(True, int) is True in Python. Exclude bool."
    )
    assert room.validate(3, {"type": "number"}) == [], "Every integer is a number."
    assert room.validate(2.5, {"type": "number"}) == []
    assert room.validate(True, {"type": "boolean"}) == []
    assert room.validate(None, {"type": "null"}) == []


def test_required_and_additional_properties_are_both_reported():
    errors = room.validate({"name": "Wren", "age": 30, "wings": 2}, PETITION)
    assert any("$.role" in e and "required" in e.lower() for e in errors), (
        f"'role' is required and missing; expected an error at $.role mentioning 'required'. Got {errors}"
    )
    assert any("$.wings" in e and "additional" in e.lower() for e in errors), (
        f"'wings' is not in properties and additionalProperties is False; expected an error at $.wings. Got {errors}"
    )


def test_enum_and_bounds_are_checked_with_their_paths():
    errors = room.validate({"name": "", "age": 200, "role": "dragon"}, PETITION)
    assert any("$.role" in e for e in errors), f"'dragon' is not in the enum; expected an error at $.role. Got {errors}"
    assert any("$.age" in e and "150" in e for e in errors), f"200 is above maximum 150; expected an error at $.age quoting the bound. Got {errors}"
    assert any("$.name" in e for e in errors), f"'' is shorter than minLength 1; expected an error at $.name. Got {errors}"
    low = room.validate({"name": "Wren", "age": -1, "role": "imp"}, PETITION)
    assert any("$.age" in e for e in low), f"-1 is below minimum 0; expected an error at $.age. Got {low}"


def test_items_and_nested_objects_carry_their_full_path():
    obj = {
        "name": "Wren",
        "age": 30,
        "role": "knight",
        "gear": ["sword", 7],
        "address": {"houses": "many", "colour": "blue"},
    }
    errors = room.validate(obj, PETITION)
    assert any(e.startswith("$.gear[1]") for e in errors), f"gear[1] is an int in an array of strings; expected an error at $.gear[1]. Got {errors}"
    assert any(e.startswith("$.address.houses") for e in errors), f"address.houses should be an integer; expected an error at $.address.houses. Got {errors}"
    assert any(e.startswith("$.address.town") for e in errors), f"address.town is required; expected an error at $.address.town. Got {errors}"
    assert any(e.startswith("$.address.colour") for e in errors), f"address.colour is an additional property; expected an error at $.address.colour. Got {errors}"


def test_all_violations_are_reported_not_just_the_first():
    errors = room.validate({"name": "", "age": 200, "role": "dragon", "wings": 2}, PETITION)
    assert len(errors) >= 4, (
        f"Four clauses are broken (name, age, role, wings) but you reported {len(errors)}: {errors}. "
        "Collect every error so the model can fix them in one retry."
    )


def test_a_top_level_type_mismatch_uses_the_root_path():
    assert room.validate([], PETITION) == ["$: expected object, got list"]


# -------------------------------------------------------------- structured_call
def test_a_fenced_reply_is_accepted_on_the_first_attempt():
    llm = ScriptedLLM(["```json\n" + GOOD_JSON + "\n```"])
    obj, attempts = room.structured_call(llm, [Message.user("Your petition, in JSON.")], PETITION)
    assert obj == GOOD, f"Expected the parsed petition, got {obj!r}"
    assert attempts == 1, f"A first-time success is 1 attempt, you reported {attempts}."
    assert len(llm.calls) == 1, f"One valid reply needs one call; you made {len(llm.calls)}."


def test_an_invalid_reply_is_sent_back_with_its_errors():
    llm = ScriptedLLM([BAD_AGE, GOOD_JSON])
    obj, attempts = room.structured_call(llm, [Message.user("Your petition, in JSON.")], PETITION)
    assert obj == GOOD and attempts == 2, f"Expected success on attempt 2, got attempts={attempts}, obj={obj!r}"
    second = llm.calls[1]["messages"]
    correction = second[-1]
    assert correction.role == "user", f"The correction goes back as a user message; the last message of the retry has role {correction.role!r}."
    assert "invalid" in correction.content.lower(), f"Tell the model its reply was invalid. Retry prompt: {correction.content!r}"
    assert "$.age" in correction.content, (
        f"The retry prompt must quote the validation errors so the model knows what to fix. Got: {correction.content!r}"
    )
    assert "JSON" in correction.content, f"Ask for JSON only. Retry prompt: {correction.content!r}"
    assert second[-2].role == "assistant" and second[-2].content == BAD_AGE, (
        "Keep the model's failed reply in the transcript, as an assistant message, right before the "
        "correction. Otherwise 'your previous reply' refers to nothing the model can see."
    )


def test_prose_without_json_is_also_a_broken_contract():
    llm = ScriptedLLM(["I would rather not fill in forms today.", GOOD_JSON])
    obj, attempts = room.structured_call(llm, [Message.user("Petition, in JSON.")], PETITION)
    assert attempts == 2 and obj == GOOD
    correction = llm.calls[1]["messages"][-1].content
    assert "JSON" in correction, f"When no JSON is found, the retry prompt must still ask for JSON. Got: {correction!r}"


def test_the_weaver_gives_up_after_max_attempts():
    llm = ScriptedLLM([BAD_AGE] * 10)
    with pytest.raises(room.ContractBroken):
        room.structured_call(llm, [Message.user("Petition.")], PETITION, max_attempts=3)
    assert len(llm.calls) == 3, (
        f"max_attempts=3 means exactly three model calls before giving up; you made {len(llm.calls)}."
    )


def test_max_attempts_one_means_exactly_one_call():
    llm = ScriptedLLM([BAD_AGE, GOOD_JSON])
    with pytest.raises(room.ContractBroken):
        room.structured_call(llm, [Message.user("Petition.")], PETITION, max_attempts=1)
    assert len(llm.calls) == 1


def test_the_petitioners_scroll_is_not_scribbled_on():
    original = [Message.system("You are a clerk."), Message.user("Petition, in JSON.")]
    snapshot = list(original)
    llm = ScriptedLLM([BAD_AGE, GOOD_JSON])
    room.structured_call(llm, original, PETITION)
    assert original == snapshot, (
        f"structured_call appended to the caller's message list (now {len(original)} messages). "
        "Copy it first: a retry loop that mutates shared history leaves duplicate turns behind."
    )
