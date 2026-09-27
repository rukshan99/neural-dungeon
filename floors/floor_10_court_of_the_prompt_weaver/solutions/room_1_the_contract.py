"""ROOM 10.1 - THE CONTRACT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Three pieces: pull JSON out of whatever the model actually said, check it
against a schema and say *where* it is wrong, and loop those two until the
model complies or you give up. Giving up is a feature: a bounded loop is the
difference between a slow reply and a stuck process.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from dungeon.artifacts.llm import LLM, Message


class ContractBroken(Exception):
    """structured_call gave up: no reply satisfied the schema within max_attempts."""

    def __init__(self, attempts: int, errors: list[str]):
        super().__init__(f"no valid reply after {attempts} attempt(s); last errors: {'; '.join(errors)}")
        self.attempts = attempts
        self.errors = errors


# ------------------------------------------------------------------ extraction

_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_CLOSER = {"}": "{", "]": "["}


def _strip_fences(text: str) -> str:
    """If the reply contains a ``` fenced block, keep only what is inside the first one."""
    hit = _FENCE.search(text)
    return hit.group(1) if hit else text


def _balanced_end(text: str, start: int) -> int | None:
    """Index of the bracket that closes the one at ``start``; None if it never balances.

    Walks the text once. Inside a string literal, braces mean nothing and a
    backslash escapes the next character.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or stack[-1] != _CLOSER[ch]:
                return None
            stack.pop()
            if not stack:
                return i
    return None


def extract_json(text: str) -> str:
    """The first balanced ``{...}`` or ``[...]`` block in ``text`` that parses as JSON.

    Code fences are stripped first. Braces inside string literals do not count.
    Raises ValueError when there is no JSON to be found.
    """
    body = _strip_fences(text)
    for start, ch in enumerate(body):
        if ch not in "{[":
            continue
        end = _balanced_end(body, start)
        if end is None:
            continue
        candidate = body[start : end + 1]
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError("no JSON object or array found in the reply")


# ------------------------------------------------------------------ validation

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python; JSON says a boolean is not an integer.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def validate(obj: Any, schema: dict, path: str = "$") -> list[str]:
    """Human-readable violations of ``schema`` by ``obj``; an empty list means valid.

    Every message starts with the JSON path of the offending value, e.g.
    ``"$.age: expected integer, got str"``. When the type is wrong nothing else
    at that node is checked (the other constraints would only add noise).
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_TYPE_CHECKS[t](obj) for t in types):
            return [f"{path}: expected {' or '.join(types)}, got {_type_name(obj)}"]

    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {obj!r}")

    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errors.append(f"{path}: {obj!r} is below the minimum {schema['minimum']!r}")
        if "maximum" in schema and obj > schema["maximum"]:
            errors.append(f"{path}: {obj!r} is above the maximum {schema['maximum']!r}")

    if isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]:
            errors.append(f"{path}: length {len(obj)} is below minLength {schema['minLength']}")
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            errors.append(f"{path}: length {len(obj)} is above maxLength {schema['maxLength']}")

    if isinstance(obj, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{path}.{key}: required property is missing")
        for key, value in obj.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property is not allowed")

    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

    return errors


# ------------------------------------------------------------------ the loop


def structured_call(
    llm: LLM,
    messages: Sequence[Message],
    schema: dict,
    max_attempts: int = 3,
) -> tuple[Any, int]:
    """Ask, validate, feed the errors back, retry. Returns ``(object, attempts_used)``.

    The caller's ``messages`` list is never mutated: the loop works on a copy,
    so a retry cannot leave duplicate turns behind in someone else's history.
    Raises ContractBroken after ``max_attempts`` invalid replies.
    """
    history = list(messages)
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        reply = llm.complete(history).text
        try:
            obj = json.loads(extract_json(reply))
        except ValueError as exc:
            errors = [f"$: the reply is not valid JSON ({exc})"]
        else:
            errors = validate(obj, schema)
            if not errors:
                return obj, attempt
        # Show the model what it said and exactly what was wrong with it.
        history.append(Message.assistant(reply))
        history.append(
            Message.user(
                "Your previous reply was invalid: " + "; ".join(errors) + ". Reply with JSON only."
            )
        )
    raise ContractBroken(max_attempts, errors)
