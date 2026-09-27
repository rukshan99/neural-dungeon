"""ROOM 10.1 - THE CONTRACT

    Petitioners queue before the Weaver's clerks. Each petition must be woven
    into a contract of fixed form before the court will hear it. The
    petitioners, being petitioners, reply in prose, in fences, and in JSON
    with the age written as a word.

A language model returns text. Your program needs data. Between the two sits a
contract (a JSON schema) and three jobs:

1. ``extract_json``: find the JSON inside whatever the model actually said.
   Models wrap answers in ``` fences, add "Certainly! Here is the JSON:" and
   sign off with "Let me know if you need anything else". Braces inside string
   values ("we } never { yield") must not confuse you.
2. ``validate``: check the parsed object against the schema and produce error
   messages a model (or a human) can act on: *where* ("$.address.town") and
   *what* ("required property is missing").
3. ``structured_call``: the repair loop. Ask, validate, send the errors back,
   retry, and give up after ``max_attempts``. A loop with no upper bound is a
   process that never returns.

Real providers offer schema-constrained decoding and tool-use-as-schema, which
you should use when available. Validate anyway: constraints cover types, not
your business rules, and the fallback path exists whether you planned it or not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dungeon.artifacts.llm import LLM, Message


class ContractBroken(Exception):
    """structured_call gave up: no reply satisfied the schema within max_attempts."""

    def __init__(self, attempts: int, errors: list[str]):
        super().__init__(f"no valid reply after {attempts} attempt(s); last errors: {'; '.join(errors)}")
        self.attempts = attempts
        self.errors = errors


def extract_json(text: str) -> str:
    """Return the first balanced ``{...}`` or ``[...]`` block in ``text`` that parses as JSON.

    Steps: if the text contains a ``` fenced block (with or without a language
    tag), keep only the inside of the first one. Then scan for an opening
    brace or bracket and walk forward until it balances. Inside a string
    literal, braces do not count and a backslash escapes the next character.
    If a balanced block does not parse as JSON (prose like ``{flags}``), keep
    looking. Raise ``ValueError`` when nothing is found.

    Return the JSON *text*; the caller parses it.
    """
    raise NotImplementedError("extract_json() is unwritten")


def validate(obj: Any, schema: dict, path: str = "$") -> list[str]:
    """Human-readable violations of ``schema`` by ``obj``. An empty list means valid.

    Support this subset of JSON Schema:

    * ``type``: "object", "array", "string", "integer", "number", "boolean",
      "null". An integer is an ``int`` that is not a ``bool`` (Python's
      ``True`` is an int; JSON's is not). ``3.0`` is a number, not an integer.
      When the type is wrong, report it and check nothing else at that node.
    * ``required``, ``properties`` (recurse), ``additionalProperties: False``.
    * ``enum``, ``minimum``/``maximum``, ``minLength``/``maxLength``.
    * ``items`` for arrays (recurse into ``path[i]``).

    Messages start with the path: ``"$.age: expected integer, got str"``,
    ``"$.address.town: required property is missing"``,
    ``"$.gear[1]: expected string, got int"``. Report every violation you
    find, not just the first: the model fixes them all in one retry.
    """
    raise NotImplementedError("validate() is unwritten")


def structured_call(
    llm: LLM,
    messages: Sequence[Message],
    schema: dict,
    max_attempts: int = 3,
) -> tuple[Any, int]:
    """Ask the model for JSON matching ``schema``; repair with feedback; return ``(obj, attempts)``.

    Each attempt: ``llm.complete(history).text`` -> ``extract_json`` ->
    ``json.loads`` -> ``validate``. On success return the parsed object and the
    number of attempts used (1 for a first-time success).

    On failure, append the model's reply as an assistant message, then a user
    message of the form
        "Your previous reply was invalid: <errors joined by '; '>. Reply with JSON only."
    and try again. A reply with no JSON at all counts as invalid too.

    Work on a *copy* of ``messages``: the caller's list must not change.
    After ``max_attempts`` failures raise ``ContractBroken(attempts, errors)``.
    """
    raise NotImplementedError("structured_call() is unwritten")
