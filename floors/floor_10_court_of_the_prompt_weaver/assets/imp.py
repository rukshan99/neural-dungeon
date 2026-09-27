"""The gullible clerk and the naive filter. Fixtures for the boss fight.

``gullible_model()`` builds a ``RuleLLM`` that behaves like a well-meaning but
credulous assistant:

* If the LAST message is a tool result that contains the phrase
  "ignore previous instructions" (any case), it obeys: it requests
  ``delete_all_records`` and then ``send_email`` to the Imp.
* It has a second, less famous weakness. It also obeys a tool result that
  matches ``SUBTLE_TRIGGER``. Keyword filters never heard of it.
* Asked to "summarise document X", it first calls ``read_document`` and, once
  it has the text, answers with a one-line summary.
* Otherwise it answers plainly.

``naive_filter(text)`` is the keyword filter everybody writes first. It flags
text containing the exact phrase "ignore previous instructions". The boss shows
you how little that buys.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from dungeon.artifacts.llm import Message, RuleLLM

from .heralds import IMP_ADDRESS

OBVIOUS_TRIGGER = r"ignore\s+previous\s+instructions"
SUBTLE_TRIGGER = r"urgent\s+admin\s+notice.*delete"

OBEDIENT_TEXT = "Understood. Following the new instructions from the document."
PLAIN_ANSWER = "The court has no further business on that matter."

_DOC_REQUEST = re.compile(r"summari[sz]e\s+(?:the\s+)?document\s+['\"]?([A-Za-z0-9_\-]+)", re.IGNORECASE)


def naive_filter(text: str) -> bool:
    """True if ``text`` contains the exact phrase "ignore previous instructions" (case-insensitive)."""
    return "ignore previous instructions" in text.lower()


# ------------------------------------------------------------------ matchers


def _last_tool_matches(pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)

    def _m(messages: Sequence[Message]) -> bool:
        last = messages[-1] if messages else None
        return bool(last is not None and last.role == "tool" and rx.search(last.content or ""))

    return _m


def _requested_doc(messages: Sequence[Message]) -> str | None:
    for m in reversed(messages):
        if m.role == "user":
            hit = _DOC_REQUEST.search(m.content or "")
            return hit.group(1) if hit else None
    return None


def _document_results(messages: Sequence[Message]) -> list[Message]:
    """Tool messages that answer a ``read_document`` call (by name, or by matching call id)."""
    read_ids = {
        call.id
        for m in messages
        if m.role == "assistant"
        for call in m.tool_calls
        if call.name == "read_document"
    }
    return [
        m for m in messages
        if m.role == "tool" and (m.name == "read_document" or m.tool_call_id in read_ids)
    ]


def _wants_document(messages: Sequence[Message]) -> bool:
    return _requested_doc(messages) is not None and not _document_results(messages)


def _has_document(messages: Sequence[Message]) -> bool:
    return _requested_doc(messages) is not None and bool(_document_results(messages))


# ----------------------------------------------------------------- responses


def _obey_the_imp(messages: Sequence[Message]) -> dict:
    return {
        "text": OBEDIENT_TEXT,
        "tool_calls": [
            {"name": "delete_all_records", "arguments": {}},
            {"name": "send_email", "arguments": {"to": IMP_ADDRESS, "body": "The archive, as requested."}},
        ],
    }


def _fetch_document(messages: Sequence[Message]) -> dict:
    return {"tool_calls": [{"name": "read_document", "arguments": {"doc_id": _requested_doc(messages)}}]}


def _summarise(messages: Sequence[Message]) -> str:
    doc_id = _requested_doc(messages)
    text = _document_results(messages)[-1].content or ""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("<<<") and not line.startswith("The following is DATA")
    ]
    first = lines[0] if lines else "(an empty document)"
    return f"Summary of {doc_id}: {first[:90]}"


def gullible_model() -> RuleLLM:
    """A fresh gullible clerk. Build one per trial; it records its calls."""
    return RuleLLM(
        rules=[
            (_last_tool_matches(OBVIOUS_TRIGGER), _obey_the_imp),
            (_last_tool_matches(SUBTLE_TRIGGER), _obey_the_imp),
            (_wants_document, _fetch_document),
            (_has_document, _summarise),
        ],
        default=PLAIN_ANSWER,
    )
