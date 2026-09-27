"""A tiny, provider-neutral LLM interface plus deterministic mock models.

Floors 9 (retrieval), 10 (agents) and 11 (evaluation) are about the engineering
*around* a language model: grounding, tool loops, retries, budgets, judges. None
of that needs a real model to learn, and a real model would make trials slow,
expensive and non-deterministic. So every trial on those floors talks to one of
the mocks below through the same ``LLM`` protocol a real adapter implements
(see ``providers.py`` for an Anthropic adapter you can use for your own
experiments).

The message and tool shapes are deliberately close to what real chat APIs use:

    Message(role="system"|"user"|"assistant"|"tool", content=str,
            tool_calls=[ToolCall(id, name, arguments)], tool_call_id=...)
    ToolSpec(name, description, parameters=<JSON schema dict>)
    Completion(message, stop_reason="end_turn"|"tool_use"|"max_tokens", usage)

Mocks:

- ``ScriptedLLM(responses)``: returns canned completions in order and records
  every call. Perfect for unit-testing a loop.
- ``RuleLLM(rules, default)``: pattern -> response. Use it to build a
  "gullible" model that follows instructions hidden in tool results, or a
  "librarian" that answers from context only when the context contains a
  keyword.
- ``FlakyLLM(inner, schedule)``: raises the errors you schedule (rate limits,
  timeouts, server errors) before delegating. For retry/backoff trials.

Token counting here is an approximation (about 4 characters per token). It is
labelled as such everywhere; real systems must count with the provider's
tokenizer or ``count_tokens`` endpoint.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ------------------------------------------------------------------- messages


@dataclass
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON-schema object."""

    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    name: str | None = None  # tool name on role="tool"

    # convenience constructors
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Sequence[ToolCall] | None = None) -> "Message":
        return cls("assistant", content, list(tool_calls or []))

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id, name=name)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class Completion:
    message: Message
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens" | "refusal"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls


# --------------------------------------------------------------------- errors


class LLMError(Exception):
    """Base class for anything a model call can raise."""


class RateLimitError(LLMError):
    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(LLMError):
    """A 5xx-style transient failure. Retryable."""


class LLMTimeoutError(LLMError):
    """The call took too long. Usually retryable."""


class LLMConnectionError(LLMError):
    """Network trouble. Retryable."""


class InvalidRequestError(LLMError):
    """A 4xx-style failure that will not go away if you retry."""


class ScriptExhausted(LLMError):
    """A ScriptedLLM was asked for more responses than it was given."""


RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (RateLimitError, ServerError, LLMTimeoutError, LLMConnectionError)


# ------------------------------------------------------------------- protocol


@runtime_checkable
class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        max_tokens: int = 1024,
    ) -> Completion: ...


# --------------------------------------------------------------------- tokens


def approx_tokens(text: str) -> int:
    """Roughly 4 characters per token. An approximation, labelled as such."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def approx_message_tokens(message: Message) -> int:
    n = approx_tokens(message.content) + 4  # role/framing overhead
    for call in message.tool_calls:
        n += approx_tokens(call.name) + approx_tokens(repr(call.arguments))
    return n


def approx_messages_tokens(messages: Iterable[Message]) -> int:
    return sum(approx_message_tokens(m) for m in messages)


# ------------------------------------------------------------------- building


_counter = {"n": 0}


def next_call_id(prefix: str = "call") -> str:
    _counter["n"] += 1
    return f"{prefix}_{_counter['n']:04d}"


def make_completion(
    text: str = "",
    tool_calls: Sequence[ToolCall | dict] | None = None,
    stop_reason: str | None = None,
    prompt_tokens: int = 0,
) -> Completion:
    """Build a Completion. Dict tool calls need ``name`` and optional ``arguments``/``id``."""
    calls: list[ToolCall] = []
    for tc in tool_calls or []:
        if isinstance(tc, ToolCall):
            calls.append(tc)
        else:
            calls.append(ToolCall(id=tc.get("id") or next_call_id(), name=tc["name"], arguments=dict(tc.get("arguments", {}))))
    if stop_reason is None:
        stop_reason = "tool_use" if calls else "end_turn"
    message = Message.assistant(text, calls)
    return Completion(message=message, stop_reason=stop_reason, usage=Usage(prompt_tokens, approx_message_tokens(message)))


def coerce_response(item: Any, messages: Sequence[Message]) -> Completion:
    """Turn a scripted item into a Completion (or raise it, if it is an exception)."""
    if isinstance(item, BaseException):
        raise item
    if isinstance(item, Completion):
        return item
    if callable(item):
        return coerce_response(item(messages), messages)
    prompt = approx_messages_tokens(messages)
    if isinstance(item, str):
        return make_completion(item, prompt_tokens=prompt)
    if isinstance(item, dict):
        return make_completion(
            text=item.get("text", ""),
            tool_calls=item.get("tool_calls"),
            stop_reason=item.get("stop_reason"),
            prompt_tokens=prompt,
        )
    raise TypeError(f"Cannot turn {type(item).__name__} into a Completion")


# ---------------------------------------------------------------------- mocks


class ScriptedLLM:
    """Returns the scripted responses in order; records every call in ``.calls``.

    Each script item may be: a str (assistant text), a dict with ``text`` and/or
    ``tool_calls`` (list of {"name", "arguments"}), a Completion, an Exception
    instance (raised when reached), or a callable(messages) returning any of those.
    """

    def __init__(self, responses: Iterable[Any]):
        self._script = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(self, messages, tools=None, *, max_tokens: int = 1024) -> Completion:
        self.calls.append({"messages": list(messages), "tools": list(tools or []), "max_tokens": max_tokens})
        if not self._script:
            raise ScriptExhausted(
                f"ScriptedLLM has no responses left after {len(self.calls) - 1} call(s). "
                "Is your loop failing to terminate?"
            )
        item = self._script.pop(0)
        return coerce_response(item, messages)


Matcher = Callable[[Sequence[Message]], bool]


def last_message(messages: Sequence[Message], role: str | None = None) -> Message | None:
    for m in reversed(messages):
        if role is None or m.role == role:
            return m
    return None


def matches(pattern: str, roles: Sequence[str] = ("user", "tool"), flags: int = re.IGNORECASE) -> Matcher:
    """A matcher that is true if ``pattern`` appears in ANY message with one of ``roles``."""
    rx = re.compile(pattern, flags)

    def _m(messages: Sequence[Message]) -> bool:
        return any(m.role in roles and rx.search(m.content or "") for m in messages)

    return _m


def last_matches(pattern: str, flags: int = re.IGNORECASE) -> Matcher:
    """A matcher that is true if ``pattern`` appears in the LAST message."""
    rx = re.compile(pattern, flags)

    def _m(messages: Sequence[Message]) -> bool:
        m = last_message(messages)
        return bool(m and rx.search(m.content or ""))

    return _m


class RuleLLM:
    """First matching rule wins; otherwise ``default``. Deterministic and inspectable.

    rules: sequence of (matcher, response). ``matcher`` is a callable(messages) -> bool
    or a regex string (matched against every user/tool message). ``response`` is
    anything ``coerce_response`` accepts (str, dict, Completion, exception, callable).
    """

    def __init__(self, rules: Sequence[tuple[Matcher | str, Any]], default: Any = "I do not know."):
        self.rules = [((matches(m) if isinstance(m, str) else m), r) for m, r in rules]
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools=None, *, max_tokens: int = 1024) -> Completion:
        self.calls.append({"messages": list(messages), "tools": list(tools or []), "max_tokens": max_tokens})
        for matcher, response in self.rules:
            if matcher(messages):
                return coerce_response(response, messages)
        return coerce_response(self.default, messages)


class FlakyLLM:
    """Wraps another LLM and raises scheduled errors first.

    ``schedule`` is consumed one entry per call: an exception instance is raised,
    ``None`` means "succeed" (delegate). When the schedule is exhausted, every
    call succeeds. ``attempts`` counts every call including the failed ones.
    """

    def __init__(self, inner: LLM, schedule: Iterable[BaseException | None]):
        self.inner = inner
        self._schedule = list(schedule)
        self.attempts = 0

    def complete(self, messages, tools=None, *, max_tokens: int = 1024) -> Completion:
        self.attempts += 1
        if self._schedule:
            item = self._schedule.pop(0)
            if item is not None:
                raise item
        return self.inner.complete(messages, tools, max_tokens=max_tokens)


class TruncatingLLM:
    """Wraps another LLM and cuts the text to ``max_tokens`` (approx), reporting stop_reason="max_tokens"."""

    def __init__(self, inner: LLM):
        self.inner = inner

    def complete(self, messages, tools=None, *, max_tokens: int = 1024) -> Completion:
        completion = self.inner.complete(messages, tools, max_tokens=max_tokens)
        if approx_tokens(completion.text) > max_tokens:
            cut = completion.text[: max_tokens * 4]
            completion = Completion(Message.assistant(cut, completion.tool_calls), "max_tokens", completion.usage)
        return completion
