"""ROOM 10.4 - THE LEDGER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A context window is a budget. ``fit_messages`` spends it on the newest turns
while keeping the transcript *valid*: system messages stay, the latest user
message stays, and a tool result never survives without the assistant message
that asked for it (nor the other way round).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dungeon.artifacts.llm import LLM, Completion, Message, Usage, approx_messages_tokens

SUMMARY_PREFIX = "Summary of earlier conversation: "


@dataclass
class ContextBudget:
    """How many tokens the prompt may use: ``max_tokens`` minus what the reply needs."""

    max_tokens: int
    reserve_for_output: int = 0
    count_fn: Callable[[Sequence[Message]], int] = approx_messages_tokens

    @property
    def available(self) -> int:
        return self.max_tokens - self.reserve_for_output

    def count(self, messages: Sequence[Message]) -> int:
        return self.count_fn(list(messages))

    def fits(self, messages: Sequence[Message]) -> bool:
        return self.count(messages) <= self.available


def _units(messages: Sequence[Message]) -> list[int]:
    """unit id per message: an assistant message with tool_calls and its tool results share one."""
    unit_of = list(range(len(messages)))
    owner_of_call: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            for call in m.tool_calls:
                owner_of_call[call.id] = i
        elif m.role == "tool" and m.tool_call_id in owner_of_call:
            unit_of[i] = owner_of_call[m.tool_call_id]
    return unit_of


def fit_messages(messages: Sequence[Message], budget: ContextBudget) -> list[Message]:
    """The same Message objects, in order, minus the oldest droppable units until it fits.

    Never dropped: system messages, the latest user message, and half of a
    tool exchange. If the protected messages alone exceed the budget they are
    returned anyway; the caller can check ``budget.fits`` and decide.
    """
    messages = list(messages)
    unit_of = _units(messages)

    protected = {unit_of[i] for i, m in enumerate(messages) if m.role == "system"}
    user_positions = [i for i, m in enumerate(messages) if m.role == "user"]
    if user_positions:
        protected.add(unit_of[user_positions[-1]])

    kept = set(range(len(messages)))
    droppable = sorted({u for u in unit_of if u not in protected})  # oldest first

    def current() -> list[Message]:
        return [messages[i] for i in sorted(kept)]

    while droppable and budget.count(current()) > budget.available:
        unit = droppable.pop(0)
        kept -= {i for i, u in enumerate(unit_of) if u == unit}
    return current()


@dataclass
class CostMeter:
    """Adds up Usage across calls and prices it per million tokens."""

    price_in_per_million: float
    price_out_per_million: float
    usage: Usage = field(default_factory=Usage)
    calls: int = 0

    def record(self, usage: Usage | Completion) -> None:
        if isinstance(usage, Completion):
            usage = usage.usage
        self.usage = self.usage + usage
        self.calls += 1

    @property
    def cost(self) -> float:
        return (
            self.usage.input_tokens * self.price_in_per_million
            + self.usage.output_tokens * self.price_out_per_million
        ) / 1_000_000


def render_transcript(messages: Sequence[Message]) -> str:
    """Plain text rendering of a conversation, for handing to a summariser."""
    lines = []
    for m in messages:
        line = f"{m.role}: {m.content}".rstrip()
        if m.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
            line += f" [called {calls}]"
        lines.append(line)
    return "\n".join(lines)


def summarize_history(llm: LLM, old_messages: Sequence[Message]) -> Message:
    """Ask the model for a summary of ``old_messages``; return it as a system note."""
    prompt = [
        Message.system("You write terse, faithful summaries of conversations. Keep every fact and decision."),
        Message.user("Summarise the following conversation:\n\n" + render_transcript(old_messages)),
    ]
    completion = llm.complete(prompt)
    return Message.system(SUMMARY_PREFIX + completion.text.strip())


def compact(messages: Sequence[Message], budget: ContextBudget, llm: LLM) -> list[Message]:
    """Fit the history to the budget; summarise what was dropped and put the summary first.

    The summary goes right after the leading system messages so the original
    system prompt stays where providers expect it. Then the result is fitted
    once more, because the summary costs tokens too.
    """
    messages = list(messages)
    kept = fit_messages(messages, budget)
    if len(kept) == len(messages):
        return kept
    kept_ids = {id(m) for m in kept}
    dropped = [m for m in messages if id(m) not in kept_ids]
    summary = summarize_history(llm, dropped)

    leading: list[Message] = []
    rest: list[Any] = list(kept)
    while rest and rest[0].role == "system":
        leading.append(rest.pop(0))
    return fit_messages([*leading, summary, *rest], budget)
