"""ROOM 10.4 - THE LEDGER

    Every word spoken in the court is written in the Ledger, and every page
    costs a copper. The Ledger has a fixed number of pages. When it fills, the
    oldest pages are torn out - carefully, so no herald's report survives
    without the order that summoned it.

A model reads a bounded window of tokens. Everything you send (system prompt,
history, tool results) plus everything it will write must fit. This room is
the bookkeeping:

* ``ContextBudget``: the window minus what the reply needs. Counting here uses
  ``approx_messages_tokens`` (about 4 characters per token). Real systems count
  with the provider's tokenizer; the *logic* is identical.
* ``fit_messages``: spend the budget on the newest turns. Keep every system
  message. Never drop the latest user message (it is the question). Drop the
  oldest turns first. An assistant message that requested tools and the tool
  messages that answered it are ONE unit: a tool result without its request
  (or the reverse) is an invalid transcript that providers reject.
* ``CostMeter``: input and output tokens are priced differently, per million.
  Add up Usage across calls; compute the bill.
* ``summarize_history`` / ``compact``: instead of forgetting the dropped turns,
  ask the model for a summary and carry that forward as a system note. Lossy,
  and it costs a call, so do it when you must, not on every turn.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from dungeon.artifacts.llm import LLM, Completion, Message, Usage, approx_messages_tokens

SUMMARY_PREFIX = "Summary of earlier conversation: "


@dataclass
class ContextBudget:
    """How many tokens the prompt may use: ``max_tokens`` minus ``reserve_for_output``."""

    max_tokens: int
    reserve_for_output: int = 0
    count_fn: Callable[[Sequence[Message]], int] = approx_messages_tokens

    def count(self, messages: Sequence[Message]) -> int:
        return self.count_fn(list(messages))

    @property
    def available(self) -> int:
        """Tokens the prompt may occupy."""
        raise NotImplementedError("ContextBudget.available is unwritten")

    def fits(self, messages: Sequence[Message]) -> bool:
        """True when ``count(messages) <= available``."""
        raise NotImplementedError("ContextBudget.fits() is unwritten")


def fit_messages(messages: Sequence[Message], budget: ContextBudget) -> list[Message]:
    """The same Message objects, in their original order, trimmed to fit ``budget``.

    Rules, in priority order:
      1. every system message is kept;
      2. the latest user message is kept;
      3. an assistant message with tool_calls and the tool messages whose
         ``tool_call_id`` it issued are dropped or kept together;
      4. drop the oldest droppable unit first, one at a time, until it fits.
    If the protected messages alone exceed the budget, return them anyway.
    Do not copy or modify messages; return the originals.
    """
    raise NotImplementedError("fit_messages() is unwritten")


@dataclass
class CostMeter:
    """Adds up Usage across calls and prices it per million tokens."""

    price_in_per_million: float
    price_out_per_million: float
    usage: Usage = field(default_factory=Usage)
    calls: int = 0

    def record(self, usage: Usage | Completion) -> None:
        """Add one call's usage (a Completion is accepted for convenience) and count the call."""
        raise NotImplementedError("CostMeter.record() is unwritten")

    @property
    def cost(self) -> float:
        """input_tokens * price_in / 1e6 + output_tokens * price_out / 1e6."""
        raise NotImplementedError("CostMeter.cost is unwritten")


def render_transcript(messages: Sequence[Message]) -> str:
    """Plain-text rendering of a conversation, for handing to a summariser. (Provided.)"""
    lines = []
    for m in messages:
        line = f"{m.role}: {m.content}".rstrip()
        if m.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
            line += f" [called {calls}]"
        lines.append(line)
    return "\n".join(lines)


def summarize_history(llm: LLM, old_messages: Sequence[Message]) -> Message:
    """Ask ``llm`` to summarise ``old_messages`` (use ``render_transcript``).

    Return ``Message.system(SUMMARY_PREFIX + summary_text)``. The prompt you
    send must contain the transcript: a summariser that is not shown the
    conversation summarises nothing.
    """
    raise NotImplementedError("summarize_history() is unwritten")


def compact(messages: Sequence[Message], budget: ContextBudget, llm: LLM) -> list[Message]:
    """``fit_messages`` plus memory: summarise what was dropped and keep the summary.

    If nothing needs dropping, return the messages unchanged and do not call the
    model. Otherwise summarise the dropped messages, insert the summary note
    right after the leading system messages (so the original system prompt
    stays first), then fit once more so the summary's own tokens are paid for.
    The kept recent turns stay verbatim.
    """
    raise NotImplementedError("compact() is unwritten")
