"""ROOM 10.2 - THE SUMMONING LOOP

    The court keeps heralds: one fetches documents, one counts coins, one
    knows the way to the kitchens. The Weaver does not run errands. It names
    a herald and waits. When the herald returns, the Weaver reads the report
    and either names another herald or gives its answer.

An "agent" is a model plus tools plus this loop. The loop is a small state
machine and every production incident with agents is a bug in it:

    CALL the model with the history and the tool specs
      stop_reason "tool_use"  -> execute EVERY requested call, in order; append
                                 one tool Message per call with the matching
                                 tool_call_id and name; CALL again
      anything else           -> return the text (done)
    more than max_steps CALLs -> raise StepBudgetExceeded

Two rules keep it sane. First, the transcript must stay *valid*: an assistant
message that requests N tools is followed by exactly N tool messages, one per
id. Providers reject anything else. Second, a tool failure is *data for the
model*, never an exception for the loop. Unknown tool, wrong arguments, a
crash inside the tool: all become a string that starts with ``TOOL_ERROR_PREFIX``
and goes back to the model, which can apologise, retry or ask for help. The
loop cannot do any of those.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dungeon.artifacts.llm import LLM, Message, ToolSpec

TOOL_ERROR_PREFIX = "ToolError:"


class StepBudgetExceeded(Exception):
    """The loop called the model max_steps times and it still wanted more tools."""


@dataclass
class Tool:
    """A callable herald: name, description, JSON-schema ``parameters``, and the Python ``fn``."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> ToolSpec:
        """The ``ToolSpec`` a model sees: name, description, parameters. Nothing about ``fn``."""
        raise NotImplementedError("Tool.spec() is unwritten")


class ToolRegistry:
    """Holds tools by name and executes them by name, turning every failure into a string."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def register(self, tool: Tool) -> None:
        """Add (or replace) a tool under ``tool.name``."""
        raise NotImplementedError("ToolRegistry.register() is unwritten")

    def specs(self) -> list[ToolSpec]:
        """One ToolSpec per registered tool, in registration order."""
        raise NotImplementedError("ToolRegistry.specs() is unwritten")

    def call(self, name: str, arguments: dict | None) -> str:
        """Run ``name`` with ``**arguments`` and return the result as a string.

        * unknown name          -> f"{TOOL_ERROR_PREFIX} unknown tool ..." (name the available ones)
        * fn raises anything    -> f"{TOOL_ERROR_PREFIX} <name> failed with <ExcType>: <message>"
          (a TypeError from wrong argument names is just another failure)
        * fn returns a str      -> as is
        * fn returns other data -> ``json.dumps`` it
        Never raise. ``arguments`` may be None.
        """
        raise NotImplementedError("ToolRegistry.call() is unwritten")


@dataclass
class AgentResult:
    text: str
    messages: list[Message] = field(default_factory=list)
    steps: int = 0  # how many times the model was called
    tool_calls_made: int = 0  # how many tool calls were executed (including failed ones)


def run_tool_loop(
    llm: LLM,
    registry: ToolRegistry,
    messages: Sequence[Message],
    max_steps: int = 8,
) -> AgentResult:
    """Drive the model until it stops asking for tools.

    Work on a copy of ``messages``. Each step: ``llm.complete(history, registry.specs())``,
    append ``completion.message`` (it carries the tool_calls). If
    ``stop_reason == "tool_use"`` and there are tool calls, run each with
    ``registry.call`` and append ``Message.tool(call.id, result, name=call.name)``
    in the same order; then loop. Otherwise return ``AgentResult(text, history,
    steps, tool_calls_made)``. Raise ``StepBudgetExceeded`` when the loop would
    make call number ``max_steps + 1``.
    """
    raise NotImplementedError("run_tool_loop() is unwritten")
