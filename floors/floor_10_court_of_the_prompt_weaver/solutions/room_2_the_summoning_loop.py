"""ROOM 10.2 - THE SUMMONING LOOP  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The loop is a small state machine:

    CALL the model
      stop_reason == "tool_use"  -> run EVERY requested call, append one tool
                                    message per call (matching ids), CALL again
      anything else              -> return the text
    more than max_steps CALLs    -> raise StepBudgetExceeded

Tool failures are *data* for the model, never exceptions for the loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dungeon.artifacts.llm import LLM, Message, ToolSpec

TOOL_ERROR_PREFIX = "ToolError:"


class StepBudgetExceeded(Exception):
    """The loop called the model max_steps times and it still wanted more tools."""


@dataclass
class Tool:
    """A callable herald: what it is called, what it does, its JSON-schema parameters, and the code."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    """Holds tools by name; turns every failure into a string the model can read."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict | None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return f"{TOOL_ERROR_PREFIX} unknown tool {name!r}. Available tools: {available}"
        try:
            result = tool.fn(**(arguments or {}))
        except Exception as exc:  # broad on purpose: the model must hear about any failure
            return f"{TOOL_ERROR_PREFIX} {name} failed with {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)


@dataclass
class AgentResult:
    text: str
    messages: list[Message] = field(default_factory=list)
    steps: int = 0
    tool_calls_made: int = 0


def run_tool_loop(
    llm: LLM,
    registry: ToolRegistry,
    messages: Sequence[Message],
    max_steps: int = 8,
) -> AgentResult:
    """Drive the model until it stops asking for tools. One 'step' is one model call."""
    history = list(messages)
    tool_calls_made = 0
    for step in range(1, max_steps + 1):
        completion = llm.complete(history, registry.specs())
        history.append(completion.message)
        if completion.stop_reason != "tool_use" or not completion.tool_calls:
            return AgentResult(completion.text, history, step, tool_calls_made)
        for call in completion.tool_calls:
            result = registry.call(call.name, call.arguments)
            history.append(Message.tool(call.id, result, name=call.name))
            tool_calls_made += 1
    raise StepBudgetExceeded(
        f"the model still wanted tools after {max_steps} steps ({tool_calls_made} tool calls made)"
    )
