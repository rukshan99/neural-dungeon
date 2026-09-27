"""SECRET - THE PARALLEL COURT   (optional)

    A side door off the Imp's antechamber. In this court the heralds all
    leave at once, and the scribe writes down the answers as they trickle
    in, one brace at a time.

Two production niceties:

* When a model requests several independent tools in one turn, run them
  concurrently. But the transcript must list the results in the order the
  calls were made, whatever order the threads finish in.
* Streaming APIs deliver tool-call arguments as fragments of JSON text. You
  cannot parse a fragment; you can tell when the fragments add up to a whole
  value. Braces inside strings still do not count.
"""

from __future__ import annotations

from typing import Any

from dungeon.artifacts.llm import ToolCall

from .room_2_the_summoning_loop import ToolRegistry


def run_tools_parallel(registry: ToolRegistry, tool_calls: list[ToolCall], max_workers: int = 4) -> list[str]:
    """Execute every call with a ``concurrent.futures.ThreadPoolExecutor``; return the results in ORIGINAL order.

    ``results[i]`` is the string ``registry.call`` returned for ``tool_calls[i]``.
    ``registry.call`` never raises, so a failing tool simply yields its error string.
    """
    raise NotImplementedError("run_tools_parallel() is unwritten")


class JSONStreamAssembler:
    """Collect streamed fragments of one JSON value and know when the value is whole."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def feed(self, chunk: str) -> None:
        self._parts.append(chunk)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def complete(self) -> bool:
        """True only when the text so far is balanced (braces outside strings, escapes respected) AND parses."""
        raise NotImplementedError("JSONStreamAssembler.complete() is unwritten")

    def result(self) -> Any:
        """The parsed value. Raise ValueError if ``complete()`` is False."""
        raise NotImplementedError("JSONStreamAssembler.result() is unwritten")
