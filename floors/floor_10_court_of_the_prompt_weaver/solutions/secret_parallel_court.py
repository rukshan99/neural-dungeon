"""SECRET - THE PARALLEL COURT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dungeon.artifacts.llm import ToolCall

from .room_2_the_summoning_loop import ToolRegistry


def run_tools_parallel(registry: ToolRegistry, tool_calls: list[ToolCall], max_workers: int = 4) -> list[str]:
    """Execute every call concurrently; return the results in the ORIGINAL order.

    ``Future.result()`` blocks until that future is done, so walking the
    futures in submission order gives submission-ordered results whatever
    order the threads finish in. (``as_completed`` would not.)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(registry.call, call.name, call.arguments) for call in tool_calls]
        return [future.result() for future in futures]


class JSONStreamAssembler:
    """Collect streamed fragments of a JSON value; know when the value is whole."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def feed(self, chunk: str) -> None:
        self._parts.append(chunk)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def _balanced(self) -> bool:
        depth = 0
        in_string = False
        escaped = False
        seen_open = False
        for ch in self.text:
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
                depth += 1
                seen_open = True
            elif ch in "}]":
                depth -= 1
                if depth < 0:
                    return False
        return seen_open and depth == 0 and not in_string

    def complete(self) -> bool:
        if not self._balanced():
            return False
        try:
            json.loads(self.text)
        except ValueError:
            return False
        return True

    def result(self) -> Any:
        if not self.complete():
            raise ValueError(f"JSON is not complete yet: {self.text!r}")
        return json.loads(self.text)
