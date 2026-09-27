"""SECRET - THE PARALLEL COURT

Heralds leave together and return in whatever order they please; the scribe
lists them in the order they were sent. Meanwhile JSON arrives one brace at a time.
"""

import threading

import pytest

from dungeon.artifacts.llm import ToolCall
from dungeon.trials import load_room

cell = load_room(__file__, "secret_parallel_court")
loop = load_room(__file__, "room_2_the_summoning_loop")

pytestmark = pytest.mark.secret

NO_PARAMS = {"type": "object", "properties": {}}


def chained_heralds(n):
    """Herald i cannot finish until herald i+1 has: they can only all complete if they run at once.

    Completion order is therefore n-1, ..., 0: the reverse of the call order.
    A herald that waits in vain returns a ToolError string instead of hanging forever.
    """
    done = [threading.Event() for _ in range(n)]
    finished = []
    lock = threading.Lock()

    def make(i):
        def herald():
            if i + 1 < n and not done[i + 1].wait(timeout=5.0):
                raise RuntimeError(f"herald {i} waited in vain for herald {i + 1}: the heralds are not running concurrently")
            with lock:
                finished.append(i)
            done[i].set()
            return str(i)

        return herald

    registry = loop.ToolRegistry()
    for i in range(n):
        registry.register(loop.Tool(f"herald_{i}", f"Herald number {i}.", NO_PARAMS, make(i)))
    calls = [ToolCall(f"c{i}", f"herald_{i}", {}) for i in range(n)]
    return registry, calls, finished


def test_heralds_leave_together_and_are_listed_in_the_order_they_were_sent():
    registry, calls, finished = chained_heralds(4)
    results = cell.run_tools_parallel(registry, calls, max_workers=4)
    assert results == ["0", "1", "2", "3"], (
        f"Results must be in the ORIGINAL call order, whatever order the threads finish in. Got {results}. "
        "Submit every call, then read each future's result() in submission order (not as_completed)."
    )
    assert finished == [3, 2, 1, 0], f"Heralds finished in order {finished}; running truly in parallel they finish 3, 2, 1, 0."


def test_a_failing_herald_does_not_spoil_the_others():
    registry = loop.ToolRegistry()
    registry.register(loop.Tool("fine", "Works.", NO_PARAMS, lambda: "fine"))
    registry.register(loop.Tool("cursed", "Explodes.", NO_PARAMS, lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    calls = [ToolCall("a", "fine", {}), ToolCall("b", "cursed", {}), ToolCall("c", "fine", {}), ToolCall("d", "nobody", {})]
    results = cell.run_tools_parallel(registry, calls, max_workers=2)
    assert len(results) == 4 and results[0] == "fine" and results[2] == "fine"
    assert results[1].startswith(loop.TOOL_ERROR_PREFIX) and "boom" in results[1], f"The failing herald's slot holds its ToolError: {results[1]!r}"
    assert results[3].startswith(loop.TOOL_ERROR_PREFIX), f"The unknown herald's slot holds its ToolError: {results[3]!r}"


def test_one_worker_still_serves_everyone():
    registry = loop.ToolRegistry()
    for i in range(3):
        registry.register(loop.Tool(f"h{i}", "", NO_PARAMS, (lambda i=i: str(i))))
    calls = [ToolCall(f"c{i}", f"h{i}", {}) for i in range(3)]
    assert cell.run_tools_parallel(registry, calls, max_workers=1) == ["0", "1", "2"]


# ------------------------------------------------------------ stream assembler
def test_the_scribe_waits_for_the_last_brace():
    chunks = ['{"na', 'me": "a}b", "n', '": [1, 2', "]}"]
    asm = cell.JSONStreamAssembler()
    for chunk in chunks[:-1]:
        asm.feed(chunk)
        assert asm.complete() is False, (
            f"Declared complete after {asm.text!r}. The braces are not balanced yet (the '}}' inside \"a}}b\" is a character)."
        )
    asm.feed(chunks[-1])
    assert asm.complete() is True, f"All chunks are in: {asm.text!r} is complete JSON."
    assert asm.result() == {"name": "a}b", "n": [1, 2]}


def test_escaped_quotes_do_not_end_the_string():
    chunks = ['{"say": "he said \\"', '}\\" and left"', "}"]
    asm = cell.JSONStreamAssembler()
    asm.feed(chunks[0])
    assert asm.complete() is False
    asm.feed(chunks[1])
    assert asm.complete() is False, f"{asm.text!r} still has an open object; the escaped quote did not end the string."
    asm.feed(chunks[2])
    assert asm.complete() is True
    assert asm.result() == {"say": 'he said "}" and left'}


def test_a_top_level_array_streams_too():
    asm = cell.JSONStreamAssembler()
    for chunk in ["[1, ", '{"a": 1}', ", 3"]:
        asm.feed(chunk)
        assert asm.complete() is False
    asm.feed("]")
    assert asm.result() == [1, {"a": 1}, 3]


def test_result_before_completion_is_an_error():
    asm = cell.JSONStreamAssembler()
    asm.feed('{"half": ')
    with pytest.raises(ValueError):
        asm.result()


def test_too_many_closers_is_never_complete():
    asm = cell.JSONStreamAssembler()
    asm.feed('{"a": 1}}')
    assert asm.complete() is False, "An extra closing brace can never become valid JSON."
    empty = cell.JSONStreamAssembler()
    assert empty.complete() is False, "Nothing fed, nothing complete."
