"""ROOM 0.1 - THE GATEKEEPER

    A rusted automaton blocks the archway. It does not want a password.
    It wants to know that your tools work and that you know how a trial feels.

This room is deliberately tiny. The point is the loop you will repeat for the
rest of the dungeon:

    1. read the docstring, 2. replace the `raise` with code,
    3. run   dungeon trial 0 room_1     4. read the verdict, repeat.

Everything below Floor 0 speaks in arrays. Notice the *dtype* and *shape* words
in the docstrings: the trials check them. numpy defaults to float64; most deep
learning happens in float32 (or smaller). Being deliberate about dtype is a
habit worth forming at the door.
"""

from __future__ import annotations

import numpy as np


def answer_the_gatekeeper() -> int:
    """The sum of the integers 1..100 (inclusive), computed with numpy.

    Return a plain Python ``int``, not a numpy scalar. (Hint: ``int(...)``.)
    """
    # TODO: replace this line with your answer.
    raise NotImplementedError("The gatekeeper waits for answer_the_gatekeeper()")


def light_the_torches(n: int) -> np.ndarray:
    """A 1-D array of ``n`` ones with dtype ``float32``.

    >>> light_the_torches(3)
    array([1., 1., 1.], dtype=float32)
    """
    raise NotImplementedError("Nobody has lit the torches: light_the_torches()")


def count_the_stones(x) -> int:
    """How many scalar elements are in ``x``, whatever its shape.

    ``x`` may be a numpy array *or* a nested Python list. Return a Python int.
    (Hint: ``np.asarray`` turns either into an array; arrays know their ``.size``.)
    """
    raise NotImplementedError("The stones are uncounted: count_the_stones()")
