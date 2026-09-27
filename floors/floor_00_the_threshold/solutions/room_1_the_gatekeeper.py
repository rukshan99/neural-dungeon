"""ROOM 0.1 - THE GATEKEEPER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import numpy as np


def answer_the_gatekeeper() -> int:
    """The sum of the integers 1..100, computed with numpy, returned as a Python int."""
    return int(np.arange(1, 101).sum())


def light_the_torches(n: int) -> np.ndarray:
    """A 1-D array of ``n`` ones with dtype float32."""
    return np.ones(n, dtype=np.float32)


def count_the_stones(x) -> int:
    """How many scalar elements are in ``x`` (any shape, any dtype, list or array)."""
    return int(np.asarray(x).size)
