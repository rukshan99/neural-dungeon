"""TRIAL 0.1 - THE GATEKEEPER

The automaton asks three questions. Answer all three and the archway opens.
"""

import numpy as np

from dungeon.trials import load_room

room = load_room(__file__, "room_1_the_gatekeeper")


def test_the_gatekeeper_accepts_the_sum():
    answer = room.answer_the_gatekeeper()
    assert answer == 5050, f"The gatekeeper expected 5050 and heard {answer!r}."
    assert type(answer) is int, (
        f"The gatekeeper wants a plain Python int, not {type(answer).__name__}. Wrap it: int(...)"
    )


def test_the_torches_are_lit_in_float32():
    torches = room.light_the_torches(7)
    assert isinstance(torches, np.ndarray), "light_the_torches must return a numpy array."
    assert torches.shape == (7,), f"Seven torches were requested; you lit an array of shape {torches.shape}."
    assert torches.dtype == np.float32, (
        f"The torches burn in float32, yours are {torches.dtype}. numpy defaults to float64; say dtype=np.float32."
    )
    assert np.all(torches == 1.0), "Every torch should be lit (== 1.0)."


def test_no_torches_is_an_empty_array_not_an_error():
    torches = room.light_the_torches(0)
    assert torches.shape == (0,) and torches.dtype == np.float32


def test_the_stones_are_counted_whatever_their_arrangement():
    assert room.count_the_stones(np.zeros((3, 4, 5))) == 60, "3 x 4 x 5 stones is 60 stones."
    assert room.count_the_stones([[1, 2], [3, 4], [5, 6]]) == 6, "Nested lists are arrays in disguise: np.asarray."
    assert room.count_the_stones(np.array(7.0)) == 1, "A 0-d array holds exactly one stone."
    assert type(room.count_the_stones(np.zeros(3))) is int, "Return a Python int."
