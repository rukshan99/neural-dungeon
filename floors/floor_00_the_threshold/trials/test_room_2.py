"""TRIAL 0.2 - THE HALL OF SHAPES

Put the statues back on the right plinths. Then the Prophecy is read aloud.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_hall_of_shapes")

rng = np.random.default_rng(0)


def test_flatten_batch_keeps_the_batch_and_flattens_the_rest():
    x = rng.standard_normal((4, 3, 5))
    out = room.flatten_batch(x)
    assert out.shape == (4, 15), f"expected (4, 15), got {out.shape}"
    assert np.array_equal(out[2], x[2].ravel()), "Row 2 of the output must be example 2, flattened in order."


def test_flatten_batch_works_for_any_number_of_trailing_axes():
    x = rng.standard_normal((2, 3, 4, 5))
    assert room.flatten_batch(x).shape == (2, 60)


def test_channels_move_first_without_reordering_pixels():
    x = rng.standard_normal((2, 5, 7, 3))  # B, H, W, C
    out = room.to_channels_first(x)
    assert out.shape == (2, 3, 5, 7), f"expected (2, 3, 5, 7), got {out.shape}"
    # The pixel at (batch 1, row 4, col 6, channel 2) must be the same value.
    assert out[1, 2, 4, 6] == x[1, 4, 6, 2], (
        "Pixels were scrambled. You reshaped where you should have transposed: "
        "reshape reinterprets memory, transpose reorders axes."
    )


def test_stack_pairs_creates_a_new_middle_axis():
    a = rng.standard_normal((6, 4))
    b = rng.standard_normal((6, 4))
    out = room.stack_pairs(a, b)
    assert out.shape == (6, 2, 4), f"expected (6, 2, 4), got {out.shape}"
    assert np.array_equal(out[:, 0], a) and np.array_equal(out[:, 1], b), (
        "out[:, 0] should be a and out[:, 1] should be b. np.stack(..., axis=1)."
    )


def test_last_column_is_one_dimensional():
    x = np.arange(12).reshape(3, 4)
    out = room.last_column(x)
    assert out.shape == (3,), f"expected (3,), got {out.shape}. x[:, -1:] keeps a dim; x[:, -1] drops it."
    assert np.array_equal(out, [3, 7, 11])


def test_every_other_row():
    x = np.arange(20).reshape(5, 4)
    out = room.every_other_row(x)
    assert out.shape == (3, 4)
    assert np.array_equal(out, x[[0, 2, 4]])


def test_add_batch_axis_makes_a_batch_of_one():
    x = rng.standard_normal((5, 6))
    out = room.add_batch_axis(x)
    assert out.shape == (1, 5, 6), f"expected (1, 5, 6), got {out.shape}"
    assert np.shares_memory(out, x) or np.array_equal(out[0], x)


# The trial owns the list of expressions: only these strings are ever evaluated,
# whatever the room's dictionary happens to contain.
PROPHECY_KEYS = [
    "np.zeros((3, 4)).T",
    "np.zeros((2, 3, 4)).reshape(6, -1)",
    "np.zeros((5,))[:, None]",
    "np.zeros((2, 3, 4)).sum(axis=1)",
    "np.zeros((2, 3, 4)).mean(axis=(0, 2))",
    "np.zeros((2, 3, 4)).sum(axis=1, keepdims=True)",
    "np.zeros((3, 1, 4)) + np.zeros((5, 4))",
    "np.zeros((4, 3)) @ np.zeros((3, 2))",
    "np.zeros((2, 3, 4)).transpose(2, 0, 1)",
    "np.zeros((6,))[None, :, None]",
]


@pytest.mark.parametrize("expression", PROPHECY_KEYS, ids=PROPHECY_KEYS)
def test_the_prophecy_holds(expression):
    prediction = room.SHAPE_PROPHECY.get(expression)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {expression!r}. Fill in SHAPE_PROPHECY.")
    actual = eval(expression, {"np": np}).shape  # noqa: S307 - PROPHECY_KEYS above are the trial's own strings
    assert tuple(prediction) == actual, (
        f"The prophecy said {tuple(prediction)} for {expression}, but the machine produced {actual}."
    )


def test_the_prophecy_is_complete():
    assert set(room.SHAPE_PROPHECY) == set(PROPHECY_KEYS), "Do not remove or rename the prophecy's expressions."
