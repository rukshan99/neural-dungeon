"""TRIAL 0.3 - THE BROADCASTING BRIDGE

Each plank is one function. The trial also reads your source: a single Python
loop and the bridge sways.
"""

import numpy as np
import pytest

from dungeon.scrutiny import is_stub, python_loops_in
from dungeon.trials import load_room

room = load_room(__file__, "room_3_broadcasting_bridge")

rng = np.random.default_rng(3)

PLANKS = [
    "center_rows",
    "normalize_rows",
    "standardize_columns",
    "outer_sum",
    "pairwise_distances",
    "scale_channels",
    "clamp_to_range",
]


@pytest.mark.parametrize("name", PLANKS)
def test_the_bridge_holds_no_loops(name):
    fn = getattr(room, name)
    if is_stub(fn):
        raise NotImplementedError(f"{name}() is unwritten")
    loops = python_loops_in(fn)
    assert not loops, (
        f"The bridge sways: {name}() contains {', '.join(loops)}. "
        "No for/while/comprehensions on this floor. Broadcast instead."
    )


def test_center_rows_makes_every_row_sum_to_zero():
    x = rng.standard_normal((5, 8)) + 3.0
    out = room.center_rows(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out[2], x[2] - x[2].mean())


def test_normalize_rows_gives_unit_norms_and_spares_the_zero_row():
    x = rng.standard_normal((6, 4))
    x[3] = 0.0
    out = room.normalize_rows(x)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(np.delete(norms, 3), 1.0, atol=1e-9)
    assert np.all(out[3] == 0.0) and not np.any(np.isnan(out)), (
        "A row of zeros divided by its own norm is NaN. That is what eps is for."
    )


def test_standardize_columns_has_zero_mean_unit_std_per_column():
    x = rng.standard_normal((100, 3)) * np.array([1.0, 10.0, 100.0]) + np.array([0.0, 5.0, -5.0])
    out = room.standardize_columns(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-6)


def test_outer_sum_shape_and_values():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    out = room.outer_sum(a, b)
    assert out.shape == (3, 2), f"expected (3, 2), got {out.shape}. a[:, None] + b[None, :]."
    np.testing.assert_array_equal(out, [[11, 21], [12, 22], [13, 23]])


def test_pairwise_distances_match_a_slow_reference():
    a = rng.standard_normal((7, 3))
    b = rng.standard_normal((5, 3))
    out = room.pairwise_distances(a, b)
    assert out.shape == (7, 5), f"expected (7, 5), got {out.shape}"
    expected = np.array([[np.linalg.norm(ai - bj) for bj in b] for ai in a])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_pairwise_distance_to_self_is_zero_on_the_diagonal():
    a = rng.standard_normal((4, 2))
    np.testing.assert_allclose(np.diag(room.pairwise_distances(a, a)), 0.0, atol=1e-12)


def test_scale_channels_scales_the_last_axis():
    images = rng.random((2, 4, 4, 3))
    scale = np.array([1.0, 0.0, 2.0])
    out = room.scale_channels(images, scale)
    assert out.shape == images.shape
    assert np.all(out[..., 1] == 0.0), "channel 1 should be scaled to zero"
    np.testing.assert_allclose(out[..., 2], 2.0 * images[..., 2])


def test_clamp_to_range_per_column():
    x = np.array([[-5.0, 0.5, 9.0], [0.2, 2.0, -1.0]])
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 5.0])
    out = room.clamp_to_range(x, low, high)
    np.testing.assert_array_equal(out, [[0.0, 0.5, 5.0], [0.2, 1.0, 0.0]])
