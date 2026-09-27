"""BOSS FIGHT - THE BROADCASTING BASILISK

Phase 1: the Basilisk petrifies numpy's broadcasting helpers and array
         constructors while you compute broadcast shapes by hand.
Phase 2: insert size-1 axes on purpose.
Phase 3: stare back - your predictions against numpy's verdict.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

boss = load_room(__file__, "boss_broadcasting_basilisk")

pytestmark = pytest.mark.boss


class BasiliskGaze(RuntimeError):
    """Raised when you look at one of the petrified helpers."""


def _petrify(name):
    def petrified(*args, **kwargs):
        raise BasiliskGaze(
            f"The Basilisk's gaze falls on np.{name}. It turns to stone in your hands. "
            "Apply the broadcasting rules to the shape tuples yourself."
        )

    return petrified


PETRIFIED = ["broadcast_shapes", "broadcast_to", "broadcast_arrays", "broadcast",
             "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like"]


@pytest.fixture
def gaze(monkeypatch):
    for name in PETRIFIED:
        monkeypatch.setattr(np, name, _petrify(name))
    yield


# Cases are (shapes, expected). Expected is computed by real numpy BEFORE petrification.
def _expected(*shapes):
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError:
        return None


PHASE1_CASES = [
    ((5, 4), (1,)),
    ((5, 4), (4,)),
    ((15, 3, 5), (15, 1, 5)),
    ((15, 3, 5), (3, 5)),
    ((15, 3, 5), (3, 1)),
    ((8, 1, 6, 1), (7, 1, 5)),
    ((3,), (4,)),
    ((2, 1), (8, 4, 3)),
    ((1,), (1,)),
    ((), (3, 2)),
    ((2, 3), (2, 3)),
    ((4, 1), (4,)),
    ((3, 1, 1), (1, 5, 1), (1, 1, 7)),
    ((2, 3), (3,), (1, 3)),
    ((2, 3), (3,), (2, 1), (4, 1, 1)),
    ((2, 3), (4, 3), (1, 3)),
    ((100_000, 1, 100_000), (1, 100_000, 1)),          # 10**15 elements
    ((10**6, 10**6), (10**6, 1)),                        # 10**12 elements
    ((10**9, 1), (1, 10**9), (10**9, 10**9)),            # 10**18 elements
]
PHASE1_EXPECTED = {shapes: _expected(*shapes) for shapes in PHASE1_CASES}


@pytest.mark.parametrize("shapes", PHASE1_CASES, ids=[str(s) for s in PHASE1_CASES])
def test_phase_1_the_rules_by_hand(gaze, shapes):
    expected = PHASE1_EXPECTED[shapes]
    try:
        got = boss.broadcast_shape(*shapes)
    except BasiliskGaze as exc:
        pytest.fail(str(exc))
    except MemoryError:
        pytest.fail("You tried to build an array with more elements than there are grains of sand. Work on the tuples.")
    if expected is None:
        assert got is None, f"{shapes} cannot broadcast, but you returned {got}. Rule 2: equal or 1."
    else:
        assert got is not None and tuple(got) == tuple(expected), (
            f"{shapes} broadcasts to {tuple(expected)}, you said {got}."
        )


def test_phase_1_no_shapes_is_the_empty_shape(gaze):
    assert boss.broadcast_shape() == (), "broadcast_shape() with no arguments is ()."


def test_phase_1_result_is_a_tuple_of_python_ints(gaze):
    got = boss.broadcast_shape((2, 1), (1, 3))
    assert isinstance(got, tuple) and all(type(d) is int for d in got), "Return a tuple of plain ints."


# --------------------------------------------------------------------- phase 2
def test_phase_2_along_axis_places_the_vector():
    v = np.array([1.0, 2.0, 3.0])
    assert boss.along_axis(v, ndim=4, axis=1).shape == (1, 3, 1, 1)
    assert boss.along_axis(v, ndim=2, axis=0).shape == (3, 1)
    assert boss.along_axis(v, ndim=3, axis=-1).shape == (1, 1, 3), "Negative axes count from the end."
    with pytest.raises(ValueError):
        boss.along_axis(np.ones((2, 2)), ndim=3, axis=0)


def test_phase_2_scale_along_any_axis():
    rng = np.random.default_rng(0)
    x = rng.random((2, 3, 4, 5))  # B, C, H, W
    gain = np.array([1.0, 0.0, 10.0])
    out = boss.scale_along(x, gain, axis=1)
    assert out.shape == x.shape
    assert np.all(out[:, 1] == 0.0)
    np.testing.assert_allclose(out[:, 2], 10.0 * x[:, 2])
    # and along the last axis, the plain-broadcasting case
    gain_w = np.arange(5.0)
    np.testing.assert_allclose(boss.scale_along(x, gain_w, axis=-1), x * gain_w)


def test_phase_2_petrified_zeroes_whole_feature_vectors():
    rng = np.random.default_rng(1)
    x = rng.random((2, 4, 3)) + 1.0  # B, T, D  (all > 0)
    mask = np.array([[True, False, False, True], [False, False, True, False]])
    out = boss.petrified(x, mask)
    assert out.shape == x.shape
    assert np.all(out[mask] == 0.0), "Masked positions should be all-zero vectors."
    np.testing.assert_array_equal(out[~mask], x[~mask])


# --------------------------------------------------------------------- phase 3
RIDDLE_KEYS = [
    ((8, 1, 6, 1), (7, 1, 5)),
    ((5, 4), (1,)),
    ((5, 4), (4,)),
    ((15, 3, 5), (15, 1, 5)),
    ((15, 3, 5), (3, 5)),
    ((15, 3, 5), (3, 1)),
    ((3,), (4,)),
    ((2, 1), (8, 4, 3)),
    ((4, 1), (4,)),
    ((1, 1, 1), ()),
]


def test_phase_3_riddles_are_all_present():
    assert set(boss.BASILISK_RIDDLES) == set(RIDDLE_KEYS), "Do not add or remove riddles; answer them."


@pytest.mark.parametrize("pair", RIDDLE_KEYS, ids=[f"{a}x{b}" for a, b in RIDDLE_KEYS])
def test_phase_3_stare_back(pair):
    prediction = boss.BASILISK_RIDDLES.get(pair)
    if prediction is None:
        raise NotImplementedError(f"You have not answered the riddle {pair}. Fill in BASILISK_RIDDLES.")
    verdict = _expected(*pair)
    if verdict is None:
        assert prediction == "error", f"{pair[0]} and {pair[1]} do not broadcast; you predicted {prediction}."
    else:
        assert tuple(prediction) == tuple(verdict), (
            f"{pair[0]} with {pair[1]} broadcasts to {tuple(verdict)}; you predicted {prediction}."
        )
