"""TRIAL 3.2 - THE TEMPERING

The initializers are measured against their own definitions, the activation
statistics against a reference walk through the same random numbers, and the
Prophecy against what actually happens at depth 20.
"""

from functools import lru_cache

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_2_the_tempering")

DEPTH, WIDTH = 20, 256
N_SAMPLES = 256  # rows in the probe batch: enough for a std over 65k elements, cheap enough to walk 24 times


# ------------------------------------------------------------ reference implementations
def ref_xavier_uniform(fan_in, fan_out, rng):
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out))


def ref_he_normal(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


def ref_small_normal(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * 0.01


def ref_large_normal(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * 1.0


REF_INITS = {
    "xavier_uniform": ref_xavier_uniform,
    "he_normal": ref_he_normal,
    "small_normal": ref_small_normal,
    "large_normal": ref_large_normal,
}
ACTS = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0.0)}


def ref_activation_statistics(depth, width, init_fn, activation, rng, n_samples=512):
    act = ACTS[activation]
    x = rng.standard_normal((n_samples, width))
    stds, sat = np.zeros(depth), np.zeros(depth)
    for layer in range(depth):
        x = act(x @ init_fn(width, width, rng))
        stds[layer] = x.std()
        sat[layer] = np.mean(np.abs(x) > 0.99)
    return {"std": stds, "saturated": sat}


def fate(stats):
    std, sat = stats["std"][-1], stats["saturated"][-1]
    if std < 1e-2:
        return "collapses"
    if sat > 0.5 or std > 100:
        return "saturates"
    if 0.1 <= std <= 10:
        return "healthy"
    return "ambiguous"


PROPHECY_KEYS = [
    ("small_normal", "tanh"),
    ("xavier_uniform", "tanh"),
    ("he_normal", "relu"),
    ("xavier_uniform", "relu"),
    ("large_normal", "tanh"),
    ("large_normal", "relu"),
]


@lru_cache(maxsize=None)
def measured(init_name, activation, seed):
    """One reference walk at the prophecy's depth and width, cached: the walks are the slow part of this trial."""
    return ref_activation_statistics(DEPTH, WIDTH, REF_INITS[init_name], activation, np.random.default_rng(seed), N_SAMPLES)


@lru_cache(maxsize=None)
def truth_for(init_name, activation):
    """The fate, measured for three seeds. The dungeon refuses to prophesy on an ambiguous case."""
    fates = {fate(measured(init_name, activation, s)) for s in range(3)}
    assert len(fates) == 1 and "ambiguous" not in fates, f"trial misconfigured: {init_name}+{activation} gave {fates}"
    return fates.pop()


# ------------------------------------------------------------ initializers
def test_xavier_uniform_has_the_right_shape_bounds_and_variance():
    W = room.xavier_uniform(400, 300, np.random.default_rng(0))
    assert W.shape == (400, 300), f"expected (fan_in, fan_out) = (400, 300), got {W.shape}"
    a = np.sqrt(6.0 / 700)
    assert np.abs(W).max() <= a + 1e-12, f"values must lie in [-a, a] with a = sqrt(6/(fan_in+fan_out)) = {a:.4f}"
    np.testing.assert_allclose(W.var(), 2.0 / 700, rtol=0.05, err_msg="Var(w) should be 2 / (fan_in + fan_out).")
    assert abs(W.mean()) < 1e-3, "the distribution is centred at zero"


def test_he_normal_has_std_sqrt_two_over_fan_in():
    W = room.he_normal(500, 200, np.random.default_rng(1))
    assert W.shape == (500, 200), f"expected (500, 200), got {W.shape}"
    np.testing.assert_allclose(W.std(), np.sqrt(2.0 / 500), rtol=0.03, err_msg=(
        "He init: std = sqrt(2 / fan_in). The 2 pays for the half of the variance ReLU discards."
    ))
    assert abs(W.mean()) < 1e-3


def test_constant_std_inits_ignore_fan_in_on_purpose():
    small = room.small_normal(1000, 200, np.random.default_rng(2))
    large = room.large_normal(1000, 200, np.random.default_rng(3))
    np.testing.assert_allclose(small.std(), 0.01, rtol=0.03, err_msg="small_normal: std 0.01 regardless of fan_in")
    np.testing.assert_allclose(large.std(), 1.0, rtol=0.03, err_msg="large_normal: std 1.0 regardless of fan_in")
    np.testing.assert_allclose(room.small_normal(10, 10, np.random.default_rng(4), std=0.5).std(), 0.5, rtol=0.1)


@pytest.mark.parametrize("name", ["xavier_uniform", "he_normal", "small_normal", "large_normal"])
def test_initializers_draw_from_the_generator_they_are_given(name):
    fn = getattr(room, name)
    a = fn(50, 60, np.random.default_rng(7))
    b = fn(50, 60, np.random.default_rng(7))
    c = fn(50, 60, np.random.default_rng(8))
    np.testing.assert_array_equal(a, b, err_msg=f"{name}: the same seed must give the same matrix. Use rng, not np.random.")
    assert not np.array_equal(a, c), f"{name}: different seeds must give different matrices."


# ------------------------------------------------------------ activation statistics
def test_activation_statistics_returns_one_std_and_one_saturation_per_layer():
    stats = room.activation_statistics(5, 32, room.he_normal, "relu", np.random.default_rng(0))
    assert isinstance(stats, dict) and set(stats) >= {"std", "saturated"}, "return {'std': ..., 'saturated': ...}"
    assert np.shape(stats["std"]) == (5,), f"std must have one entry per layer: shape (5,), got {np.shape(stats['std'])}"
    assert np.shape(stats["saturated"]) == (5,), f"saturated must be shape (5,), got {np.shape(stats['saturated'])}"
    assert np.all(np.isfinite(stats["std"])) and np.all((stats["saturated"] >= 0) & (stats["saturated"] <= 1))


@pytest.mark.parametrize("init_name,activation", PROPHECY_KEYS, ids=[f"{i}+{a}" for i, a in PROPHECY_KEYS])
def test_activation_statistics_walks_the_same_path_as_the_reference(init_name, activation):
    mine = room.activation_statistics(8, 64, REF_INITS[init_name], activation, np.random.default_rng(11))
    ref = ref_activation_statistics(8, 64, REF_INITS[init_name], activation, np.random.default_rng(11))
    np.testing.assert_allclose(mine["std"], ref["std"], rtol=1e-6, atol=1e-12, err_msg=(
        "std per layer differs from the reference walk. Draw x first, then one W per layer from rng, "
        "apply the activation after each matmul, and take .std() over every element of the layer output."
    ))
    np.testing.assert_allclose(mine["saturated"], ref["saturated"], atol=1e-12, err_msg=(
        "saturated per layer differs: it is the fraction of elements with |a| > 0.99."
    ))


# ------------------------------------------------------------ the prophecy
def test_the_prophecy_is_complete():
    assert set(room.TEMPERING_PROPHECY) == set(PROPHECY_KEYS), "Do not add or remove pairs from TEMPERING_PROPHECY; answer them."


@pytest.mark.parametrize("init_name,activation", PROPHECY_KEYS, ids=[f"{i}+{a}" for i, a in PROPHECY_KEYS])
def test_the_prophecy_holds_at_depth_twenty(init_name, activation):
    prediction = room.TEMPERING_PROPHECY.get((init_name, activation))
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied ({init_name}, {activation}). Fill in TEMPERING_PROPHECY.")
    assert prediction in ("collapses", "saturates", "healthy"), f"{prediction!r} is not one of the three fates."
    truth = truth_for(init_name, activation)
    stats = measured(init_name, activation, 0)
    assert prediction == truth, (
        f"You prophesied {prediction!r} for {init_name}+{activation}; after {DEPTH} layers of width {WIDTH} "
        f"the activations have std {stats['std'][-1]:.3g} and {100 * stats['saturated'][-1]:.0f}% of units pinned: "
        f"that is {truth!r}. Reason from Var(z) = fan_in * Var(w) * Var(x), then apply the activation."
    )


@pytest.mark.parametrize("init_name,activation", PROPHECY_KEYS, ids=[f"{i}+{a}" for i, a in PROPHECY_KEYS])
def test_your_own_initializers_meet_the_same_fate(init_name, activation):
    mine = room.activation_statistics(DEPTH, WIDTH, getattr(room, init_name), activation, np.random.default_rng(5), N_SAMPLES)
    assert fate(mine) == truth_for(init_name, activation), (
        f"Using YOUR {init_name} and YOUR activation_statistics, {init_name}+{activation} ends with std "
        f"{mine['std'][-1]:.3g} and {100 * mine['saturated'][-1]:.0f}% pinned, which is {fate(mine)!r}; "
        f"it should be {truth_for(init_name, activation)!r}. One of the two functions has the wrong scale."
    )
