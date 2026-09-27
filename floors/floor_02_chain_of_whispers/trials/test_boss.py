"""BOSS FIGHT - THE VANISHING WRAITH

Phase 1: build the deep chain and measure the gradient norm at every layer.
Phase 2: prophesy who vanishes, who explodes and who survives fifty layers.
Phase 3: name a configuration the Wraith cannot starve.

The trial judges with its own independent numpy backprop, so a wrong Phase 1
cannot bluff its way through Phases 2 and 3.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

boss = load_room(__file__, "boss_vanishing_wraith")

pytestmark = pytest.mark.boss

DEPTH, WIDTH = 50, 64
VANISH_BELOW, EXPLODE_ABOVE = 1e-3, 1e3
SURVIVE_LOW, SURVIVE_HIGH = 0.05, 20.0
PROPHECY_KEYS = [("sigmoid", "small"), ("tanh", "xavier"), ("relu", "he"), ("relu", "large")]


def _activation(name):
    if name == "sigmoid":
        s = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return s, lambda z: s(z) * (1.0 - s(z))
    if name == "tanh":
        return np.tanh, lambda z: 1.0 / np.cosh(z) ** 2
    if name == "relu":
        return (lambda z: z * (z > 0)), (lambda z: (z > 0).astype(float))
    if name == "linear":
        return (lambda z: z), (lambda z: np.ones_like(z))
    raise ValueError(name)


def _reference_norms(depth, width, activation, init_scale, seed, use_residual=False):
    """Independent backprop through the chain, written as explicit Jacobian-vector products."""
    act, dact = _activation(activation)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(width)
    Ws = [init_scale * rng.standard_normal((width, width)) for _ in range(depth)]
    zs = []
    for W in Ws:
        z = W @ x
        zs.append(z)
        x = x + act(z) if use_residual else act(z)
    g = np.ones(width)
    norms = [float(np.linalg.norm(g))]
    for W, z in zip(Ws[::-1], zs[::-1]):
        jac = (dact(z)[:, None] * W)  # d x_{l+1} / d x_l  (without the skip)
        if use_residual:
            jac = jac + np.eye(width)
        g = jac.T @ g
        norms.append(float(np.linalg.norm(g)))
    return np.array(norms[::-1])


def _ratio(activation, init_scale, seed, use_residual=False):
    n = _reference_norms(DEPTH, WIDTH, activation, init_scale, seed, use_residual)
    return n[0] / n[-1]


def _classify(ratio: float) -> str:
    if ratio < VANISH_BELOW:
        return "vanishes"
    if ratio > EXPLODE_ABOVE:
        return "explodes"
    return "survives"


# ------------------------------------------------------------------ phase 1
def test_phase_1_one_norm_per_layer_boundary():
    norms = boss.gradient_norms_per_layer(depth=6, width=5, activation="tanh", init_scale=0.4, seed=1)
    norms = np.asarray(norms)
    assert norms.shape == (7,), f"depth 6 has 7 layer boundaries x_0..x_6: expected shape (7,), got {norms.shape}"
    assert np.all(np.isfinite(norms)) and np.all(norms > 0), f"norms must be finite and positive, got {norms}"
    assert math.isclose(norms[-1], math.sqrt(5)), (
        f"loss = sum(x_L) means dL/dx_L is a vector of ones, whose norm is sqrt(width) = {math.sqrt(5):.4f}; "
        f"got {norms[-1]:.4f} for the last entry. Is norms[l] indexed by layer, with layer 0 first?"
    )


@pytest.mark.parametrize(
    "activation,depth,width,scale",
    [("linear", 1, 3, 0.7), ("relu", 3, 4, 0.9), ("tanh", 4, 6, 0.5), ("sigmoid", 5, 3, 1.5)],
    ids=["linear-1", "relu-3", "tanh-4", "sigmoid-5"],
)
def test_phase_1_matches_an_independent_backprop(activation, depth, width, scale):
    got = np.asarray(boss.gradient_norms_per_layer(depth, width, activation, scale, seed=5))
    expected = _reference_norms(depth, width, activation, scale, seed=5)
    assert got.shape == expected.shape, f"expected shape {expected.shape}, got {got.shape}"
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-12, err_msg=(
        f"{activation}, depth {depth}: your per-layer norms differ from the reference backprop. "
        "Check the draw order (x_0 first, then W_0..W_{depth-1}) and that dL/dx_l = W_l^T (act'(z_l) * dL/dx_{l+1})."
    ))


def test_phase_1_the_residual_path_is_an_identity_whisper():
    got = np.asarray(boss.gradient_norms_per_layer(4, 5, "relu", 0.8, seed=2, use_residual=True))
    expected = _reference_norms(4, 5, "relu", 0.8, seed=2, use_residual=True)
    np.testing.assert_allclose(got, expected, rtol=1e-8, err_msg=(
        "with use_residual=True the forward is x + act(W x) and the backward gains a term that passes "
        "dL/dx_{l+1} straight through, unchanged"
    ))


def test_phase_1_the_wraith_appears():
    scale = boss.INIT_SCALES["small"](WIDTH)
    norms = np.asarray(boss.gradient_norms_per_layer(DEPTH, WIDTH, "sigmoid", scale, seed=0))
    ratio = norms[0] / norms[-1]
    assert ratio < 1e-6, (
        f"sigmoid with small init at depth 50: the gradient norm at layer 0 should be under 1e-6 of the norm "
        f"at layer 50 (it is about 1e-46). Yours is {ratio:.3e}. Are the layers indexed the right way round?"
    )
    assert norms[0] > 0.0, "the whisper is tiny but it is not exactly zero in float64"


# ------------------------------------------------------------------ phase 2
def test_phase_2_the_prophecy_names_every_combination():
    assert set(boss.WRAITH_PROPHECY) == set(PROPHECY_KEYS), "Do not add or remove entries from WRAITH_PROPHECY; answer them."


@pytest.mark.parametrize("combo", PROPHECY_KEYS, ids=[f"{a}-{i}" for a, i in PROPHECY_KEYS])
def test_phase_2_prophesy_the_fate_of_the_whisper(combo):
    prediction = boss.WRAITH_PROPHECY.get(combo)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {combo}. Fill in WRAITH_PROPHECY.")
    activation, init = combo
    scale = boss.INIT_SCALES[init](WIDTH)
    ratios = [_ratio(activation, scale, seed) for seed in range(3)]
    typical = float(np.exp(np.mean(np.log(ratios))))
    verdict = _classify(typical)
    assert prediction == verdict, (
        f"{activation} with {init} init at depth {DEPTH}, width {WIDTH}: the gradient norm at layer 0 is "
        f"{typical:.3e} x the norm at layer {DEPTH}, so it {verdict} (vanishes < {VANISH_BELOW:g}, "
        f"explodes > {EXPLODE_ABOVE:g}). You prophesied {prediction!r}."
    )


# ------------------------------------------------------------------ phase 3
def _config():
    cfg = boss.banish_the_wraith()
    assert isinstance(cfg, dict), f"banish_the_wraith() must return a dict, got {type(cfg).__name__}"
    for key in ("activation", "use_residual"):
        assert key in cfg, f"the configuration needs a {key!r} entry"
    assert "init" in cfg or "init_scale" in cfg, "name an init ('init') or give a number ('init_scale')"
    activation = cfg["activation"]
    assert activation in ("sigmoid", "tanh", "relu"), (
        f"activation must be sigmoid, tanh or relu, got {activation!r}. A chain with no nonlinearity is not a network."
    )
    if "init_scale" in cfg:
        scale = float(cfg["init_scale"])
    else:
        assert cfg["init"] in boss.INIT_SCALES, f"unknown init {cfg['init']!r}; choose from {sorted(boss.INIT_SCALES)}"
        scale = boss.INIT_SCALES[cfg["init"]](WIDTH)
    assert isinstance(cfg["use_residual"], bool), "use_residual must be a bool"
    return activation, scale, cfg["use_residual"]


def test_phase_3_the_configuration_is_well_formed():
    _config()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phase_3_the_whisper_survives_fifty_layers(seed):
    activation, scale, residual = _config()
    ratio = _ratio(activation, scale, seed, residual)
    assert SURVIVE_LOW <= ratio <= SURVIVE_HIGH, (
        f"seed {seed}: with {activation}, init scale {scale:.4f}{' and residual connections' if residual else ''}, "
        f"the gradient norm at layer 0 is {ratio:.3e} x the norm at layer {DEPTH}. The Wraith demands it stay "
        f"within [{SURVIVE_LOW}, {SURVIVE_HIGH}]. "
        + ("It vanished." if ratio < SURVIVE_LOW else "It exploded: residual branches ADD variance every layer.")
    )
