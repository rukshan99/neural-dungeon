"""ROOM 3.2 - THE TEMPERING  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The one equation behind every initializer: for z = x @ W with independent
zero-mean entries, Var(z_j) = fan_in * Var(w) * Var(x). To keep the signal's
scale from multiplying itself by a constant at every layer you want
fan_in * Var(w) ~ 1 (linear-ish activations such as tanh near 0) or ~ 2
(ReLU, which discards half of the variance).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

InitFn = Callable[[int, int, np.random.Generator], np.ndarray]


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """U(-a, a) with a = sqrt(6 / (fan_in + fan_out)), so Var(w) = 2 / (fan_in + fan_out)."""
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out))


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """N(0, 2 / fan_in): the factor 2 pays for the half of the variance ReLU throws away."""
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


def small_normal(fan_in: int, fan_out: int, rng: np.random.Generator, std: float = 0.01) -> np.ndarray:
    """N(0, std^2) with a fixed, tiny std. Looks harmless. Watch what it does at depth."""
    return rng.standard_normal((fan_in, fan_out)) * std


def large_normal(fan_in: int, fan_out: int, rng: np.random.Generator, std: float = 1.0) -> np.ndarray:
    """N(0, std^2) with a fixed, large std. Every pre-activation is huge from layer 1."""
    return rng.standard_normal((fan_in, fan_out)) * std


_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
}


def activation_statistics(
    depth: int,
    width: int,
    init_fn: InitFn,
    activation: str,
    rng: np.random.Generator,
    n_samples: int = 512,
) -> dict[str, np.ndarray]:
    """Push standard-normal inputs through ``depth`` square layers; record what comes out.

    Returns {"std": (depth,), "saturated": (depth,)} where std[l] is the standard
    deviation of layer l's output (over every element) and saturated[l] is the
    fraction of those outputs with |a| > 0.99. No biases: they start at zero and
    would not change the picture.
    """
    act = _ACTIVATIONS[activation]
    x = rng.standard_normal((n_samples, width))
    stds = np.zeros(depth)
    saturated = np.zeros(depth)
    for layer in range(depth):
        W = init_fn(width, width, rng)
        x = act(x @ W)
        stds[layer] = x.std()
        saturated[layer] = np.mean(np.abs(x) > 0.99)
    return {"std": stds, "saturated": saturated}


# The Prophecy. Inputs are standard normal, so "std" is already relative to the input.
#   "collapses":  the final std is below 1e-2 (the signal, and every gradient, is gone)
#   "saturates":  more than half the final units are pinned (|a| > 0.99) or the std
#                 blew past 100 (ReLU has no ceiling, so its saturation is an explosion)
#   "healthy":    the final std is within [0.1, 10] and fewer than half the units are pinned
TEMPERING_PROPHECY: dict[tuple[str, str], str] = {
    ("small_normal", "tanh"): "collapses",  # 0.01 * sqrt(256) = 0.16 per layer -> 0.16**20
    ("xavier_uniform", "tanh"): "healthy",  # the case Xavier init was derived for
    ("he_normal", "relu"): "healthy",  # the case He init was derived for
    ("xavier_uniform", "relu"): "collapses",  # ReLU halves the variance; 2**-10 after 20 layers
    ("large_normal", "tanh"): "saturates",  # pre-activations ~16 std: tanh pins everything to +-1
    ("large_normal", "relu"): "saturates",  # no ceiling: the std multiplies by ~11 per layer
}
