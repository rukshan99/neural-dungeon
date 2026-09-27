"""ROOM 3.2 - THE TEMPERING

    A blade quenched too fast shatters. Quenched too slow, it will not hold an
    edge. The smith watches the colour of the steel at every stage. You will
    watch the standard deviation of the activations at every layer.

Initialization decides whether a signal can travel through a deep stack at all.
For one layer z = x @ W with independent zero-mean weights and inputs,

    Var(z_j) = fan_in * Var(w) * Var(x)

so each layer multiplies the variance by fan_in * Var(w). Over 20 layers that
factor is raised to the 20th power. Anything but ~1 is a disaster: 0.5**20 is
1e-6 (the signal, and every gradient behind it, vanishes); 2**20 is 1e6.

* Xavier/Glorot: Var(w) = 2 / (fan_in + fan_out). Keeps the factor near 1 in
  both directions for activations that are roughly linear near zero (tanh).
  The uniform version draws from U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).
* He/Kaiming: Var(w) = 2 / fan_in. ReLU zeroes half of its inputs, which
  halves the variance; the 2 pays for it. Normal version: N(0, 2 / fan_in).
* Constant-std inits ("just use 0.01") ignore fan_in entirely. Fine for one
  layer, fatal at depth. You will measure exactly how fatal.

tanh adds a second failure mode that the std alone cannot see: with huge
pre-activations every unit is pinned at +-1 (saturated). Its std looks healthy
(about 1) but its derivative 1 - tanh^2 is ~0 everywhere, so nothing learns.
That is why activation_statistics also reports the saturated fraction.

Then the Prophecy: predict how each (init, activation) pair behaves at depth 20
BEFORE running anything. The trial measures and compares.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

InitFn = Callable[[int, int, np.random.Generator], np.ndarray]


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """(fan_in, fan_out) drawn from U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).

    Use ``rng.uniform``. Var(w) comes out as 2 / (fan_in + fan_out).
    """
    raise NotImplementedError("xavier_uniform() is unwritten")


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """(fan_in, fan_out) drawn from N(0, 2 / fan_in). Use ``rng.standard_normal`` times a std."""
    raise NotImplementedError("he_normal() is unwritten")


def small_normal(fan_in: int, fan_out: int, rng: np.random.Generator, std: float = 0.01) -> np.ndarray:
    """(fan_in, fan_out) drawn from N(0, std^2). The std ignores fan_in on purpose."""
    raise NotImplementedError("small_normal() is unwritten")


def large_normal(fan_in: int, fan_out: int, rng: np.random.Generator, std: float = 1.0) -> np.ndarray:
    """(fan_in, fan_out) drawn from N(0, std^2). Same as small_normal with a different default."""
    raise NotImplementedError("large_normal() is unwritten")


def activation_statistics(
    depth: int,
    width: int,
    init_fn: InitFn,
    activation: str,
    rng: np.random.Generator,
    n_samples: int = 512,
) -> dict[str, np.ndarray]:
    """Push a standard-normal batch through ``depth`` square layers and record what comes out.

    x starts as rng.standard_normal((n_samples, width)). For each layer, draw
    W = init_fn(width, width, rng), compute x = act(x @ W) (no biases), and record

        std[l]       = x.std()                  over every element of the layer output
        saturated[l] = np.mean(np.abs(x) > 0.99) fraction of units pinned near +-1

    ``activation`` is "tanh" or "relu". Return {"std": (depth,), "saturated": (depth,)}.
    Draw W from rng in layer order so the trial can reproduce your numbers.
    """
    raise NotImplementedError("activation_statistics() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY
# For depth=20, width=256, standard-normal inputs. Predict the fate of the
# final layer's activations for each (init, activation) pair. The three fates:
#
#   "collapses"   final std < 1e-2: the signal is gone and so is every gradient
#   "saturates"   more than half the final units are pinned (|a| > 0.99), OR the
#                 std has blown past 100 (ReLU has no ceiling: its version of
#                 saturation is an explosion)
#   "healthy"     final std within [0.1, 10] and fewer than half the units pinned
#
# Reason from Var(z) = fan_in * Var(w) * Var(x). Replace every None with one of
# the three strings. Commit first; the trial measures afterwards.
# ---------------------------------------------------------------------------
TEMPERING_PROPHECY: dict[tuple[str, str], str | None] = {
    ("small_normal", "tanh"): None,
    ("xavier_uniform", "tanh"): None,
    ("he_normal", "relu"): None,
    ("xavier_uniform", "relu"): None,
    ("large_normal", "tanh"): None,
    ("large_normal", "relu"): None,
}
