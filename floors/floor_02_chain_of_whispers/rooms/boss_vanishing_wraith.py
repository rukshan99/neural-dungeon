"""BOSS - THE VANISHING WRAITH

                    .------.
                  .'  _  _  '.        "Say what you like at the far end.
                 /   (o)(o)   \\        By the time it reaches the door
                |      /\\      |       it will be a rounding error."
                |   \\______/   |
                 \\            /       The Wraith is fifty layers deep. Every
                .-'\\   ..   /'-.      layer it passes through multiplies the
               /    \\ .''. /    \\     gradient by a derivative smaller than
              /      '    '      \\    one. It does not need to attack you.
             ( ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ )  It only needs to wait.
              '~~~~~~~~~~~~~~~~~~~'

WEAKNESS: activations and initializations that keep the gradient's norm
roughly constant from one layer to the next, and skip connections that give
the gradient a path with derivative exactly 1.

The chain:   x_{l+1} = act(W_l @ x_l)     for l = 0 .. depth-1,   loss = sum(x_depth)

Backward through one layer (this is the whole fight):

    dL/dx_l = W_l^T ( act'(z_l) * dL/dx_{l+1} )        with z_l = W_l @ x_l

Each layer multiplies the gradient's norm by roughly  ||act'|| * init_scale * sqrt(width).
Fifty layers raise that factor to the fiftieth power. If it is 0.25 (sigmoid at
its very best) the gradient at layer 0 is 1e-30 of the gradient at layer 50. If
it is 2, it is 1e15 times bigger. The Wraith lives in the first case; its cousin
lives in the second.

The fight has three phases:

  Phase 1  gradient_norms_per_layer(...)  - build the chain and measure the wraith.
  Phase 2  WRAITH_PROPHECY                - predict who vanishes, who explodes, who survives.
  Phase 3  banish_the_wraith()            - name a configuration that survives 50 layers.

Run:  dungeon fight 2
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# ------------------------------------------------------------------ given
# (forward, derivative-as-a-function-of-the-INPUT) for each activation.
ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (
        lambda z: 1.0 / (1.0 + np.exp(-z)),
        lambda z: (1.0 / (1.0 + np.exp(-z))) * (1.0 - 1.0 / (1.0 + np.exp(-z))),
    ),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0.0).astype(np.float64)),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}

# Named initialization scales (the std of each weight), all of the form c / sqrt(width).
INIT_SCALES: dict[str, Callable[[int], float]] = {
    "small": lambda width: 0.5 / math.sqrt(width),
    "xavier": lambda width: 1.0 / math.sqrt(width),   # Glorot: keeps variance for tanh-ish units
    "he": lambda width: math.sqrt(2.0 / width),       # He: the 2 pays for ReLU killing half the units
    "large": lambda width: 3.0 / math.sqrt(width),
}


# ---------------------------------------------------------------- phase 1
def gradient_norms_per_layer(
    depth: int,
    width: int,
    activation: str,
    init_scale: float,
    seed: int,
    use_residual: bool = False,
) -> np.ndarray:
    """L2 norm of dLoss/dx_l for every l = 0..depth, for loss = sum(x_depth). Shape (depth + 1,).

    Draw order (the trial rebuilds the same network from the same seed):
        rng = np.random.default_rng(seed)
        x_0 = rng.standard_normal(width)
        W_l = init_scale * rng.standard_normal((width, width))   for l = 0, 1, ..., depth-1

    Forward:  z_l = W_l @ x_l,  x_{l+1} = act(z_l)
              (with use_residual:  x_{l+1} = x_l + act(z_l))
    Backward: dL/dx_depth = ones(width);  dL/dx_l = W_l^T (act'(z_l) * dL/dx_{l+1})
              (with use_residual, add the skip path:  + dL/dx_{l+1})

    Return norms[l] = ||dL/dx_l||_2 as a float64 array. norms[depth] is therefore sqrt(width).
    Plain numpy is fine; so is the Tensor from Room 2.3 via ``from .room_3_tensor_whisper import Tensor``.
    """
    raise NotImplementedError("gradient_norms_per_layer() is unwritten")


# ---------------------------------------------------------------- phase 2
# For each (activation, init) at depth 50, width 64: does the gradient norm at
# layer 0, divided by the norm at layer 50, come out below 1e-3 ("vanishes"),
# above 1e3 ("explodes"), or in between ("survives")? Predict before you run.
WRAITH_PROPHECY: dict[tuple[str, str], str | None] = {
    ("sigmoid", "small"): None,
    ("tanh", "xavier"): None,
    ("relu", "he"): None,
    ("relu", "large"): None,
}


# ---------------------------------------------------------------- phase 3
def banish_the_wraith() -> dict:
    """A configuration that survives depth 50 at width 64 for seeds 0, 1 and 2.

    Return {"activation": <"sigmoid" | "tanh" | "relu">,
            "init": <a name from INIT_SCALES>   (or "init_scale": <float>),
            "use_residual": <bool>}

    The trial builds the chain and demands  0.05 <= norms[0] / norms[50] <= 20  for every seed.
    "linear" is not accepted: a chain with no nonlinearity is not a network.
    """
    raise NotImplementedError("banish_the_wraith() is unwritten")
