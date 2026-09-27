"""BOSS - THE VANISHING WRAITH  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Plain numpy backprop through a deep chain. The gradient at layer l is
    dL/dx_l = W_l^T ( act'(z_l) * dL/dx_{l+1} )
so its norm changes by roughly  ||act'|| * init_scale * sqrt(width)  per layer,
and fifty layers of that is a fiftieth power.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# ------------------------------------------------------------------ given
# (forward, derivative-as-a-function-of-the-input) for each activation.
ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (
        lambda z: 1.0 / (1.0 + np.exp(-z)),
        lambda z: (1.0 / (1.0 + np.exp(-z))) * (1.0 - 1.0 / (1.0 + np.exp(-z))),
    ),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0.0).astype(np.float64)),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}

# Named initialization scales, all of the form  c / sqrt(width).
INIT_SCALES: dict[str, Callable[[int], float]] = {
    "small": lambda width: 0.5 / math.sqrt(width),
    "xavier": lambda width: 1.0 / math.sqrt(width),
    "he": lambda width: math.sqrt(2.0 / width),
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
    """L2 norm of dLoss/dx_l for l = 0..depth, for loss = sum(x_depth).

    Draw order (the trial rebuilds the same network):
        rng = np.random.default_rng(seed)
        x_0 = rng.standard_normal(width)
        W_l = init_scale * rng.standard_normal((width, width))   for l = 0..depth-1
    """
    act, dact = ACTIVATIONS[activation]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(width)
    weights = [init_scale * rng.standard_normal((width, width)) for _ in range(depth)]

    # Forward: keep every pre-activation, the backward pass needs act'(z_l).
    xs = [x]
    zs = []
    for W in weights:
        z = W @ xs[-1]
        zs.append(z)
        xs.append(xs[-1] + act(z) if use_residual else act(z))

    # Backward: loss = sum(x_L)  =>  dL/dx_L = ones.
    grad = np.ones(width)
    norms = [np.linalg.norm(grad)]
    for W, z in zip(reversed(weights), reversed(zs)):
        dz = dact(z) * grad  # through the activation
        grad_prev = W.T @ dz  # through the matmul
        if use_residual:
            grad_prev = grad_prev + grad  # the skip connection: an identity path
        grad = grad_prev
        norms.append(np.linalg.norm(grad))
    return np.array(norms[::-1])  # index l = layer l


# ---------------------------------------------------------------- phase 2
WRAITH_PROPHECY: dict[tuple[str, str], str | None] = {
    ("sigmoid", "small"): "vanishes",
    ("tanh", "xavier"): "survives",
    ("relu", "he"): "survives",
    ("relu", "large"): "explodes",
}


# ---------------------------------------------------------------- phase 3
def banish_the_wraith() -> dict:
    """A configuration whose gradient norm at layer 0 is within [0.05, 20] x the norm at layer 50.

    He initialization with ReLU keeps E||dL/dx_l||^2 constant through depth:
    each layer multiplies the squared norm by  init_scale^2 * width * P(z > 0)
    = (2 / width) * width * 1/2 = 1.
    """
    return {"activation": "relu", "init": "he", "use_residual": False}
