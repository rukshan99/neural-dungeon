"""ROOM 1.2 - THE NUMERICAL ORACLE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The oracle is slow (two function calls per element) and that is fine: it is a
*test*, not a training step. Its job is to catch a wrong analytic gradient.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def forward_difference_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """(f(x + eps e_i) - f(x)) / eps for every element i. Error is O(eps)."""
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        bumped = x.copy()
        bumped[idx] += eps
        grad[idx] = (f(bumped) - f0) / eps
    return grad


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """(f(x + eps e_i) - f(x - eps e_i)) / (2 eps) for every element i. Error is O(eps**2).

    Works for x of any shape (including 0-d): np.ndindex walks every index.
    Never mutates x.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        up = x.copy()
        down = x.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2.0 * eps)
    return grad


def gradient_check(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Relative error ||g_num - g_ana|| / (||g_num|| + ||g_ana|| + eps).

    0 means perfect agreement, ~1 means the two point in different directions.
    In float64 a correct gradient scores below 1e-6; anything above 1e-3 is a
    bug. The eps in the denominator is only a guard against 0 / 0.
    """
    g_num = numerical_gradient(f, x, eps)
    g_ana = np.asarray(grad_f(x), dtype=float)
    if g_ana.shape != g_num.shape:
        raise ValueError(f"grad_f returned shape {g_ana.shape}, but x has shape {g_num.shape}")
    diff = np.linalg.norm(g_num - g_ana)
    return float(diff / (np.linalg.norm(g_num) + np.linalg.norm(g_ana) + eps))


# The prophecy: error orders of the two difference schemes, and a trap.
ORACLE_PROPHECY: dict[str, int | bool | None] = {
    # Taylor: f(x+h) = f + h f' + h^2/2 f'' + ...  ->  forward error ~ h/2 f''.
    "forward_error_order": 1,
    # The h^2 terms cancel in f(x+h) - f(x-h)  ->  central error ~ h^2/6 f'''.
    "central_error_order": 2,
    # Round-off: f(x+h) - f(x-h) loses ~16 digits of cancellation once h ~ 1e-12,
    # and dividing by 2h magnifies what is left. 1e-5 is near the sweet spot.
    "is_eps_1e-12_more_accurate_than_1e-5": False,
}
