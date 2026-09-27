"""SECRET - THE WHISPERING SADDLE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A line search replaces the learning rate with a question asked at every step:
"did this step drop the loss by at least a fraction c of what the slope
promised?" If not, shrink and ask again. That question is the Armijo condition.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad: np.ndarray,
    x: np.ndarray,
    direction: np.ndarray,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> float:
    """Largest alpha in {alpha0, alpha0 rho, alpha0 rho^2, ...} satisfying Armijo:

        f(x + alpha d) <= f(x) + c * alpha * grad . d

    grad is the gradient at x (an array, not a function). d must be a descent
    direction (grad . d < 0), otherwise ValueError.
    """
    slope = float(np.dot(np.ravel(grad), np.ravel(direction)))
    if slope >= 0.0:
        raise ValueError(f"Not a descent direction: grad . d = {slope:.3e} >= 0")
    fx = f(x)
    alpha = float(alpha0)
    for _ in range(100):
        if f(x + alpha * direction) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    raise RuntimeError("The Armijo condition was never satisfied. Is f smooth, and is grad really its gradient?")


def gd_with_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_steps: int = 10_000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """Steepest descent, d = -grad, step length chosen by backtracking each time.

    Stops when ||grad|| < tol or after max_steps. Returns (x, history of f values)
    with history[0] = f(x0), so the history is non-increasing.
    """
    x = np.array(x0, dtype=float)
    history = [float(f(x))]
    for _ in range(max_steps):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        alpha = backtracking_line_search(f, g, x, -g)
        x = x - alpha * g
        history.append(float(f(x)))
    return x, history
