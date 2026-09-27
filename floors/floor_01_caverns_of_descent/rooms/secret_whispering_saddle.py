"""SECRET - THE WHISPERING SADDLE   (optional)

    A side ledge shaped like a saddle. When you lean into the slope, the rock
    whispers a number back: how far you may step before the ground stops
    dropping as fast as it promised. Take it. Then ask again.

Every room so far needed a learning rate chosen by a human. A *line search*
asks the function instead. Given a descent direction d at x (steepest
descent uses d = -grad), try a step alpha and accept it only if the loss fell
by at least a fraction c of what the local slope predicted:

    f(x + alpha d)  <=  f(x) + c * alpha * (grad . d)          (Armijo condition)

grad . d is negative for a descent direction, so the right-hand side sits
below f(x). If the condition fails, shrink alpha by rho and try again. c is
small (1e-4 is the textbook value): you only demand a *sliver* of the promised
decrease, so a step is rarely rejected without reason. This is called
backtracking, and it makes plain steepest descent converge monotonically on
Rosenbrock's banana valley with nobody choosing a learning rate at all.

The cost is extra loss evaluations per step, which is why deep learning
mostly does not do this (a loss evaluation is a full forward pass over a
batch), and why schedules and adaptive optimizers exist instead.
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
    """Return the first alpha in alpha0, alpha0*rho, alpha0*rho^2, ... that satisfies Armijo.

    `grad` is the gradient AT x (an array), `direction` the proposed step
    direction. If grad . direction >= 0 the direction is not a descent direction:
    raise ValueError. Give up with RuntimeError after ~100 shrinks (it means f or
    grad is wrong). Return a Python float.
    """
    raise NotImplementedError("backtracking_line_search() is unwritten")


def gd_with_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_steps: int = 10_000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """Steepest descent with backtracking: direction -grad, alpha from the line search each step.

    Stop when ||grad_f(x)|| < tol or after max_steps updates. Return (x, history)
    where history[0] = f(x0) and history[k] = f after k updates. Because every
    accepted step satisfies Armijo, history is non-increasing. Do not modify x0.
    Use the line search's default alpha0, rho, c.
    """
    raise NotImplementedError("gd_with_line_search() is unwritten")
