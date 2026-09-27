"""BOSS - THE LEARNING-RATE LICH  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Lich's valley is a quadratic with curvatures 1 and 1000. Plain gradient
descent must keep lr below 2/1000 to survive the steep wall, and at that lr
the flat direction shrinks by only 0.998 per step: about 3500 steps to reach
1e-3, and the budget is 2000. Two ways out:

  * per-coordinate step sizes (Adam), which do not care that one direction is
    a thousand times steeper than the other, or
  * momentum, which turns kappa into sqrt(kappa) in the step count,

plus a schedule so the last steps are small enough to settle. This solution
uses Adam with linear warmup and a cosine decay to zero. The same code, with
the same hyperparameters, also crosses the Rosenbrock valley.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from .room_5_momentum_chamber import adam_step


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Linear warmup from 0 to lr_max over warmup_steps, then cosine decay to lr_min.

        step <  warmup_steps :  lr_max * step / warmup_steps
        step >= total_steps  :  lr_min
        otherwise            :  progress = (step - warmup) / (total - warmup)
                                lr_min + 0.5 (lr_max - lr_min) (1 + cos(pi * progress))
    """
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def lr_range_test(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    candidate_lrs: Sequence[float],
    steps: int = 50,
) -> float:
    """The largest candidate lr for which `steps` plain GD steps from x0 do not diverge.

    A candidate diverges if any loss along its run is NaN/inf or rises above
    the starting loss. Raises ValueError if every candidate diverges.
    """
    x0 = np.asarray(x0, dtype=float)
    loss0 = loss_fn(x0)
    survivors: list[float] = []
    for lr in candidate_lrs:
        x = x0.copy()
        stable = True
        for _ in range(steps):
            x = x - lr * grad_fn(x)
            loss = loss_fn(x)
            if not np.isfinite(loss) or loss > loss0:
                stable = False
                break
        if stable:
            survivors.append(float(lr))
    if not survivors:
        raise ValueError("Every candidate learning rate diverged. Try smaller ones.")
    return max(survivors)


def slay_the_lich(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_steps: int = 2000,
) -> np.ndarray:
    """Minimize with at most max_steps gradient evaluations, on a problem you are not told about.

    Adam with warmup + cosine decay. Adam's per-coordinate normalization makes
    the first steps about lr_max in size whatever the gradient scale, and the
    decay to zero lets the iterate settle instead of jittering around the
    minimum. lr_max = 0.1 is a fine default when parameters are O(1).
    """
    x = np.array(x0, dtype=float)
    state: dict = {}
    warmup = max(1, max_steps // 20)
    for step in range(max_steps):
        g = grad_fn(x)
        if np.linalg.norm(g) < 1e-12:  # already at the bottom; save the budget
            break
        lr = cosine_with_warmup(step, max_steps, warmup, lr_max=0.1, lr_min=0.0)
        x, state = adam_step(x, g, state, {"lr": lr})
    return x
