"""BOSS - THE LEARNING-RATE LICH

                    .-''''-.
                   /  _  _  \\          "Choose your step, little optimizer.
                  |  (o)(o)  |          Too large, and my east wall throws you
                  |    /\\    |          into the ceiling. Too small, and you
                   \\  \\__/  /           will still be crawling west when the
                    '-.__.-'            torches burn out. There is no number
                   /|  ||  |\\           that pleases me."
                  / |  ||  | \\
                 /  |  ||  |  \\         The Lich's valley is a quadratic whose
                    |  ||  |            curvature is 1 along one axis and 1000
                   /|  ||  |\\           along the other. Plain gradient descent
                  '-'  ''  '-'          must use lr < 2/1000 to survive, and at
                                        that lr the flat axis shrinks by 0.998
                                        per step: ~3500 steps to reach 1e-3.
                                        You have 2000.

WEAKNESS: it thinks a learning rate is one number for all time and all
directions. Adaptive per-coordinate steps (Adam) or momentum, and a schedule
that shrinks the step as you settle, cross its valley with room to spare. It
also cannot hide the fact that stability is decided by the SHARPEST direction
alone: the boundary is lr = 2 / a_max, and a range test finds it empirically.

The fight has three phases:

  Phase 1  cosine_with_warmup - the schedule every modern training run uses.
  Phase 2  lr_range_test      - probe candidate learning rates on the actual
                                problem and keep the largest that survives.
                                The trial checks you against 2 / a_max.
  Phase 3  slay_the_lich      - reach the bottom of the kappa = 1000 valley
                                within 2000 gradient evaluations, then cross
                                the Rosenbrock valley within 5000. Same code
                                for both: you are not told which one you face.

You may import your Room 1.5 optimizers (the line below does). A schedule
changes lr every step, so you will write your own loop rather than use
run_optimizer.

Run:  dungeon fight 1
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .room_5_momentum_chamber import (  # noqa: F401  (use whichever you like)
    adam_step,
    momentum_step,
)


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Learning rate at `step` (0-based) for linear warmup followed by cosine decay.

        step <  warmup_steps :  lr_max * step / warmup_steps              (so lr(0) = 0)
        step >= total_steps  :  lr_min                                    (clamped after the end)
        otherwise            :  progress = (step - warmup_steps) / (total_steps - warmup_steps)
                                lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))

    Exact consequences the trial checks: lr(warmup_steps) == lr_max; halfway
    through the decay lr == (lr_max + lr_min) / 2; lr(total_steps) == lr_min;
    warmup_steps == 0 means no warmup (lr(0) == lr_max). Return a Python float.
    """
    raise NotImplementedError("cosine_with_warmup() is unwritten")


def lr_range_test(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    candidate_lrs: Sequence[float],
    steps: int = 50,
) -> float:
    """The LARGEST candidate lr for which plain GD from x0 does not diverge in `steps` steps.

    For each candidate run x <- x - lr * grad_fn(x) for `steps` updates from a
    copy of x0. The candidate DIVERGES if any loss_fn(x) along the way is NaN or
    inf, or is larger than loss_fn(x0). Return the largest surviving candidate
    as a Python float (candidates arrive in any order). If none survives, raise
    ValueError.

    On a quadratic with largest curvature a_max, theory says the answer is the
    largest candidate below 2 / a_max. The trial checks exactly that.
    """
    raise NotImplementedError("lr_range_test() is unwritten")


def slay_the_lich(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_steps: int = 2000,
) -> np.ndarray:
    """Minimize an unknown function from x0 using at most max_steps calls to grad_fn.

    Return the final parameter array (same shape as x0; do not modify x0). The
    trial counts your grad_fn calls and demands:

      * the kappa = 1000 quadratic: ||x|| < 1e-3 within max_steps = 2000
      * Rosenbrock from (-1.5, 2.0): loss_fn(x) < 1e-2 within max_steps = 5000

    with one implementation. Adam with a warmup + cosine schedule, lr_max
    around 0.1 decaying to 0, does both with a wide margin. Momentum with an lr
    found by a range test also works. Plain GD does not, and that is the point.
    """
    raise NotImplementedError("slay_the_lich() is unwritten")
