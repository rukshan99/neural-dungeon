"""ROOM 1.4 - THE PROPHECY OF CURVES  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The whole room is one line of algebra. For f(x) = 0.5 * a * x^2 the gradient
is a x, so one step of gradient descent is

    x_new = x - lr * a * x = (1 - lr * a) * x.

Call r = 1 - lr * a. After k steps x_k = r^k x_0, so:

    |r| < 1  converges      r in (0, 1): every step shrinks x, same sign  -> smooth
                            r = 0:       x_1 = 0 exactly                   -> one step
                            r in (-1, 0): shrinks but flips sign each step -> oscillates
    |r| > 1  diverges       (r < -1 here: it flips sign AND grows)
    |r| = 1  never moves closer (lr = 2/a is the edge of the cliff)

For a bowl with several curvatures, every coordinate has its own r_i, and the
step size must satisfy ALL of them. The steepest coordinate sets the limit;
the flattest one sets the pace.
"""

from __future__ import annotations

# ------------------------------------------------------------ one curve
CURVE_A: float = 4.0
CURVE_LEARNING_RATES: tuple[float, ...] = (0.1, 0.25, 0.4, 0.6)

CURVE_PROPHECY: dict[float, str | None] = {
    0.1: "converges_smoothly",       # r = 1 - 0.4  =  0.6
    0.25: "converges_smoothly",      # r = 1 - 1.0  =  0.0   (lands on the minimum at once)
    0.4: "converges_oscillating",    # r = 1 - 1.6  = -0.6
    0.6: "diverges",                 # r = 1 - 2.4  = -1.4
}

ONE_STEP_LR: float | None = 0.25          # lr = 1 / a
LARGEST_STABLE_LR: float | None = 0.5     # lr = 2 / a: the supremum; at exactly 2/a, x just flips sign forever

# ---------------------------------------------------------- the 2-D bowl
BOWL_A: tuple[float, float] = (1.0, 25.0)

BOWL_PROPHECY: dict[str, int | float | None] = {
    # The stability condition lr < 2 / a_i must hold for every i; the largest a wins.
    "coordinate_that_sets_the_stable_lr": 1,
    # 2 / max(a) = 2 / 25
    "largest_stable_lr": 0.08,
    # At lr = 0.07: r_0 = 0.93 (slow crawl), r_1 = -0.75 (fast, oscillating). The
    # flat coordinate is the slow one once the steep coordinate has capped the lr.
    "slowest_coordinate_at_lr_0.07": 0,
}
