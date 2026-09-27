"""ROOM 1.4 - THE PROPHECY OF CURVES

    A round chamber with a smooth stone bowl in the floor. A marble rests in
    your palm. Etched around the rim: four step sizes. Before you roll, you
    must say what each will do. The chamber remembers guesses.

No code in this room. Only predictions, checked by running the real thing.

THE ONE-DIMENSIONAL BOWL.  f(x) = 0.5 * a * x^2, with gradient a * x. One
gradient-descent step is

    x_new = x - lr * a * x = (1 - lr * a) * x.

So each step multiplies x by the constant r = 1 - lr * a. Everything about
convergence is in that one number:

    0 < r < 1    x shrinks every step and keeps its sign     "converges_smoothly"
    r = 0        x is exactly 0 after ONE step                (also smooth)
    -1 < r < 0   x shrinks but flips sign every step         "converges_oscillating"
    |r| > 1      x grows without bound                       "diverges"
    |r| = 1      x never gets closer                         (the cliff edge)

For each learning rate in CURVE_LEARNING_RATES, write one of the three
strings. Then name the lr that reaches the minimum in exactly one step, and
the largest lr that still converges (strictly speaking the supremum: at that
exact value r = -1 and the marble bounces between two walls forever).

THE TWO-DIMENSIONAL BOWL.  f(x) = 0.5 * (a_0 x_0^2 + a_1 x_1^2) with
a = (1, 25). Gradient descent treats each coordinate independently with its
own r_i = 1 - lr * a_i, and the SAME lr must work for both. That single fact
is the reason learning rates are hard to pick: the steepest direction sets
the ceiling on lr, and the flattest direction then decides how slowly you
crawl. This is what "ill-conditioned" means, and the ratio max(a)/min(a) is
the condition number. Predict which coordinate limits the step, the exact
boundary, and which coordinate is the slow one at lr = 0.07.
"""

from __future__ import annotations

# ------------------------------------------------------------ one curve
CURVE_A: float = 4.0
CURVE_LEARNING_RATES: tuple[float, ...] = (0.1, 0.25, 0.4, 0.6)

# One of "converges_smoothly", "converges_oscillating", "diverges" per lr.
CURVE_PROPHECY: dict[float, str | None] = {
    0.1: None,
    0.25: None,
    0.4: None,
    0.6: None,
}

# Which lr in CURVE_LEARNING_RATES lands exactly on the minimum in one step?
ONE_STEP_LR: float | None = None

# The largest lr (a float, not necessarily one of the four) that still converges.
LARGEST_STABLE_LR: float | None = None

# ---------------------------------------------------------- the 2-D bowl
BOWL_A: tuple[float, float] = (1.0, 25.0)

BOWL_PROPHECY: dict[str, int | float | None] = {
    # 0 or 1: the index into BOWL_A of the coordinate that caps the stable lr.
    "coordinate_that_sets_the_stable_lr": None,
    # A float: the largest lr for which both coordinates converge.
    "largest_stable_lr": None,
    # 0 or 1: at lr = 0.07, which coordinate is still furthest from 0 after many steps?
    "slowest_coordinate_at_lr_0.07": None,
}
