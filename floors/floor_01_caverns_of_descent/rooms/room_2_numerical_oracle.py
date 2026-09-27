"""ROOM 1.2 - THE NUMERICAL ORACLE

    In an alcove sits a blind oracle. It cannot see your gradient formula.
    It can only nudge each stone a hair to the left, a hair to the right,
    and feel how the floor tilts. It is slow. It is also never wrong.

Every hand-derived gradient on this floor, and every backward pass on the
floors below, gets checked against finite differences before it is trusted.
The derivative definition, made computable:

    forward difference   (f(x + h) - f(x)) / h              error ~ (h/2)   f''
    central difference   (f(x + h) - f(x - h)) / (2h)       error ~ (h^2/6) f'''

Both come from Taylor expanding f(x +- h). In the central formula the even
powers of h cancel, which is why it is one order more accurate for free.

But h cannot shrink forever. f(x + h) and f(x - h) are two nearly equal
float64 numbers; subtracting them loses about 16 - log10(1/h) significant
digits, and dividing by 2h magnifies the rubble. In float64 the sweet spot
for central differences is h around 1e-5 to 1e-6. The prophecy at the bottom
asks you to commit to these facts before the trial measures them.

The oracle's verdict is a *relative* error, so it means the same thing for
a gradient of size 1e-3 and one of size 1e3:

    rel = ||g_num - g_ana|| / (||g_num|| + ||g_ana|| + eps)

Below 1e-6: correct. Around 1e-4: probably a kink (relu, abs) or a loose
eps. Above 1e-2: a bug. About 1.0: a sign error or a wrong shape.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def forward_difference_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Gradient of the scalar function f at x by FORWARD differences.

    out[i] = (f(x + eps * e_i) - f(x)) / eps for every element i, where e_i is
    the array that is 1 at index i and 0 elsewhere. x may have any shape and the
    result has that same shape. Do not modify x: work on copies.
    (np.ndindex(x.shape) walks every multi-index of an array.)
    """
    raise NotImplementedError("forward_difference_gradient() is unwritten")


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Gradient of the scalar function f at x by CENTRAL differences.

    out[i] = (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps). Any shape, same
    shape out, x untouched. Two evaluations of f per element; slow and proud.
    """
    raise NotImplementedError("numerical_gradient() is unwritten")


def gradient_check(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Relative error between the numerical gradient of f and the analytic grad_f at x.

        ||g_num - g_ana|| / (||g_num|| + ||g_ana|| + eps)

    Norms are Euclidean over all elements (np.linalg.norm on the flattened
    difference). The eps in the denominator only guards against 0 / 0 when both
    gradients vanish. Raise ValueError if grad_f(x) does not have x's shape:
    a wrong shape is a bug the oracle should name, not average away.
    Return a Python float.
    """
    raise NotImplementedError("gradient_check() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY
# Fill in every None BEFORE running the trial. The trial measures each claim
# on f(x) = sin(x) and f(x) = x^3 in float64 and compares with your answer.
#
#   forward_error_order / central_error_order : an int k meaning "the error
#       shrinks like eps**k". Concretely: when eps is divided by 10, by roughly
#       what factor does the error shrink? 10 -> order 1, 100 -> order 2.
#   is_eps_1e-12_more_accurate_than_1e-5 : True or False. For the central
#       difference of x**3 at x = 1.0, is eps = 1e-12 closer to the true
#       derivative (3.0) than eps = 1e-5 is?
# ---------------------------------------------------------------------------
ORACLE_PROPHECY: dict[str, int | bool | None] = {
    "forward_error_order": None,
    "central_error_order": None,
    "is_eps_1e-12_more_accurate_than_1e-5": None,
}
