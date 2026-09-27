"""SECRET - THE WHISPERING SADDLE

The Armijo condition must hold for every step the search returns, the step
must be the FIRST in the backtracking sequence that satisfies it, and plain
steepest descent with the search must walk the Rosenbrock valley on its own.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

ledge = load_room(__file__, "secret_whispering_saddle")

pytestmark = pytest.mark.secret

rng = np.random.default_rng(42)


def _rosenbrock():
    def f(p):
        x, y = p
        return float((1 - x) ** 2 + 100 * (y - x * x) ** 2)

    def g(p):
        x, y = p
        return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])

    return f, g


def _armijo_holds(f, grad, x, d, alpha, c):
    return f(x + alpha * d) <= f(x) + c * alpha * float(grad @ d) + 1e-15


def test_a_full_step_that_already_satisfies_armijo_is_accepted_as_is():
    f = lambda x: float(0.5 * x @ x)  # noqa: E731
    x = np.array([3.0, -4.0])
    alpha = ledge.backtracking_line_search(f, x, x, -x, alpha0=1.0, rho=0.5, c=1e-4)
    assert type(alpha) is float, "Return a Python float."
    assert alpha == 1.0, f"On 0.5||x||^2 the full step d = -grad lands exactly on the minimum and satisfies Armijo; alpha0 = 1.0 should be returned, got {alpha}."


@pytest.mark.parametrize("seed", range(5))
def test_the_returned_step_satisfies_armijo_and_is_the_first_to_do_so(seed):
    r = np.random.default_rng(seed)
    a = r.uniform(1.0, 200.0, size=4)  # a stiff bowl so alpha0 = 1 is usually too bold
    f = lambda x: float(0.5 * np.sum(a * x * x))  # noqa: E731
    x = r.standard_normal(4)
    grad = a * x
    d = -grad
    alpha = ledge.backtracking_line_search(f, grad, x, d, alpha0=1.0, rho=0.5, c=1e-4)
    assert _armijo_holds(f, grad, x, d, alpha, 1e-4), (
        f"alpha = {alpha} violates Armijo: f(x + alpha d) = {f(x + alpha * d):.6f} > f(x) + c alpha grad.d = {f(x) + 1e-4 * alpha * float(grad @ d):.6f}."
    )
    k = round(math.log(alpha) / math.log(0.5))
    assert math.isclose(alpha, 0.5**k, rel_tol=1e-12), f"alpha must be alpha0 * rho^k for an integer k, got {alpha}."
    if alpha < 1.0:
        assert not _armijo_holds(f, grad, x, d, alpha / 0.5, 1e-4), (
            f"alpha / rho = {alpha / 0.5} also satisfies Armijo, so you shrank one time too many. Return the FIRST alpha that passes."
        )


def test_an_uphill_direction_is_refused():
    f = lambda x: float(0.5 * x @ x)  # noqa: E731
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        ledge.backtracking_line_search(f, x, x, +x)  # grad . d = ||x||^2 > 0: not a descent direction


def test_rho_and_c_are_honoured():
    a = np.array([50.0, 200.0])
    f = lambda x: float(0.5 * np.sum(a * x * x))  # noqa: E731
    x = np.array([1.0, 1.0])
    grad = a * x
    alpha = ledge.backtracking_line_search(f, grad, x, -grad, alpha0=0.1, rho=0.1, c=0.3)
    k = round(math.log(alpha / 0.1) / math.log(0.1))
    assert math.isclose(alpha, 0.1 * 0.1**k, rel_tol=1e-12), f"With alpha0 = 0.1 and rho = 0.1 the candidates are 0.1, 0.01, 0.001, ...; got {alpha}."
    assert _armijo_holds(f, grad, x, -grad, alpha, 0.3)
    if alpha < 0.1:
        assert not _armijo_holds(f, grad, x, -grad, alpha / 0.1, 0.3), "The previous candidate also passed with c = 0.3: you shrank too far."


def test_steepest_descent_with_the_search_walks_down_a_bowl_monotonically():
    a = np.array([1.0, 30.0, 300.0])
    f = lambda x: float(0.5 * np.sum(a * x * x))  # noqa: E731
    g = lambda x: a * x  # noqa: E731
    x0 = np.array([1.0, 1.0, 1.0])
    x, history = ledge.gd_with_line_search(f, g, x0, max_steps=5000, tol=1e-9)
    np.testing.assert_array_equal(x0, [1.0, 1.0, 1.0], err_msg="gd_with_line_search modified x0.")
    assert history[0] == f(x0), "history[0] is f(x0)."
    assert np.all(np.diff(history) <= 1e-15), "Every accepted Armijo step decreases f, so history must be non-increasing."
    assert np.linalg.norm(x) < 1e-6, f"Should reach the bottom of the bowl; ended at distance {np.linalg.norm(x):.2e}."
    assert len(history) < 5001, "With tol = 1e-9 the loop should stop early once the gradient is tiny, not run all 5000 steps."


def test_the_banana_valley_yields_to_a_walker_who_asks_the_slope():
    f, g = _rosenbrock()
    x, history = ledge.gd_with_line_search(f, g, np.array([-1.5, 2.0]), max_steps=10_000, tol=1e-8)
    assert np.all(np.diff(history) <= 1e-15), "history must be non-increasing on Rosenbrock too: that is what the line search buys you."
    assert f(x) < 1e-4, (
        f"After {len(history) - 1} steps the loss is {f(x):.3e} at {np.round(x, 4).tolist()}; expected < 1e-4 (the minimum is at (1, 1)). "
        "Steepest descent with backtracking reaches ~1e-7 in 10,000 steps here."
    )
