"""TRIAL 1.2 - THE NUMERICAL ORACLE

The oracle must reproduce known gradients of any shape, bless the Altar's
gradients, catch a forgery planted here, and the prophecy about error orders
is measured against the machine.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room

oracle = load_room(__file__, "room_2_numerical_oracle")
altar = load_room(__file__, "room_1_altar_of_loss")

rng = np.random.default_rng(12)


# ----------------------------------------------------------- the two schemes
def test_forward_differences_are_close_but_only_to_first_order():
    x = rng.uniform(0.5, 2.0, size=6)
    exact = 3 * x**2
    got = oracle.forward_difference_gradient(lambda v: float(np.sum(v**3)), x, eps=1e-5)
    assert got.shape == x.shape, f"The gradient must have x's shape {x.shape}, got {got.shape}."
    err = np.linalg.norm(got - exact)
    assert err < 1e-3, f"Forward differences of sum(x^3) are off by {err:.2e}; expected ~1e-4. Check (f(x + eps e_i) - f(x)) / eps."
    assert err > 1e-7, (
        f"An error of {err:.2e} is suspiciously small for a forward difference with eps = 1e-5 "
        "(it should be ~eps/2 * f''). Did you write the central formula here?"
    )


def test_central_differences_nail_a_quadratic_of_any_shape():
    x = rng.standard_normal((2, 3, 2))
    w = rng.uniform(0.5, 3.0, size=(2, 3, 2))
    got = oracle.numerical_gradient(lambda v: float(np.sum(w * v**2)), x)
    assert got.shape == x.shape, f"The gradient must have x's shape {x.shape}, got {got.shape}."
    np.testing.assert_allclose(got, 2 * w * x, atol=1e-7, err_msg="d/dx sum(w x^2) = 2 w x. Central differences are exact to ~1e-9 here.")


def test_the_oracle_leaves_the_stones_where_they_were():
    x = rng.standard_normal((3, 3))
    before = x.copy()
    oracle.numerical_gradient(lambda v: float(np.sum(np.sin(v))), x)
    oracle.forward_difference_gradient(lambda v: float(np.sum(np.sin(v))), x)
    np.testing.assert_array_equal(x, before, err_msg="numerical_gradient modified x in place. Perturb a copy.")


def test_the_oracle_can_handle_a_single_stone():
    x = np.array(1.3)  # 0-d: a scalar parameter such as a bias
    got = oracle.numerical_gradient(lambda v: float(v**2), x)
    assert np.shape(got) == (), f"A 0-d input gives a 0-d gradient, got shape {np.shape(got)}."
    assert math.isclose(float(got), 2.6, abs_tol=1e-7)


# ------------------------------------------------------------ gradient_check
def test_gradient_check_blesses_a_correct_gradient():
    w = rng.standard_normal(5)
    x = rng.standard_normal(5)
    rel = oracle.gradient_check(lambda v: float(np.sum(w * np.sin(v))), lambda v: w * np.cos(v), x)
    assert type(rel) is float, "gradient_check returns a Python float."
    assert rel < 1e-7, f"A correct gradient should score below 1e-7 in float64, got {rel:.2e}."


def test_gradient_check_catches_the_planted_forgery():
    x = rng.standard_normal(6)
    f = lambda v: float(np.sum(v**2))  # noqa: E731
    forgot_the_two = oracle.gradient_check(f, lambda v: v, x)  # true gradient is 2v
    assert forgot_the_two > 1e-2, (
        f"A gradient missing its factor of 2 scored {forgot_the_two:.2e}. It should be ~0.33. Check the formula."
    )
    assert math.isclose(forgot_the_two, 1.0 / 3.0, rel_tol=1e-3), f"||2v - v|| / (||2v|| + ||v||) = 1/3, got {forgot_the_two:.4f}."
    sign_flip = oracle.gradient_check(f, lambda v: -2 * v, x)
    assert sign_flip > 0.9, f"A sign-flipped gradient should score ~1.0, got {sign_flip:.2e}."


def test_gradient_check_names_a_wrong_shape_instead_of_averaging_it_away():
    x = rng.standard_normal(4)
    with pytest.raises(ValueError):
        oracle.gradient_check(lambda v: float(np.sum(v)), lambda v: np.ones((4, 1)), x)


def test_gradient_check_does_not_divide_by_zero_on_flat_ground():
    x = rng.standard_normal(3)
    rel = oracle.gradient_check(lambda v: 7.0, lambda v: np.zeros_like(v), x)
    assert math.isfinite(rel) and rel == 0.0, f"Both gradients are zero: the eps guard should give exactly 0.0, got {rel!r}."


# ------------------------------------------------- the Altar meets the Oracle
def _altar_cases():
    r = np.random.default_rng(1)
    y_true = r.standard_normal((3, 4))
    y_pred = r.standard_normal((3, 4))
    y_far = y_pred - np.sign(r.standard_normal((3, 4))) * (0.1 + r.random((3, 4)))
    p = r.uniform(0.05, 0.95, size=(3, 4))
    y_bin = (r.random((3, 4)) < 0.5).astype(float)
    z = r.standard_normal((3, 4)) * 2
    logits = r.standard_normal((4, 5))
    labels = r.integers(0, 5, size=4)
    return [
        ("mse", lambda v: altar.mse(v, y_true), lambda v: altar.mse_grad(v, y_true), y_pred),
        ("mae", lambda v: altar.mae(v, y_far), lambda v: altar.mae_grad(v, y_far), y_pred),
        ("binary_cross_entropy", lambda v: altar.binary_cross_entropy(v, y_bin), lambda v: altar.binary_cross_entropy_grad(v, y_bin), p),
        ("bce_with_logits", lambda v: altar.bce_with_logits(v, y_bin), lambda v: altar.bce_with_logits_grad(v, y_bin), z),
        ("softmax_cross_entropy", lambda v: altar.softmax_cross_entropy(v, labels), lambda v: altar.softmax_cross_entropy_grad(v, labels), logits),
    ]


@pytest.mark.parametrize("name,f,grad_f,x", _altar_cases(), ids=[c[0] for c in _altar_cases()])
def test_the_altar_passes_the_oracle(name, f, grad_f, x):
    rel = oracle.gradient_check(f, grad_f, x)
    assert rel < 1e-6, f"The oracle rejects {name}_grad from Room 1.1: relative error {rel:.2e}. One of the two rooms is wrong."


# ------------------------------------------------------------- the prophecy
def _measured_order(scheme):
    """Divide eps by 10 and see how the error shrinks; log10 of that ratio is the order."""
    f, df, x = math.sin, math.cos, 0.7
    errors = []
    for eps in (1e-2, 1e-3):
        if scheme == "forward":
            est = (f(x + eps) - f(x)) / eps
        else:
            est = (f(x + eps) - f(x - eps)) / (2 * eps)
        errors.append(abs(est - df(x)))
    return round(math.log10(errors[0] / errors[1])), errors


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_the_prophecy_of_error_orders(scheme):
    key = f"{scheme}_error_order"
    prediction = oracle.ORACLE_PROPHECY.get(key)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in ORACLE_PROPHECY.")
    order, (e_big, e_small) = _measured_order(scheme)
    assert prediction == order, (
        f"You said {scheme} differences have error order {prediction}. Measured on sin(x): error {e_big:.2e} at eps=1e-2 "
        f"and {e_small:.2e} at eps=1e-3, a factor of {e_big / e_small:.0f}, so the order is {order}."
    )


def test_the_prophecy_of_the_tiny_eps():
    key = "is_eps_1e-12_more_accurate_than_1e-5"
    prediction = oracle.ORACLE_PROPHECY.get(key)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in ORACLE_PROPHECY.")
    f = lambda t: t**3  # noqa: E731
    err = {eps: abs((f(1.0 + eps) - f(1.0 - eps)) / (2 * eps) - 3.0) for eps in (1e-5, 1e-12)}
    truth = err[1e-12] < err[1e-5]
    assert prediction is truth, (
        f"You said {prediction}. Measured: central difference of x^3 at 1.0 has error {err[1e-5]:.1e} with eps=1e-5 "
        f"and {err[1e-12]:.1e} with eps=1e-12. Subtracting two nearly equal float64 numbers loses digits; "
        "dividing by 2e-12 then magnifies the rubble a trillion-fold."
    )


def test_the_prophecy_is_complete():
    assert set(oracle.ORACLE_PROPHECY) == {"forward_error_order", "central_error_order", "is_eps_1e-12_more_accurate_than_1e-5"}, (
        "Do not rename or remove the prophecy's questions; answer them."
    )
