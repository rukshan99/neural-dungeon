"""TRIAL 2.4 - THE CURSED BACKWARD

Six curses, one targeted test each. The messages describe the symptom, not the
fix. When all six pass, the whole engine is checked against finite differences.
"""

import math

import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_4_cursed_backward")
CV = room.CursedValue


def _finite_difference(f, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def _root_must_be_seeded(out) -> None:
    if out.grad == 0.0:
        pytest.fail(
            "The root of the graph has grad 0.0 after backward(). Nothing can be diagnosed until "
            "the root itself whispers; lift that curse first (see test_the_root_whispers_at_full_volume)."
        )


# ---------------------------------------------------------------- curse: seed
def test_the_root_whispers_at_full_volume():
    x = CV(1.5)
    y = x.relu()
    y.backward()
    assert y.grad == 1.0, (
        f"After backward(), the root's own grad is {y.grad}. Every downstream grad is a multiple of it, "
        f"so x.grad is {x.grad} too. The whisper is starting at zero volume."
    )


# --------------------------------------------------------------- curse: order
def test_the_whisper_reaches_the_far_end_of_a_chain():
    x = CV(0.5)
    y = x + 1.0
    z = y + 2.0
    w = z.relu()
    w.backward()
    _root_must_be_seeded(w)
    chain = [x.grad, y.grad, z.grad, w.grad]
    assert math.isclose(x.grad, 1.0), (
        f"Gradients along the chain x -> y -> z -> w are {chain}. Only + and relu are involved, so every "
        "one should be 1.0. The whisper stops one step from the root: nodes are being asked to speak "
        "before they have heard anything."
    )


# ---------------------------------------------------------- curse: mul assigns
def test_a_node_multiplied_by_itself_hears_both_whispers():
    a = CV(3.0)
    y = a * a
    y.backward()
    _root_must_be_seeded(y)
    assert math.isclose(a.grad, 6.0), (
        f"d(a*a)/da at a=3 is 2a = 6.0, but a.grad is {a.grad}. The node is BOTH operands of the product, "
        "so it should receive two whispers of 3.0. It only kept one of them."
    )


# ---------------------------------------------------------------- curse: tanh
def test_tanh_never_whispers_louder_than_one():
    x = CV(0.5)
    y = x.tanh()
    y.backward()
    _root_must_be_seeded(y)
    expected = 1.0 - math.tanh(0.5) ** 2
    assert math.isclose(x.grad, expected, rel_tol=1e-9), (
        f"tanh is a squashing function: its derivative is at most 1 and equals {expected:.4f} at x=0.5. "
        f"Yours is {x.grad:.4f}. A whisperer that makes the message LOUDER is not tanh."
    )


# ---------------------------------------------------------------- curse: pow
def test_the_power_rule_keeps_its_coefficient():
    x = CV(2.0)
    y = x**3
    y.backward()
    _root_must_be_seeded(y)
    assert math.isclose(x.grad, 12.0), (
        f"d(x**3)/dx at x=2 is 12.0; yours is {x.grad}. Notice that {x.grad} is exactly x**2: "
        "a factor that the power rule multiplies by has gone missing."
    )
    b = CV(2.0)
    q = 1.0 / b
    q.backward()
    assert math.isclose(b.grad, -0.25), (
        f"d(1/b)/db at b=2 is -1/b^2 = -0.25; yours is {b.grad}. Division goes through the same power rule, "
        "so the same missing factor flips its sign."
    )


# ---------------------------------------------------------------- curse: exp
def test_exp_whispers_its_own_output():
    x = CV(2.0)
    y = x.exp()
    y.backward()
    _root_must_be_seeded(y)
    assert math.isclose(x.grad, math.exp(2.0), rel_tol=1e-9), (
        f"d(exp(x))/dx at x=2 is e^2 = {math.exp(2.0):.4f}; yours is {x.grad:.4f}. That is the INPUT, "
        "not the derivative. The derivative of exp is the one thing exp has already computed."
    )


# ----------------------------------------------------------- the uncursed engine
def test_the_uncursed_engine_agrees_with_finite_differences():
    def f(a_val, b_val):
        return math.tanh(a_val * b_val + a_val**2) * math.exp(b_val / a_val) + math.log(a_val) - 1.0 / b_val

    a, b = CV(1.3), CV(0.8)
    out = (a * b + a**2).tanh() * (b / a).exp() + a.log() - 1.0 / b
    assert math.isclose(out.data, f(1.3, 0.8), rel_tol=1e-12), "the forward pass is wrong (the curses are in the backward)"
    out.backward()
    da = _finite_difference(lambda x: f(x, 0.8), 1.3)
    db = _finite_difference(lambda x: f(1.3, x), 0.8)
    assert math.isclose(a.grad, da, rel_tol=1e-6), f"df/da should be {da:.6f} (finite differences), got {a.grad:.6f}"
    assert math.isclose(b.grad, db, rel_tol=1e-6), f"df/db should be {db:.6f} (finite differences), got {b.grad:.6f}"


def test_the_uncursed_engine_handles_a_diamond_at_depth():
    a, b, c = CV(1.5), CV(-2.0), CV(0.5)
    d = a * b
    e = d + a * c
    out = (e * d).relu() + e
    out.backward()

    def f(av, bv, cv):
        d_ = av * bv
        e_ = d_ + av * cv
        return max(e_ * d_, 0.0) + e_

    for name, node, val in [("a", a, 1.5), ("b", b, -2.0), ("c", c, 0.5)]:
        args = {"a": 1.5, "b": -2.0, "c": 0.5}

        def g(x, name=name):
            args2 = dict(args)
            args2[name] = x
            return f(args2["a"], args2["b"], args2["c"])

        numeric = _finite_difference(g, val)
        assert math.isclose(node.grad, numeric, rel_tol=1e-6, abs_tol=1e-9), (
            f"d out / d {name} should be {numeric:.6f}, got {node.grad:.6f}"
        )
