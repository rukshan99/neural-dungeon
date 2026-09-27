"""TRIAL 2.1 - THE WHISPERING VALUE

Every whisperer is checked against finite differences. Then the diamond,
the double backward, and a chain of a hundred tanh whisperers.
"""

import math

import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_1_whispering_value")
Value = room.Value


def _finite_difference(f, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


# ------------------------------------------------------------------ basics
def test_a_value_carries_its_data_and_starts_silent():
    v = Value(3)
    assert v.data == 3.0 and isinstance(v.data, float), f"data should be the float 3.0, got {v.data!r}"
    assert v.grad == 0.0, f"a fresh Value has heard nothing yet: grad should be 0.0, got {v.grad}"


def test_an_operation_remembers_its_children():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    assert c.data == 5.0, f"2 + 3 should be 5, got {c.data}"
    assert isinstance(c._prev, set) and c._prev == {a, b}, (
        f"c._prev should be the set {{a, b}} so backward() can find its way, got {c._prev!r}"
    )
    assert isinstance(c._op, str) and c._op, "c._op should name the operation (any non-empty string)."


def test_add_and_mul_whisper_the_right_local_derivatives():
    a, b = Value(2.0), Value(-3.0)
    y = a * b + a
    y.backward()
    assert y.grad == 1.0, f"the root's own gradient is d y / d y = 1.0, got {y.grad}"
    assert math.isclose(a.grad, b.data + 1.0), (
        f"d(a*b + a)/da = b + 1 = {b.data + 1.0}, got {a.grad}. "
        "a is used twice here: once in a*b (whisper b) and once alone (whisper 1)."
    )
    assert math.isclose(b.grad, a.data), f"d(a*b + a)/db = a = {a.data}, got {b.grad}"


def test_plain_numbers_are_welcome_on_either_side():
    a = Value(4.0)
    y = 2 * a + a * 3 - 1 + (1 - a) + a / 2 + 2 / a
    # y = 2a + 3a - 1 + 1 - a + a/2 + 2/a  =  4.5a + 2/a
    expected_value = 4.5 * 4.0 + 2 / 4.0
    expected_grad = 4.5 - 2 / 16.0
    assert math.isclose(y.data, expected_value), (
        f"2a + 3a - 1 + (1 - a) + a/2 + 2/a at a=4 is {expected_value}, got {y.data}. "
        "Check the reflected operators: 2 * a calls __rmul__, 1 - a calls __rsub__, 2 / a calls __rtruediv__."
    )
    y.backward()
    assert math.isclose(a.grad, expected_grad, rel_tol=1e-9), (
        f"dy/da should be 4.5 - 2/a^2 = {expected_grad}, got {a.grad}"
    )


def test_neg_and_sub_are_built_on_the_engine():
    a = Value(1.5)
    y = -a
    assert y.data == -1.5, f"-a at a = 1.5 should be -1.5, got {y.data}"
    assert a in y._prev or any(a in p._prev for p in y._prev), (
        "-a must be part of the graph (built from Value operations), not a detached Value(-a.data)."
    )
    b = Value(0.5)
    z = a - b
    z.backward()
    assert math.isclose(a.grad, 1.0) and math.isclose(b.grad, -1.0), (
        f"d(a-b)/da = 1 and d(a-b)/db = -1; got {a.grad} and {b.grad}"
    )


# ------------------------------------------------- each whisperer, numerically
UNARY = [
    ("exp", lambda v: v.exp(), math.exp, 0.7),
    ("log", lambda v: v.log(), math.log, 2.3),
    ("tanh", lambda v: v.tanh(), math.tanh, 0.4),
    ("tanh_far_out", lambda v: v.tanh(), math.tanh, 3.0),
    ("relu_positive", lambda v: v.relu(), lambda x: max(x, 0.0), 1.7),
    ("sigmoid", lambda v: v.sigmoid(), lambda x: 1 / (1 + math.exp(-x)), -0.8),
    ("pow_3", lambda v: v**3, lambda x: x**3, 1.3),
    ("pow_half", lambda v: v**0.5, lambda x: x**0.5, 2.0),
    ("pow_minus_2", lambda v: v**-2, lambda x: x**-2, 1.5),
]


@pytest.mark.parametrize("name,op,ref,x", UNARY, ids=[u[0] for u in UNARY])
def test_each_whisperer_repeats_the_correct_local_derivative(name, op, ref, x):
    v = Value(x)
    out = op(v)
    assert math.isclose(out.data, ref(x), rel_tol=1e-12), (
        f"{name}: forward value at x={x} should be {ref(x)}, got {out.data}"
    )
    out.backward()
    numeric = _finite_difference(ref, x)
    assert math.isclose(v.grad, numeric, rel_tol=1e-5, abs_tol=1e-8), (
        f"{name}: the whisperer at x={x} should repeat the message times {numeric:.6f} "
        f"(finite differences), but repeated it times {v.grad:.6f}."
    )


def test_relu_shuts_the_gate_on_negative_input():
    v = Value(-2.0)
    out = v.relu()
    assert out.data == 0.0, f"relu(-2) is 0, got {out.data}"
    out.backward()
    assert v.grad == 0.0, f"relu passes NO gradient through a negative input; got {v.grad}"


def test_sigmoid_whispers_at_most_a_quarter():
    # The loudest sigmoid can ever whisper is 0.25, at x = 0. This is the Wraith's favourite fact.
    v = Value(0.0)
    v.sigmoid().backward()
    assert math.isclose(v.grad, 0.25), f"sigmoid'(0) = 0.25, got {v.grad}"


# ------------------------------------------------------------- graph shape
def test_the_diamond_graph_accumulates_instead_of_overwriting():
    a = Value(3.0)
    b = a * a + a  # a is heard from three places: twice inside a*a, once alone
    b.backward()
    assert math.isclose(a.grad, 2 * 3.0 + 1.0), (
        f"d(a*a + a)/da = 2a + 1 = 7.0 at a=3, got {a.grad}. "
        "When a Value is used more than once, each use whispers to it and the whispers must be "
        "ADDED (grad += ...). Assigning with = keeps only the last one."
    )


def test_a_node_used_at_two_depths_still_hears_everything():
    # f = (a*b + a) * (a*b): the node d = a*b is used directly and through e.
    def f(a_val, b_val):
        return (a_val * b_val + a_val) * (a_val * b_val)

    a, b = Value(1.3), Value(-0.7)
    d = a * b
    e = d + a
    out = e * d
    out.backward()
    da = _finite_difference(lambda x: f(x, b.data), a.data)
    db = _finite_difference(lambda x: f(a.data, x), b.data)
    assert math.isclose(a.grad, da, rel_tol=1e-6), (
        f"df/da should be {da:.6f} (finite differences), got {a.grad:.6f}. "
        "d = a*b feeds two consumers at different depths: it must hear from BOTH before it "
        "whispers to a and b. That is what the topological order guarantees."
    )
    assert math.isclose(b.grad, db, rel_tol=1e-6), f"df/db should be {db:.6f}, got {b.grad:.6f}"


def test_a_second_backward_adds_to_the_first():
    # Documented behaviour: grads are never reset by the engine. On a single-op graph a second
    # backward() exactly doubles the leaves' gradients.
    a, b = Value(2.0), Value(5.0)
    y = a * b
    y.backward()
    first = a.grad
    y.backward()
    assert math.isclose(first, 5.0), f"after one backward, d(a*b)/da = b = 5.0, got {first}"
    assert math.isclose(a.grad, 10.0), (
        f"after a second backward() without zeroing, a.grad should be 2 x 5.0 = 10.0, got {a.grad}. "
        "The engine accumulates; only a explicit grad = 0.0 clears it."
    )


def test_parameters_reused_across_fresh_graphs_accumulate_unless_zeroed():
    # The realistic version of the same bug: a training loop rebuilds the graph every step
    # from the same parameter leaves. Without zero_grad the second step's gradient is doubled.
    w, x = Value(0.8), Value(1.5)
    (w * x).tanh().backward()
    single = w.grad
    (w * x).tanh().backward()
    assert math.isclose(w.grad, 2 * single, rel_tol=1e-12), (
        f"two backward passes through fresh graphs from the same leaf: expected {2 * single:.6f}, got {w.grad:.6f}"
    )
    w.grad = 0.0
    (w * x).tanh().backward()
    assert math.isclose(w.grad, single, rel_tol=1e-12), (
        "after setting w.grad = 0.0, one backward should give the single-pass gradient again"
    )


def test_a_chain_of_a_hundred_whisperers_still_delivers():
    x = Value(0.5)
    v = x
    for _ in range(100):
        v = v.tanh()
    v.backward()
    assert math.isfinite(x.grad) and x.grad > 0.0, (
        f"a hundred tanh whisperers should still deliver a finite, positive gradient; got {x.grad}"
    )

    def chain(x0: float) -> float:
        t = x0
        for _ in range(100):
            t = math.tanh(t)
        return t

    numeric = _finite_difference(chain, 0.5, h=1e-5)
    assert math.isclose(x.grad, numeric, rel_tol=1e-3), (
        f"the gradient through 100 tanh nodes should be {numeric:.6e} (finite differences), got {x.grad:.6e}"
    )
    assert x.grad < 0.05, (
        f"a hundred derivatives each below 1 multiply to something small: {x.grad:.4f} is suspiciously loud."
    )
