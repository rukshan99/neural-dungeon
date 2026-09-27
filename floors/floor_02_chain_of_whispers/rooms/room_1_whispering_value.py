"""ROOM 2.1 - THE WHISPERING VALUE

    The cavern is long and lightless. Whatever is said at the far end comes
    back to you along a chain of whisperers, one per step. Each whisperer
    repeats the message multiplied by their own local derivative. Nothing
    else. That is the whole of backpropagation.

This room builds a scalar autograd engine (in the style of micrograd). A
``Value`` holds a number (``.data``), the gradient of the final output with
respect to it (``.grad``), the Values it was computed from (``._prev``), a name
for the operation (``._op``) and a closure (``._backward``) that passes its
gradient one step back to its children.

The chain rule, in the exact shape this engine uses it:

    out = f(a, b)                    forward:  compute out.data, remember (a, b)
    a.grad += df/da * out.grad       backward: local derivative x upstream gradient
    b.grad += df/db * out.grad

Two rules the trial enforces, and the rest of the dungeon depends on:

  1. ACCUMULATE with ``+=``. If a Value is used twice (``b = a*a + a``) it gets
     a whisper from each use and they must be ADDED. ``=`` silently drops one.
  2. ORDER matters. A node must have heard its *whole* upstream gradient before
     it whispers to its children. ``backward()`` builds a topological order
     (depth-first, children appended before their parent) and then walks it in
     REVERSE, starting from the root with grad = 1.0.

A consequence of rule 1: this engine never resets a grad by itself. Calling
``backward()`` again adds to whatever is already there. Room 2.2 gives you
``zero_grad()`` for exactly that reason.

Every operator below follows one pattern:

    other = other if isinstance(other, Value) else Value(other)   # accept floats
    out = Value(<forward value>, (self, other), "<op>")
    def _backward():
        self.grad += <d out / d self> * out.grad
        other.grad += <d out / d other> * out.grad
    out._backward = _backward
    return out
"""

from __future__ import annotations

import math  # noqa: F401 - you will want math.exp, math.log, math.tanh


class Value:
    """A scalar that remembers how it was made, so it can be differentiated."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # leaves have nothing to pass back
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other: Value | float) -> Value:
        """self + other. ``other`` may be a Value or a plain number.

        Local derivatives: d(a+b)/da = 1, d(a+b)/db = 1.
        """
        raise NotImplementedError("Value.__add__() is unwritten")

    def __mul__(self, other: Value | float) -> Value:
        """self * other. Local derivatives: d(a*b)/da = b, d(a*b)/db = a."""
        raise NotImplementedError("Value.__mul__() is unwritten")

    def __pow__(self, exponent: float) -> Value:
        """self ** exponent, for a plain int/float exponent (not a Value).

        Local derivative: d(x^n)/dx = n * x^(n-1). Keep the n.
        """
        raise NotImplementedError("Value.__pow__() is unwritten")

    def __neg__(self) -> Value:
        """-self. One line, in terms of __mul__."""
        raise NotImplementedError("Value.__neg__() is unwritten")

    def __sub__(self, other: Value | float) -> Value:
        """self - other. One line, in terms of __add__ and __neg__."""
        raise NotImplementedError("Value.__sub__() is unwritten")

    def __truediv__(self, other: Value | float) -> Value:
        """self / other. One line: a / b == a * b**-1."""
        raise NotImplementedError("Value.__truediv__() is unwritten")

    # Reflected operators. Python calls these when the LEFT operand is a plain
    # number:  2 + v -> v.__radd__(2),  2 * v,  2 - v,  2 / v.
    def __radd__(self, other: float) -> Value:
        raise NotImplementedError("Value.__radd__() is unwritten")

    def __rmul__(self, other: float) -> Value:
        raise NotImplementedError("Value.__rmul__() is unwritten")

    def __rsub__(self, other: float) -> Value:
        """other - self."""
        raise NotImplementedError("Value.__rsub__() is unwritten")

    def __rtruediv__(self, other: float) -> Value:
        """other / self."""
        raise NotImplementedError("Value.__rtruediv__() is unwritten")

    # -------------------------------------------------------- nonlinearities
    def exp(self) -> Value:
        """e ** self. Local derivative: e^x, which is the OUTPUT you just computed."""
        raise NotImplementedError("Value.exp() is unwritten")

    def log(self) -> Value:
        """Natural log. Local derivative: 1 / x."""
        raise NotImplementedError("Value.log() is unwritten")

    def tanh(self) -> Value:
        """Hyperbolic tangent. Local derivative: 1 - tanh(x)^2, never above 1."""
        raise NotImplementedError("Value.tanh() is unwritten")

    def relu(self) -> Value:
        """max(x, 0). Local derivative: 1 if x > 0 else 0 (use 0 at exactly 0)."""
        raise NotImplementedError("Value.relu() is unwritten")

    def sigmoid(self) -> Value:
        """1 / (1 + e^-x). Local derivative: s * (1 - s) where s is the output. At most 0.25."""
        raise NotImplementedError("Value.sigmoid() is unwritten")

    # -------------------------------------------------------------- backward
    def backward(self) -> None:
        """Set self.grad = 1.0, then call every node's _backward in reverse topological order.

        Build the order with a depth-first walk over ``_prev``: visit each node
        once, append a node only AFTER all of its children have been appended.
        Then iterate the list in reverse so the root speaks first and the
        leaves last.

        Do not zero anything here. Grads accumulate; that is the contract.
        """
        raise NotImplementedError("Value.backward() is unwritten")
