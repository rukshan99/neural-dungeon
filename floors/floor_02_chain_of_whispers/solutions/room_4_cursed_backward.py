"""ROOM 2.4 - THE CURSED BACKWARD  (reference solution: the six curses lifted)

Spoilers below. The cursed file you are meant to fix is in ../rooms/.

The six curses, and the lines that lift them:

  1. tanh backward used (1 + t^2). The derivative of tanh is (1 - t^2).
  2. mul backward ASSIGNED grads with '='. Gradients must ACCUMULATE with '+='.
  3. backward() walked the topological order forwards (leaves first). The
     root must speak first: iterate in REVERSE.
  4. pow backward dropped the exponent: d(x^n)/dx = n * x^(n-1), not x^(n-1).
  5. exp backward multiplied by the INPUT x. d(e^x)/dx = e^x, the OUTPUT.
  6. backward() seeded the root's grad with 0.0. d(out)/d(out) is 1.0.
"""

from __future__ import annotations

import math

CURSES_FOUND: list[str] = ["tanh", "__mul__", "backward (order)", "__pow__", "exp", "backward (seed)"]


class CursedValue:
    """A scalar autograd node. The design is the one from Room 2.1."""

    def __init__(self, data: float, _children: tuple[CursedValue, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"CursedValue(data={self.data:.6g}, grad={self.grad:.6g})"

    def __add__(self, other: CursedValue | float) -> CursedValue:
        other = other if isinstance(other, CursedValue) else CursedValue(other)
        out = CursedValue(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: CursedValue | float) -> CursedValue:
        other = other if isinstance(other, CursedValue) else CursedValue(other)
        out = CursedValue(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad  # curse 2 lifted: += not =
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> CursedValue:
        out = CursedValue(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad  # curse 4 lifted

        out._backward = _backward
        return out

    def __neg__(self) -> CursedValue:
        return self * -1.0

    def __sub__(self, other: CursedValue | float) -> CursedValue:
        return self + (-other)

    def __truediv__(self, other: CursedValue | float) -> CursedValue:
        return self * other**-1.0

    def __radd__(self, other: float) -> CursedValue:
        return self + other

    def __rmul__(self, other: float) -> CursedValue:
        return self * other

    def __rsub__(self, other: float) -> CursedValue:
        return (-self) + other

    def __rtruediv__(self, other: float) -> CursedValue:
        return self**-1.0 * other

    def exp(self) -> CursedValue:
        out = CursedValue(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # curse 5 lifted: the output, not the input

        out._backward = _backward
        return out

    def log(self) -> CursedValue:
        out = CursedValue(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> CursedValue:
        t = math.tanh(self.data)
        out = CursedValue(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad  # curse 1 lifted: minus, not plus

        out._backward = _backward
        return out

    def relu(self) -> CursedValue:
        out = CursedValue(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[CursedValue] = []
        visited: set[CursedValue] = set()

        def build(v: CursedValue) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0  # curse 6 lifted: the root whispers "one"
        for v in reversed(topo):  # curse 3 lifted: root first, leaves last
            v._backward()
