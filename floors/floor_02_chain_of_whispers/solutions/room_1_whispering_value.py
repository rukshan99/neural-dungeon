"""ROOM 2.1 - THE WHISPERING VALUE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A scalar autograd engine in the style of micrograd. Every operator builds a new
Value that remembers its children and a closure that knows how to pass the
gradient back to them. ``backward()`` orders the graph so every node hears the
whole whisper before it speaks, then plays the closures in reverse.
"""

from __future__ import annotations

import math


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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1. Accumulate: a node may be used twice.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        if isinstance(exponent, Value):
            raise TypeError("Value ** Value is not supported; the exponent must be a plain number")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            # d(x^n)/dx = n * x^(n-1). Do not drop the n.
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1.0

    # Reflected operators: Python calls these for  2 + v,  2 * v,  2 - v,  2 / v.
    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return (-self) + other

    def __rtruediv__(self, other: float) -> Value:
        return self**-1.0 * other

    # -------------------------------------------------------- nonlinearities
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            # d(e^x)/dx = e^x, which is the OUTPUT, already computed.
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d tanh/dx = 1 - tanh(x)^2. Never larger than 1: the whisper only shrinks.
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # The gate is open (derivative 1) for positive inputs, shut (0) otherwise.
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            # d sigmoid/dx = s (1 - s), at most 0.25. Fifty of these in a row is 1e-30.
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # -------------------------------------------------------------- backward
    def backward(self) -> None:
        """Seed d(self)/d(self) = 1 and pass gradients back through the whole graph.

        Gradients ACCUMULATE. Calling backward() twice without zeroing every
        node's ``grad`` doubles them; that is the contract, not a bug.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)  # a node is appended only after all its children

        build(self)
        self.grad = 1.0
        for v in reversed(topo):  # root first, leaves last
            v._backward()
