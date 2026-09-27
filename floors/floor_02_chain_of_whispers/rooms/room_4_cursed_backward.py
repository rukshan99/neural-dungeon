"""ROOM 2.4 - THE CURSED BACKWARD

    Someone built a chain of whisperers before you and left it running. It
    looks right. It even gives plausible numbers. But six of the whisperers
    have been cursed: each passes on a message that is *almost* what it heard.

This is not a stub room. The engine below is COMPLETE and it is WRONG in
exactly six places. Every function has a bug or does not; some bugs live in a
local derivative, some in the machinery of ``backward()`` itself. Your job is
to find and fix all six. Do not rewrite the file from Room 2.1: read this one,
run the trial, follow the symptoms.

The trial has one targeted test per curse. Each failure message describes
what the symptom looks like (a gradient that is too big, a whisper that never
arrives, a node that hears only one of two messages) but not the fix. When all
six pass, an integration test compares the whole engine against finite
differences.

Skills this room trains: reading autograd code critically, and the habit of
checking gradients numerically, which is how every one of these bugs is found
in real code.

As you go, list what you fixed in CURSES_FOUND (for your own record; it is not
graded).
"""

from __future__ import annotations

import math

CURSES_FOUND: list[str] = []


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
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> CursedValue:
        out = CursedValue(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += self.data ** (exponent - 1) * out.grad

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
            self.grad += self.data * out.grad

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
            self.grad += (1.0 + t * t) * out.grad

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
        self.grad = 0.0
        for v in topo:
            v._backward()
