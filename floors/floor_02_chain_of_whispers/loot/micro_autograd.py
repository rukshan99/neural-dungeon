"""micro_autograd - loot from Floor 2 of the Neural Dungeon.

A complete, dependency-light (numpy only) reverse-mode autodiff engine in two
sizes:

    Value   a scalar node, for reading and teaching (the micrograd design)
    Tensor  an n-d array node with broadcasting-aware gradients

plus ``softmax_cross_entropy`` as a fused op and ``gradient_check`` for the
habit that finds every autograd bug. Copy this file wherever you need a
gradient and do not want a framework.

Design, shared by both classes:

    * ``data``      the forward value
    * ``grad``      d(root)/d(this), accumulated with += (a node may feed many consumers)
    * ``_prev``     the nodes this one was computed from
    * ``_backward`` a closure that adds  local_derivative * self.grad  into each child
    * ``backward()`` seeds the root with 1 and calls the closures in reverse topological order

Gradients are never reset by the engine. Zero them yourself before each pass
(``p.grad = 0.0`` / ``p.grad[...] = 0``).

Run this file to execute its self-checks:  python micro_autograd.py
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

# =============================================================================
# Shared machinery
# =============================================================================


def topological_order(root) -> list:
    """Children before parents, root last. Iterative, so deep chains do not hit the recursion limit."""
    order: list = []
    visited: set = set()
    stack = [(root, iter(root._prev))]
    visited.add(root)
    while stack:
        node, children = stack[-1]
        for child in children:
            if child not in visited:
                visited.add(child)
                stack.append((child, iter(child._prev)))
                break
        else:  # every child of ``node`` has been emitted
            stack.pop()
            order.append(node)
    return order


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``: undo the copying that broadcasting did on the way forward."""
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# =============================================================================
# Value: scalar autograd
# =============================================================================


class Value:
    """A scalar that remembers how it was made."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data: float, _children: Iterable[Value] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n: float) -> Value:
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward() -> None:
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other) -> Value:
        return self + (-other)

    def __truediv__(self, other) -> Value:
        return self * other**-1.0

    def __radd__(self, other) -> Value:
        return self + other

    def __rmul__(self, other) -> Value:
        return self * other

    def __rsub__(self, other) -> Value:
        return (-self) + other

    def __rtruediv__(self, other) -> Value:
        return self**-1.0 * other

    # -- nonlinearities -------------------------------------------------------
    def _unary(self, value: float, local: float, op: str) -> Value:
        out = Value(value, (self,), op)

        def _backward() -> None:
            self.grad += local * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        return self._unary(e, e, "exp")

    def log(self) -> Value:
        return self._unary(math.log(self.data), 1.0 / self.data, "log")

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return self._unary(t, 1.0 - t * t, "tanh")

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        return self._unary(s, s * (1.0 - s), "sigmoid")

    def relu(self) -> Value:
        return self._unary(max(self.data, 0.0), 1.0 if self.data > 0.0 else 0.0, "relu")

    # -- backward -------------------------------------------------------------
    def backward(self) -> None:
        order = topological_order(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


# =============================================================================
# Tensor: n-d autograd with broadcasting
# =============================================================================


class Tensor:
    """An n-dimensional array that remembers how it was made."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children: Iterable[Tensor] = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> Tensor:  # noqa: N802
        return self.transpose()

    @staticmethod
    def _wrap(other) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    # -- elementwise ----------------------------------------------------------
    def __add__(self, other) -> Tensor:
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> Tensor:
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, n: float) -> Tensor:
        out = Tensor(self.data**n, (self,), f"**{n}")

        def _backward() -> None:
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        return self + (-self._wrap(other))

    def __truediv__(self, other) -> Tensor:
        return self * self._wrap(other) ** -1.0

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rsub__(self, other) -> Tensor:
        return (-self) + other

    def _unary(self, value: np.ndarray, local: np.ndarray, op: str) -> Tensor:
        out = Tensor(value, (self,), op)

        def _backward() -> None:
            self.grad += local * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        e = np.exp(self.data)
        return self._unary(e, e, "exp")

    def log(self) -> Tensor:
        return self._unary(np.log(self.data), 1.0 / self.data, "log")

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        return self._unary(t, 1.0 - t * t, "tanh")

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-self.data))
        return self._unary(s, s * (1.0 - s), "sigmoid")

    def relu(self) -> Tensor:
        return self._unary(np.maximum(self.data, 0.0), (self.data > 0.0).astype(np.float64), "relu")

    # -- matmul ---------------------------------------------------------------
    def __matmul__(self, other) -> Tensor:
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.shape)

        out._backward = _backward
        return out

    # -- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        return s * (s.data.size / self.data.size)

    def max(self, axis=None, keepdims: bool = False) -> Tensor:
        """Gradient flows to the (first) arg-max positions only."""
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), "max")

        def _backward() -> None:
            g = out.grad
            m = out.data
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
                m = np.expand_dims(m, axis)
            mask = (self.data == m).astype(np.float64)
            mask /= mask.sum(axis=axis, keepdims=True) if axis is not None else mask.sum()
            self.grad += mask * g

        out._backward = _backward
        return out

    # -- shape ----------------------------------------------------------------
    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        perm = tuple(axes) if axes else tuple(range(self.data.ndim))[::-1]
        inverse = np.argsort(perm)
        out = Tensor(self.data.transpose(perm), (self,), "transpose")

        def _backward() -> None:
            self.grad += out.grad.transpose(inverse)

        out._backward = _backward
        return out

    # -- backward -------------------------------------------------------------
    def backward(self) -> None:
        order = topological_order(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()


# =============================================================================
# Fused ops
# =============================================================================


def softmax_cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean cross-entropy of softmax(logits) against integer targets, as one node.

    Forward is the stable log-softmax; backward is (softmax - onehot) / N.
    """
    z = logits.data
    targets = np.asarray(targets, dtype=np.int64)
    n = z.shape[0]
    rows = np.arange(n)
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    out = Tensor(-log_probs[rows, targets].mean(), (logits,), "softmax_xent")

    def _backward() -> None:
        probs = np.exp(log_probs)
        probs[rows, targets] -= 1.0
        logits.grad += probs / n * out.grad

    out._backward = _backward
    return out


# =============================================================================
# Gradient checking
# =============================================================================


def gradient_check(f: Callable[[list[Tensor]], Tensor], inputs: list[np.ndarray], h: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7) -> bool:
    """Compare f's analytic gradients with central finite differences. Returns True if they agree.

    ``f`` maps a list of Tensors to a scalar Tensor. Every input is checked element by element.
    """
    tensors = [Tensor(x) for x in inputs]
    f(tensors).backward()
    ok = True
    for i, x in enumerate(inputs):
        numeric = np.zeros_like(x, dtype=np.float64)
        for idx in np.ndindex(*x.shape):
            old = x[idx]
            x[idx] = old + h
            up = float(f([Tensor(a) for a in inputs]).data)
            x[idx] = old - h
            down = float(f([Tensor(a) for a in inputs]).data)
            x[idx] = old
            numeric[idx] = (up - down) / (2 * h)
        if not np.allclose(tensors[i].grad, numeric, rtol=rtol, atol=atol):
            ok = False
            print(f"input {i}: analytic and numeric gradients differ by up to {np.max(np.abs(tensors[i].grad - numeric)):.3e}")
    return ok


# =============================================================================
# Self-check
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Scalar: the diamond and a chain.
    a = Value(3.0)
    (a * a + a).backward()
    assert math.isclose(a.grad, 7.0), a.grad
    x = Value(0.5)
    v = x
    for _ in range(2000):  # deeper than Python's recursion limit: the iterative topo sort copes
        v = v.tanh()
    v.backward()
    assert math.isfinite(x.grad)

    # Tensor: a two-layer MLP with a fused loss, against finite differences.
    X = rng.standard_normal((5, 3))
    W1, b1 = rng.standard_normal((3, 4)) * 0.5, rng.standard_normal(4) * 0.1
    W2, b2 = rng.standard_normal((4, 3)) * 0.5, rng.standard_normal(3) * 0.1
    y = rng.integers(0, 3, size=5)

    def mlp_loss(ts: list[Tensor]) -> Tensor:
        tX, tW1, tb1, tW2, tb2 = ts
        h = (tX @ tW1 + tb1).relu()
        return softmax_cross_entropy(h @ tW2 + tb2, y)

    assert gradient_check(mlp_loss, [X, W1, b1, W2, b2])

    # Broadcasting in every direction.
    def broadcast_soup(ts: list[Tensor]) -> Tensor:
        p, q, r = ts
        return ((p + q) * r).transpose(1, 0, 2).reshape(-1).sum()

    assert gradient_check(broadcast_soup, [rng.standard_normal((2, 3, 4)), rng.standard_normal((3, 1)), rng.standard_normal((4,))])
    print("micro_autograd: every whisper arrived intact.")
