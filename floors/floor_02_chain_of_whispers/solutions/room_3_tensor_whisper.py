"""ROOM 2.3 - THE TENSOR WHISPER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The same design as the scalar Value, one level up: ``data`` and ``grad`` are
numpy arrays, and every backward closure has to undo broadcasting before it
adds to a child's gradient. ``unbroadcast`` is the whole trick.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing what broadcasting did on the way forward.

    Broadcasting a (4,) up to (3, 4) COPIES the vector into 3 rows. Each copy
    receives its own gradient, so the gradient of the original is the SUM over
    the 3 rows. Two kinds of stretching must be undone:

      1. Leading axes that broadcasting inserted: sum them away entirely.
      2. Axes that were 1 in ``shape`` but larger in ``grad``: sum with keepdims.
    """
    grad = np.asarray(grad)
    shape = tuple(shape)
    # 1. Sum away the extra leading axes.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. Sum over the axes that were stretched from 1.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An n-dimensional array that remembers how it was made."""

    def __init__(self, data, _children: tuple[Tensor, ...] = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> Tensor:  # noqa: N802 - mirrors numpy's .T
        return self.transpose()

    @staticmethod
    def _wrap(other) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    # --------------------------------------------------------- elementwise
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

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise TypeError("Tensor ** Tensor is not supported; the exponent must be a plain number")
        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

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

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------ matmul
    def __matmul__(self, other) -> Tensor:
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            # C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC.
            # swapaxes (not .T) so batched leading dims also work; unbroadcast
            # handles a B that was shared across a batch.
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.shape)

        out._backward = _backward
        return out

    # --------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the reduced axes back as 1s
            # Every input element contributed with weight 1, so each gets the
            # gradient of the sum it landed in: broadcast back to the input shape.
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size / max(s.data.size, 1)  # how many elements each mean averaged
        return s * (1.0 / count)

    # -------------------------------------------------------------- shape
    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        perm = tuple(axes) if axes else tuple(range(self.data.ndim))[::-1]
        out = Tensor(self.data.transpose(perm), (self,), "transpose")
        inverse = np.argsort(perm)  # the permutation that puts the axes back

        def _backward() -> None:
            self.grad += out.grad.transpose(inverse)

        out._backward = _backward
        return out

    # ----------------------------------------------------------- backward
    def backward(self) -> None:
        """Seed the root with ones and pass gradients back through the graph.

        For a scalar output the seed is 1.0 (d out / d out). For a non-scalar
        output the seed is all ones, which computes d(sum(out))/d(everything).
        Gradients accumulate across calls, exactly as in Room 2.1.
        """
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
