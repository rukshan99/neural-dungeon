"""ROOM 2.3 - THE TENSOR WHISPER

    The passage widens into a hall where the whisperers stand in grids.
    A message spoken once to a row is heard by every column; when it is
    whispered back, every column answers and the row hears all of them at
    once. Summed.

The same autograd design as Room 2.1, with ``numpy`` arrays for ``data`` and
``grad``. Every op's backward is the scalar rule applied elementwise, plus ONE
new idea: broadcasting must be undone on the way back.

    Forward:  (3, 4) + (4,)  ->  the (4,) vector is copied into 3 rows.
    Backward: each of the 3 copies gets its own gradient row, so the gradient
              of the original (4,) is the SUM over those rows.

That is ``unbroadcast(grad, shape)``: sum ``grad`` down to ``shape`` by summing
away leading axes that broadcasting inserted, then summing (with keepdims)
over axes that were 1 in ``shape`` but stretched in ``grad``. Every binary op
calls it before ``+=``-ing into a child's grad, so ``grad.shape == data.shape``
always holds. The trial checks that invariant on every op.

Other rules, unchanged from Room 2.1: accumulate with ``+=``, topological
order, seed the root. The seed here is ``np.ones_like(data)`` (1.0 for a scalar
loss; for a non-scalar root this computes the gradient of ``sum(root)``).
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing what broadcasting did on the way forward.

    unbroadcast(ones((3, 4)), (4,))    -> [3, 3, 3, 3]           (leading axis summed away)
    unbroadcast(ones((3, 4)), (3, 1))  -> [[4], [4], [4]]        (stretched axis summed, kept as 1)
    unbroadcast(ones((2, 3, 4)), (3, 1)) -> [[8], [8], [8]]      (both at once)
    unbroadcast(g, g.shape)            -> g                       (nothing to undo)
    unbroadcast(ones(5), ())           -> 5.0 as a 0-d array      (a scalar was broadcast)
    """
    raise NotImplementedError("unbroadcast() is unwritten")


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
        """Plain numbers and arrays become Tensors so every op has two Tensor children."""
        return other if isinstance(other, Tensor) else Tensor(other)

    # --------------------------------------------------------- elementwise
    def __add__(self, other) -> Tensor:
        """Elementwise sum with numpy broadcasting. Backward: unbroadcast(out.grad, child.shape) to each child."""
        raise NotImplementedError("Tensor.__add__() is unwritten")

    def __mul__(self, other) -> Tensor:
        """Elementwise product with broadcasting. d(a*b)/da = b, then unbroadcast to a's shape."""
        raise NotImplementedError("Tensor.__mul__() is unwritten")

    def __pow__(self, exponent: float) -> Tensor:
        """Elementwise x ** n for a plain number n. Backward: n * x^(n-1) * out.grad (no broadcasting)."""
        raise NotImplementedError("Tensor.__pow__() is unwritten")

    def __neg__(self) -> Tensor:
        """-self, in terms of __mul__."""
        raise NotImplementedError("Tensor.__neg__() is unwritten")

    def __sub__(self, other) -> Tensor:
        """self - other, in terms of __add__ and __neg__ (wrap ``other`` first)."""
        raise NotImplementedError("Tensor.__sub__() is unwritten")

    # Given: these only combine the ops above.
    def __truediv__(self, other) -> Tensor:
        return self * self._wrap(other) ** -1.0

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rsub__(self, other) -> Tensor:
        return (-self) + other

    def exp(self) -> Tensor:
        """Elementwise e^x. Backward: out.data * out.grad."""
        raise NotImplementedError("Tensor.exp() is unwritten")

    def log(self) -> Tensor:
        """Elementwise natural log. Backward: out.grad / x."""
        raise NotImplementedError("Tensor.log() is unwritten")

    def relu(self) -> Tensor:
        """Elementwise max(x, 0). Backward: out.grad where x > 0, else 0."""
        raise NotImplementedError("Tensor.relu() is unwritten")

    # ------------------------------------------------------------ matmul
    def __matmul__(self, other) -> Tensor:
        """Matrix product for 2-D operands: (n, k) @ (k, m) -> (n, m).

        With C = A @ B and upstream gradient dC (shape of C):
            dA = dC @ B.T        (n, m) @ (m, k) -> (n, k)
            dB = A.T @ dC        (k, n) @ (n, m) -> (k, m)
        Check the shapes: they are the only way to get the order right.
        """
        raise NotImplementedError("Tensor.__matmul__() is unwritten")

    # --------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum over ``axis`` (None = everything), like ``np.sum``.

        Backward: every input element contributed with weight 1, so each gets the
        gradient of the sum it landed in. If the axis was removed (keepdims=False),
        put it back with ``np.expand_dims(out.grad, axis)`` first, then
        ``np.broadcast_to(..., self.shape)``.
        """
        raise NotImplementedError("Tensor.sum() is unwritten")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Mean over ``axis``. Express it as ``sum`` times ``1 / count`` and get the backward for free.

        ``count`` is the number of input elements that were averaged into each output element:
        ``self.data.size / out.data.size``.
        """
        raise NotImplementedError("Tensor.mean() is unwritten")

    # -------------------------------------------------------------- shape
    def reshape(self, *shape: int) -> Tensor:
        """Same elements, new shape. Backward: reshape out.grad back to self.shape.

        Accept both ``t.reshape(3, 4)`` and ``t.reshape((3, 4))``.
        """
        raise NotImplementedError("Tensor.reshape() is unwritten")

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes like ``np.transpose``. No axes = reverse them all.

        Backward: apply the INVERSE permutation to out.grad (``np.argsort(axes)``).
        Accept both ``t.transpose(2, 0, 1)`` and ``t.transpose((2, 0, 1))``.
        """
        raise NotImplementedError("Tensor.transpose() is unwritten")

    # ----------------------------------------------------------- backward
    def backward(self) -> None:
        """Seed ``self.grad = np.ones_like(self.data)`` and walk the graph in reverse topological order.

        Same algorithm as Value.backward() in Room 2.1. Grads accumulate.
        """
        raise NotImplementedError("Tensor.backward() is unwritten")
