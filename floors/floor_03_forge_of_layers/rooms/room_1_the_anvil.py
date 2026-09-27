"""ROOM 3.1 - THE ANVIL

    The forge is loud. On the anvil lies a bar of raw matrix multiplication,
    glowing. Every strike shapes it into a LAYER: something that takes a
    tensor in, sends a tensor out, and remembers just enough to send the
    blame back the way it came.

A layer is a function with a memory. Two methods, one contract:

    out = layer.forward(x)      compute the output; stash what backward will
                                need in self.cache
    dx  = layer.backward(dout)  dout is dLoss/dout (same shape as out). Return
                                dLoss/dx (same shape as x). If the layer has
                                parameters, ADD dLoss/dparam into self.grads.

Gradients ACCUMULATE: backward adds into self.grads rather than overwriting,
which is what every autograd framework does (a parameter used twice in one
graph collects two contributions). Call zero_grad() before each new backward.

The chain rule does all the work. For a layer out = f(x) and a loss L that
depends on out, dL/dx = dL/dout * dout/dx. You never form the full Jacobian
dout/dx; you compute its product with dout directly, which for every layer on
this floor is one or two lines of numpy.

Shapes are the whole game. For Linear with x (N, in), W (in, out), b (out,):

    out = x @ W + b                              (N, out)
    dW  = x.T @ dout                             (in, out)   <- x.T, not x
    db  = dout.sum(axis=0)                       (out,)      <- sum over the batch
    dx  = dout @ W.T                             (N, in)

If you get a shape error, you have the transpose on the wrong operand. If you
get no error and a wrong gradient, you forgot to sum db over the batch axis.
The trial checks every gradient against central finite differences.

The floor works in float64 so that finite differences are trustworthy. numpy's
default is float64; just do not cast anything to float32 in here.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class. Holds the params/grads dicts and the cache. Do not edit."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset every gradient to zeros, in place."""
        for g in self.grads.values():
            g[...] = 0.0


class Linear(Layer):
    """A fully connected layer: out = x @ W + b.

    params["W"] has shape (in_features, out_features); params["b"] has shape
    (out_features,). grads has the same keys and shapes. ``__init__`` is written
    for you: W starts as N(0, 1/in_features) (Room 2 explains the scale) and b
    starts at zero.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        self.params["W"] = rng.standard_normal((in_features, out_features)) / np.sqrt(in_features)
        self.params["b"] = np.zeros(out_features)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, in_features) -> (N, out_features). Cache what backward needs (hint: x)."""
        raise NotImplementedError("Linear.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout (N, out_features) -> dx (N, in_features).

        Also ADD dW = x.T @ dout into grads["W"] and db = dout.sum(axis=0) into grads["b"].
        """
        raise NotImplementedError("Linear.backward() is unwritten")


class ReLU(Layer):
    """out = max(x, 0). No parameters.

    The gradient passes through where x > 0 and is zero elsewhere. Cache the
    mask (x > 0), not x, and the backward is one multiplication.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ReLU.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ReLU.backward() is unwritten")


class Tanh(Layer):
    """out = tanh(x). No parameters.

    d tanh(x) / dx = 1 - tanh(x)^2, which is 1 - out^2. Cache the OUTPUT and the
    backward never needs x again.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Tanh.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Tanh.backward() is unwritten")


class SoftmaxCrossEntropy:
    """Softmax over the class axis followed by the mean negative log-likelihood, fused.

    forward(logits, y) -> float   logits (N, C) real scores, y (N,) integer class ids
                                  loss = -(1/N) * sum_n log softmax(logits[n])[y[n]]
    backward()         -> (N, C)  dloss/dlogits = (softmax(logits) - onehot(y)) / N

    Fusing the two makes the gradient trivially simple and the forward stable:
    compute log softmax as  (logits - m) - log(sum(exp(logits - m)))  with m the
    row max. Never exponentiate raw logits (exp(1000) is inf), and never take
    log of a probability that might have rounded to 0.

    The 1/N is not optional: the loss is a MEAN over the batch, so its gradient
    carries the same 1/N. A loss with no parameters still has a backward.
    """

    def __init__(self) -> None:
        self.cache = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Return a Python float. Cache the probabilities and y for backward."""
        raise NotImplementedError("SoftmaxCrossEntropy.forward() is unwritten")

    def backward(self) -> np.ndarray:
        """No argument: the loss is the top of the graph, so dloss/dloss = 1."""
        raise NotImplementedError("SoftmaxCrossEntropy.backward() is unwritten")
