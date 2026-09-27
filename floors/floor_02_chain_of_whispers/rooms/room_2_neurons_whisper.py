"""ROOM 2.2 - THE NEURON'S WHISPER

    Deeper in, the whisperers stand in rows. Each one listens to every voice
    in the row before, weighs what it hears, adds a little of its own, and
    passes a softened version on. Rows of them can learn to say anything.
    Today they learn XOR.

A neuron is  tanh(w . x + b). A layer is several neurons reading the same
input. An MLP is layers in sequence. Built from Values, every one of these is
differentiable for free: the engine from Room 2.1 does all the calculus, this
room only does bookkeeping.

    forward   preds = model(x)              (builds a fresh graph each call)
    loss      loss = mse_loss(preds, ys)    (one Value at the root)
    zero      model.zero_grad()             (grads ACCUMULATE; clear last step's)
    backward  loss.backward()               (every parameter now has .grad)
    step      p.data -= lr * p.grad         (walk downhill)

The order matters. Forget ``zero_grad`` and every step's gradient is added to
the last one's, which is the most common autograd bug in the wild. Room 2.1
warned you; here it bites.

Reproducibility contract (the trial checks it): a Neuron draws its weights
from ``rng.uniform(-1.0, 1.0)`` one at a time in order, and its bias starts at
0.0. Two networks built from generators with the same seed must be identical.
"""

from __future__ import annotations

import numpy as np

from .room_1_whispering_value import Value

# The four-point XOR problem. Targets are -1/+1 so a tanh network is comfortable.
XOR_INPUTS: list[list[float]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_TARGETS: list[float] = [-1.0, 1.0, 1.0, -1.0]


class Module:
    """Anything that owns parameters."""

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        """Set ``grad = 0.0`` on every Value returned by ``self.parameters()``."""
        raise NotImplementedError("Module.zero_grad() is unwritten")


class Neuron(Module):
    """out = tanh(w . x + b), or just w . x + b when ``nonlin`` is False."""

    def __init__(self, n_in: int, rng: np.random.Generator, nonlin: bool = True) -> None:
        """Create ``n_in`` weight Values drawn from rng.uniform(-1, 1) in order, and a bias Value(0.0).

        Store them as ``self.w`` (list of Value), ``self.b`` (Value) and ``self.nonlin``.
        """
        raise NotImplementedError("Neuron.__init__() is unwritten")

    def __call__(self, x: list[Value | float]) -> Value:
        """Weighted sum of the inputs plus the bias, through tanh if ``self.nonlin``.

        ``x`` is a list of Values or plain floats, one per weight. Return ONE Value.
        Every + and * must go through the engine (so ``sum`` needs a Value start).
        """
        raise NotImplementedError("Neuron.__call__() is unwritten")

    def parameters(self) -> list[Value]:
        """The weights followed by the bias: ``n_in + 1`` Values."""
        raise NotImplementedError("Neuron.parameters() is unwritten")


class Layer(Module):
    """``n_out`` Neurons that all read the same ``n_in`` inputs."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, nonlin: bool = True) -> None:
        """Build ``self.neurons``: a list of ``n_out`` Neurons, created in order from ``rng``."""
        raise NotImplementedError("Layer.__init__() is unwritten")

    def __call__(self, x: list[Value | float]) -> list[Value]:
        """One output Value per neuron, as a list."""
        raise NotImplementedError("Layer.__call__() is unwritten")

    def parameters(self) -> list[Value]:
        """Every neuron's parameters, concatenated in neuron order."""
        raise NotImplementedError("Layer.parameters() is unwritten")


class MLP(Module):
    """Layers of sizes n_in -> n_outs[0] -> ... -> n_outs[-1]. The LAST layer is linear (no tanh)."""

    def __init__(self, n_in: int, n_outs: list[int], rng: np.random.Generator) -> None:
        """Build ``self.layers``. MLP(2, [8, 1]) has a tanh Layer(2, 8) then a linear Layer(8, 1)."""
        raise NotImplementedError("MLP.__init__() is unwritten")

    def __call__(self, x: list[Value | float]) -> list[Value]:
        """Feed ``x`` through every layer in turn. Returns the last layer's list of Values."""
        raise NotImplementedError("MLP.__call__() is unwritten")

    def parameters(self) -> list[Value]:
        """Every layer's parameters, concatenated in layer order."""
        raise NotImplementedError("MLP.parameters() is unwritten")


def mse_loss(preds: list[Value], targets: list[float]) -> Value:
    """Mean of (pred - target)^2 over the lists, as a single Value.

    Build it out of Value operations so that ``loss.backward()`` reaches the parameters.
    """
    raise NotImplementedError("mse_loss() is unwritten")


def train_xor(steps: int = 200, lr: float = 0.1, seed: int = 0) -> tuple[MLP, list[float]]:
    """Train ``MLP(2, [8, 1])`` on XOR_INPUTS / XOR_TARGETS with plain SGD.

    Each step: forward all four points, ``mse_loss``, ``zero_grad``, ``backward``,
    then ``p.data -= lr * p.grad`` for every parameter. Record ``loss.data`` per
    step. Returns ``(model, losses)`` with ``len(losses) == steps``.

    Use ``np.random.default_rng(seed)`` for the model's initialization.
    """
    raise NotImplementedError("train_xor() is unwritten")
