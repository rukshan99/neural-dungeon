"""ROOM 2.2 - THE NEURON'S WHISPER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Neuron, Layer and MLP are nothing but Values wired together. The autograd
engine from Room 2.1 does every derivative; this file only does bookkeeping
(which Values are parameters) and the training loop.
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
        """Set every parameter's grad to 0.0. Gradients accumulate; do this before each backward()."""
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """out = tanh(w . x + b), or just w . x + b when nonlin=False."""

    def __init__(self, n_in: int, rng: np.random.Generator, nonlin: bool = True) -> None:
        # Draw the weights in order so that a seed reproduces the network exactly.
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value | float]) -> Value:
        # sum() with a Value start so every + goes through the engine.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """n_out Neurons that all read the same n_in inputs."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, nonlin: bool = True) -> None:
        self.neurons = [Neuron(n_in, rng, nonlin) for _ in range(n_out)]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Layers of sizes n_in -> n_outs[0] -> ... -> n_outs[-1]. The last layer is linear."""

    def __init__(self, n_in: int, n_outs: list[int], rng: np.random.Generator) -> None:
        sizes = [n_in] + list(n_outs)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng, nonlin=(i != len(n_outs) - 1))
            for i in range(len(n_outs))
        ]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(preds: list[Value], targets: list[float]) -> Value:
    """Mean of (pred - target)^2 over the list, as a Value (so it can be differentiated)."""
    total = sum(((p - t) ** 2 for p, t in zip(preds, targets)), start=Value(0.0))
    return total * (1.0 / len(preds))


def train_xor(steps: int = 300, lr: float = 0.1, seed: int = 0) -> tuple[MLP, list[float]]:
    """Train MLP(2, [8, 1]) on XOR with plain SGD. Returns (model, loss per step)."""
    rng = np.random.default_rng(seed)
    model = MLP(2, [8, 1], rng)
    losses: list[float] = []
    for _ in range(steps):
        preds = [model(x)[0] for x in XOR_INPUTS]
        loss = mse_loss(preds, XOR_TARGETS)
        model.zero_grad()  # the whispers from the last step must not linger
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad  # walk against the gradient
        losses.append(loss.data)
    return model, losses
