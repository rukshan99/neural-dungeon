"""ROOM 3.3 - THE BELLOWS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The training loop is the same five lines on every floor of the dungeon:
zero the grads, forward, loss, backward, step. Everything else is bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .room_1_the_anvil import SoftmaxCrossEntropy


class Sequential:
    """A stack of layers applied in order. Backward runs the stack in reverse."""

    def __init__(self, layers: list) -> None:
        self.layers = list(layers)
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> dict[str, np.ndarray]:
        """Flat view: {"0.W": ..., "0.b": ..., "2.W": ...}. The arrays are the layers' own."""
        return {f"{i}.{name}": p for i, layer in enumerate(self.layers) for name, p in layer.params.items()}

    def grads(self) -> dict[str, np.ndarray]:
        """Same keys as params(), holding the gradients from the latest backward."""
        return {f"{i}.{name}": g for i, layer in enumerate(self.layers) for name, g in layer.grads.items()}

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def train(self) -> None:
        """Training mode: layers that behave differently at train time (Dropout, BatchNorm) switch on."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False


class SGD:
    """Plain stochastic gradient descent with optional L2 weight decay.

    p <- p - lr * (g + weight_decay * p)   for weight matrices (p.ndim >= 2)
    p <- p - lr * g                        for 1-D parameters (biases, norm gains)
    """

    def __init__(self, params: dict[str, np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads: dict[str, np.ndarray]) -> None:
        for name, p in self.params.items():
            g = grads[name]
            if self.weight_decay and p.ndim >= 2:
                g = g + self.weight_decay * p
            p -= self.lr * g  # in place, so the layer sees the update


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True) -> Iterator[np.ndarray]:
    """Yield index arrays covering range(n) exactly once. The last batch may be short."""
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train(
    model: Sequential,
    loss: SoftmaxCrossEntropy,
    opt: SGD,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[float]:
    """Mini-batch SGD. Returns the mean training loss of each epoch (one float per epoch)."""
    n = len(X)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in iterate_minibatches(n, batch_size, rng):
            model.zero_grad()
            logits = model.forward(X[idx])
            batch_loss = loss.forward(logits, y[idx])
            model.backward(loss.backward())
            opt.step(model.grads())
            total += batch_loss * len(idx)  # weight by batch size: the last batch may be short
        history.append(total / n)
    return history


def accuracy(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of argmax predictions equal to y, computed in eval mode. Leaves the mode as it found it."""
    was_training = model.training
    model.eval()
    pred = model.forward(X).argmax(axis=1)
    if was_training:
        model.train()
    return float(np.mean(pred == y))
