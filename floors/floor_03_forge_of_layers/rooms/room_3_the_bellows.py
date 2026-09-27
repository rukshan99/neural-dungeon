"""ROOM 3.3 - THE BELLOWS

    Layers alone are cold iron. The bellows feed the fire: pump, and the
    forge breathes in a batch, breathes out a gradient, and the metal moves
    a little. Ten thousand breaths later you have a blade.

This room builds the training loop you will write on every floor below, in
every framework you will ever use. It is five lines:

    model.zero_grad()                       forget last batch's gradients
    logits = model.forward(X[idx])          forward
    L = loss.forward(logits, y[idx])        measure
    model.backward(loss.backward())         backward: blame flows down the stack
    opt.step(model.grads())                 move every parameter a little

Everything else is plumbing: a Sequential that runs layers forward and then
backward in reverse, an optimizer that owns the update rule, and a minibatch
iterator that visits every example exactly once per epoch in a fresh random
order.

Why minibatches? The full-dataset gradient is expensive and, past a point, no
more informative than a sample of it. A batch of 32 gives you a noisy but
unbiased estimate of the same gradient for 1/1000th of the cost, and the noise
turns out to help: it shakes the parameters out of sharp, poorly generalizing
minima. Why shuffle? Because if the batches are the same every epoch, the
model sees the same 32 examples together forever, and a sorted dataset (all
class 0, then all class 1...) makes every batch a lie about the whole.

Weight decay is here too, sleeping. SGD takes ``weight_decay`` and adds
``weight_decay * W`` to the gradient of every weight matrix (never to biases).
Room 4 and the boss will wake it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .room_1_the_anvil import SoftmaxCrossEntropy


class Sequential:
    """A stack of layers. forward applies them in order; backward in reverse.

    ``params()`` and ``grads()`` return FLAT dicts keyed "<layer index>.<name>",
    e.g. {"0.W": ..., "0.b": ..., "2.W": ..., "2.b": ...} for Linear, ReLU, Linear.
    The arrays in params() must be the layers' own arrays (not copies), so that
    an optimizer updating them in place updates the model.

    ``training`` is a flag that train()/eval() propagate to every layer that has
    a ``training`` attribute (Dropout, BatchNorm). Plain layers ignore it.
    """

    def __init__(self, layers: list) -> None:
        self.layers = list(layers)
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Feed x through every layer in order and return the last output."""
        raise NotImplementedError("Sequential.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Feed dout through every layer in REVERSE order and return dx for the input."""
        raise NotImplementedError("Sequential.backward() is unwritten")

    def params(self) -> dict[str, np.ndarray]:
        """{"<i>.<name>": array} for every parameter of every layer. The layers' own arrays."""
        raise NotImplementedError("Sequential.params() is unwritten")

    def grads(self) -> dict[str, np.ndarray]:
        """Same keys as params(), holding each layer's current gradients."""
        raise NotImplementedError("Sequential.grads() is unwritten")

    def zero_grad(self) -> None:
        """Call zero_grad() on every layer."""
        raise NotImplementedError("Sequential.zero_grad() is unwritten")

    # Mode switching is plumbing, so it is written for you. Notice what it does:
    # nothing on this floor's Room 1 layers, everything for Dropout and BatchNorm.
    def train(self) -> None:
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
    """Stochastic gradient descent with optional L2 weight decay.

    For every parameter p with gradient g:
        p <- p - lr * (g + weight_decay * p)    if p.ndim >= 2 (a weight matrix)
        p <- p - lr * g                         if p.ndim == 1 (a bias or a norm gain)

    Update IN PLACE (``p -= ...``, never ``p = p - ...``): ``params`` holds the
    model's own arrays and a rebinding would silently detach the optimizer from
    the model.
    """

    def __init__(self, params: dict[str, np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads: dict[str, np.ndarray]) -> None:
        """Apply one update to every parameter. ``grads`` has the same keys as ``self.params``."""
        raise NotImplementedError("SGD.step() is unwritten")


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True) -> Iterator[np.ndarray]:
    """Yield integer index arrays that together cover range(n) exactly once.

    Every batch has ``batch_size`` indices except possibly the last, which holds
    the remainder (never drop it: those examples deserve their turn). With
    ``shuffle=True`` the order is a fresh ``rng.permutation(n)``; with
    ``shuffle=False`` it is 0, 1, 2, ... (useful for evaluation).
    """
    raise NotImplementedError("iterate_minibatches() is unwritten")


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
    """Mini-batch SGD for ``epochs`` passes over the data. Return one mean training loss per epoch.

    Each epoch: put the model in train mode, iterate shuffled minibatches, and
    for each run the five lines from the module docstring. The epoch's loss is
    the average per-example loss over the epoch, i.e. sum(batch_loss * len(idx)) / n,
    so a short last batch is not over-weighted.
    """
    raise NotImplementedError("train() is unwritten")


def accuracy(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of examples whose argmax logit equals y. A Python float.

    Evaluate in eval mode (Dropout must be off when you measure), then put the
    model back in whatever mode it was in. A model you merely looked at should
    not come back changed.
    """
    raise NotImplementedError("accuracy() is unwritten")
