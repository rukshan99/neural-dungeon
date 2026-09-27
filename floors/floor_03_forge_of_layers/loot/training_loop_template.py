"""Training Loop Template - loot from Floor 3, The Forge of Layers.

A complete, dependency-free (numpy only) supervised training loop with:

    * mini-batch SGD with shuffling and a short last batch,
    * L2 weight decay (on weight matrices only),
    * inverted dropout with train/eval modes,
    * a validation set evaluated after every epoch,
    * early stopping with the best parameters restored,
    * a diagnosis of the two loss curves.

Copy it into a project and replace the tiny layer classes with whatever you
use there (PyTorch modules follow the same shape: zero_grad, forward, loss,
backward, step). The structure is the thing worth keeping.

Run it:  python training_loop_template.py
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

# --------------------------------------------------------------------------- layers
# Each layer: forward(x) -> out (caching what backward needs), backward(dout) -> dx,
# parameters in .params, gradients ACCUMULATED into .grads, zero_grad() to reset.


class Linear:
    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator, init: str = "he") -> None:
        std = np.sqrt(2.0 / fan_in) if init == "he" else np.sqrt(1.0 / fan_in)  # He for ReLU, LeCun for tanh
        self.params = {"W": rng.standard_normal((fan_in, fan_out)) * std, "b": np.zeros(fan_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        self.grads["W"] += x.T @ dout  # (in, N) @ (N, out)
        self.grads["b"] += dout.sum(axis=0)  # one bias, every example pushes on it
        return dout @ self.params["W"].T  # (N, out) @ (out, in)

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0


class ReLU:
    params: dict = {}
    grads: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x > 0
        return np.where(self.cache, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.cache

    def zero_grad(self) -> None:
        pass


class Dropout:
    """Inverted dropout: scale survivors by 1/(1-p) at train time so eval is the identity."""

    params: dict = {}
    grads: dict = {}

    def __init__(self, p: float, rng: np.random.Generator) -> None:
        self.p, self.rng, self.training, self.cache = p, rng, True, None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.cache = None
            return x
        self.cache = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.cache

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout if self.cache is None else dout * self.cache

    def zero_grad(self) -> None:
        pass


class SoftmaxCrossEntropy:
    """Fused softmax + mean NLL. Stable via log-sum-exp. backward() = (probs - onehot) / N."""

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.cache = (np.exp(log_probs), y)
        return float(-log_probs[np.arange(len(y)), y].mean())

    def backward(self) -> np.ndarray:
        probs, y = self.cache
        d = probs.copy()
        d[np.arange(len(y)), y] -= 1.0
        return d / len(y)


class Sequential:
    def __init__(self, layers: list) -> None:
        self.layers, self.training = list(layers), True

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> dict[str, np.ndarray]:
        return {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.params.items()}

    def grads(self) -> dict[str, np.ndarray]:
        return {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.grads.items()}

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

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


# --------------------------------------------------------------------------- optimizer


class SGD:
    """p <- p - lr * (g + weight_decay * p) for weight matrices; biases are never decayed."""

    def __init__(self, params: dict[str, np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        self.params, self.lr, self.weight_decay = params, lr, weight_decay

    def step(self, grads: dict[str, np.ndarray]) -> None:
        for name, p in self.params.items():
            g = grads[name]
            if self.weight_decay and p.ndim >= 2:
                g = g + self.weight_decay * p
            p -= self.lr * g  # in place: the model's own array moves


# --------------------------------------------------------------------------- the loop


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True) -> Iterator[np.ndarray]:
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def evaluate(model: Sequential, loss: SoftmaxCrossEntropy, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(mean loss, accuracy) in eval mode; the model's mode is restored afterwards."""
    was_training = model.training
    model.eval()
    logits = model.forward(X)
    value = loss.forward(logits, y)
    acc = float(np.mean(logits.argmax(axis=1) == y))
    if was_training:
        model.train()
    return value, acc


def fit(
    model: Sequential,
    loss: SoftmaxCrossEntropy,
    opt: SGD,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    patience: int | None = 20,
    log: Callable[[str], None] | None = print,
) -> dict[str, list[float]]:
    """Train with validation after every epoch and early stopping on val loss.

    Returns the history dict. On return the model holds the parameters of the
    epoch with the lowest validation loss (if patience is set).
    """
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, since_best = np.inf, None, 0
    n = len(X_train)
    for epoch in range(1, epochs + 1):
        # ---- one epoch of training: the five lines, per batch
        model.train()
        total = 0.0
        for idx in iterate_minibatches(n, batch_size, rng):
            model.zero_grad()  # 1. forget the previous batch's gradients
            logits = model.forward(X_train[idx])  # 2. forward
            batch_loss = loss.forward(logits, y_train[idx])  # 3. measure
            model.backward(loss.backward())  # 4. backward
            opt.step(model.grads())  # 5. update
            total += batch_loss * len(idx)
        history["train_loss"].append(total / n)

        # ---- hold it up to the mirror
        val_loss, val_acc = evaluate(model, loss, X_val, y_val)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if log and (epoch % 10 == 0 or epoch == 1):
            log(f"epoch {epoch:4d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}  val acc {val_acc:.3f}")

        # ---- early stopping: snapshot on improvement, quit after `patience` epochs without one
        if patience is not None:
            if val_loss < best_val:
                best_val, since_best = val_loss, 0
                best_state = {k: v.copy() for k, v in model.params().items()}
            else:
                since_best += 1
                if since_best >= patience:
                    if log:
                        log(f"early stop at epoch {epoch}; best val loss {best_val:.4f} at epoch {epoch - since_best}")
                    break
    if best_state is not None:
        for k, p in model.params().items():
            p[...] = best_state[k]
    return history


def diagnose(train_losses: list[float], val_losses: list[float], rise: float = 0.10, floor: float = 0.5) -> str:
    """"overfitting" if val climbed >rise above its min while train kept falling; "underfitting" if train never halved."""
    tr, va = np.asarray(train_losses), np.asarray(val_losses)
    best = int(np.argmin(va))
    if va[-1] > (1 + rise) * va[best] and tr[-1] < tr[best]:
        return "overfitting"
    if tr[-1] > floor * tr[0]:
        return "underfitting"
    return "healthy"


# --------------------------------------------------------------------------- demo


def build_mlp(sizes: list[int], rng: np.random.Generator, dropout: float = 0.0) -> Sequential:
    layers: list = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(fan_in, fan_out, rng, init="he"))
        if i < len(sizes) - 2:
            layers.append(ReLU())
            if dropout > 0:
                layers.append(Dropout(dropout, rng))
    return Sequential(layers)


if __name__ == "__main__":
    # Any (X, y) with X float64 (N, D) and y int (N,) in 0..C-1 will do. Here: two noisy blobs.
    rng = np.random.default_rng(0)
    N, D, C = 600, 2, 3
    X = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)
    X += np.eye(C)[y][:, :D] * 2.0  # shift each class a little
    perm = rng.permutation(N)
    X_train, y_train, X_val, y_val = X[perm[:450]], y[perm[:450]], X[perm[450:]], y[perm[450:]]

    model = build_mlp([D, 64, 64, C], rng, dropout=0.1)
    opt = SGD(model.params(), lr=0.1, weight_decay=1e-3)
    hist = fit(model, SoftmaxCrossEntropy(), opt, X_train, y_train, X_val, y_val, epochs=200, batch_size=32, rng=rng, patience=20)
    print("diagnosis:", diagnose(hist["train_loss"], hist["val_loss"]))
    print("final val acc:", evaluate(model, SoftmaxCrossEntropy(), X_val, y_val)[1])
