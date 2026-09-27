"""BOSS - THE OVERFIT HYDRA  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Hydra grows a head for every training example the model memorizes. Its
weakness is everything Room 4 taught: hold something back, watch it, and make
memorizing expensive (weight decay) or unreliable (dropout).
"""

from __future__ import annotations

import numpy as np

from dungeon.artifacts.toydata import make_spirals

from .room_1_the_anvil import Linear, ReLU, SoftmaxCrossEntropy
from .room_2_the_tempering import he_normal
from .room_3_the_bellows import SGD, Sequential, accuracy
from .room_4_mirror_of_validation import train_with_validation


def summon_the_hydra(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The arena, exactly as the trial builds it: (X_train, y_noisy, y_clean, X_val, y_val)."""
    X_train, y_clean = make_spirals(n_per_class=100, n_classes=3, noise=0.05, seed=1)
    rng = np.random.default_rng(seed)
    n_flip = int(round(0.2 * len(y_clean)))
    flip = rng.choice(len(y_clean), size=n_flip, replace=False)
    y_noisy = y_clean.copy()
    y_noisy[flip] = (y_clean[flip] + rng.integers(1, 3, size=n_flip)) % 3  # always a DIFFERENT class
    X_val, y_val = make_spirals(n_per_class=200, n_classes=3, noise=0.05, seed=2)
    return X_train.astype(np.float64), y_noisy, y_clean, X_val.astype(np.float64), y_val


# --------------------------------------------------------------------------- phase 1


def l2_penalty(params: dict[str, np.ndarray], weight_decay: float) -> float:
    """0.5 * weight_decay * sum of squares of every weight matrix (ndim >= 2). Biases are exempt.

    Its gradient with respect to W is weight_decay * W: exactly the term SGD's
    weight_decay adds to the gradient. Penalty in the loss, decay in the update:
    the same thing seen from two sides.
    """
    return float(0.5 * weight_decay * sum(np.sum(p**2) for p in params.values() if p.ndim >= 2))


class Dropout:
    """Inverted dropout. Train: zero each unit with probability p, scale the rest by 1/(1-p). Eval: identity."""

    def __init__(self, p: float, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.training = True
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.cache = None
            return x
        keep = self.rng.random(x.shape) >= self.p
        self.cache = keep / (1.0 - self.p)  # the scale is folded into the mask
        return x * self.cache

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            return dout
        return dout * self.cache  # the SAME mask: dropped units get no gradient

    def zero_grad(self) -> None:
        pass


# --------------------------------------------------------------------------- phase 2


def count_heads(model, X_train: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray) -> int:
    """How many flipped labels the model has memorized: flipped AND predicted == noisy label."""
    was_training = model.training
    model.eval()
    pred = model.forward(X_train).argmax(axis=1)
    if was_training:
        model.train()
    flipped = y_noisy != y_clean
    return int(np.sum(flipped & (pred == y_noisy)))


# --------------------------------------------------------------------------- phase 3


def slay_the_hydra(
    X_train: np.ndarray,
    y_noisy: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: np.random.Generator,
):
    """A regularized MLP, trained with the mirror held up. Returned in eval mode.

    Two weapons carry this fight: L2 weight decay (memorizing an isolated point
    needs sharp, large weights, and decay taxes them every step) and early
    stopping on the clean validation set (networks learn the pattern first and
    the lies later; the mirror says when to stop). Dropout is implemented above
    and you should try it, but on a 64-wide net solving a 2-D problem it slowed
    learning more than it stopped memorizing in our calibration. Dropout earns
    its keep on wide layers with many redundant features.
    """
    sizes = [2, 64, 64, 3]
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = Linear(fan_in, fan_out, rng)
        lin.params["W"][...] = he_normal(fan_in, fan_out, rng)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(ReLU())
    model = Sequential(layers)
    loss = SoftmaxCrossEntropy()
    opt = SGD(model.params(), lr=0.2, weight_decay=1e-3)
    train_with_validation(
        model, loss, opt, X_train, y_noisy, X_val, y_val,
        epochs=800, batch_size=16, rng=rng, patience=50,
    )
    model.eval()
    return model


if __name__ == "__main__":  # pragma: no cover - a peek at the arena
    X_train, y_noisy, y_clean, X_val, y_val = summon_the_hydra()
    model = slay_the_hydra(X_train, y_noisy, X_val, y_val, np.random.default_rng(0))
    print("val accuracy", accuracy(model, X_val, y_val))
    print("heads", count_heads(model, X_train, y_noisy, y_clean), "of", int((y_noisy != y_clean).sum()))
