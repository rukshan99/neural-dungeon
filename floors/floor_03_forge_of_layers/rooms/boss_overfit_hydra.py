"""BOSS - THE OVERFIT HYDRA

                     __/\\__      __/\\__
                    (  oo  )    (  oo  )        "Show me your training set.
                     \\ \\/ /      \\ \\/ /          I will learn every example.
                 __/\\_\\  /__/\\__  \\  /_/\\__     Every. Single. One."
                (  oo  )( oo  )(  oo  )(  oo )
                 \\ \\/ /  \\ \\/ /  \\ \\/ /  \\ \\/ /   Each head is one training
                  \\  /    \\  /    \\  /    \\  /    example the model has
                   ||      ||      ||      ||     MEMORIZED. One in five of
                 __||______||______||______||__   the labels in its lair is
                /                              \\  a lie, and the Hydra grows
               (   t r a i n i n g   s e t     )  a head for every lie it
                \\______________________________/  learns by heart.

WEAKNESS: regularization and validation. Weight decay makes memorizing
expensive, dropout makes it unreliable, early stopping cuts it short, and the
mirror (a validation set) is the only way to see the heads at all.

THE ARENA (built by the trial; summon_the_hydra() builds the same one for you):

    X_train, y_clean  = make_spirals(n_per_class=100, n_classes=3, noise=0.05, seed=1)
    y_noisy           = y_clean with 20% of the labels flipped to a WRONG class
    X_val, y_val      = make_spirals(n_per_class=200, n_classes=3, noise=0.05, seed=2), clean

    a "head" = a training example whose label was flipped AND whose predicted
               class equals the flipped (wrong) label. The model learned the lie.

The fight has three phases:

  Phase 1  l2_penalty and Dropout: the two classic regularizers, exactly.
  Phase 2  count_heads: see the memorization.
  Phase 3  slay_the_hydra: train a model on the noisy labels that reaches
           >= 0.80 validation accuracy with at most 15 heads (of 60 lies) and a
           train-val accuracy gap of at most 0.20. The trial also trains a big
           model with NO regularization and shows you it grows 20+ heads.

Run:  dungeon fight 3
"""

from __future__ import annotations

import numpy as np

from dungeon.artifacts.toydata import make_spirals

from .room_1_the_anvil import Linear, ReLU, SoftmaxCrossEntropy  # noqa: F401
from .room_2_the_tempering import he_normal  # noqa: F401
from .room_3_the_bellows import SGD, Sequential, accuracy  # noqa: F401
from .room_4_mirror_of_validation import train_with_validation  # noqa: F401


def summon_the_hydra(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The arena, exactly as the trial builds it: (X_train, y_noisy, y_clean, X_val, y_val). Written for you."""
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
    """0.5 * weight_decay * sum over WEIGHT MATRICES (ndim >= 2) of sum(W**2). A Python float.

    Biases (1-D) are exempt: they shift, they do not amplify, and decaying them
    only hurts. Check for yourself that d(penalty)/dW = weight_decay * W: exactly
    the term Room 3's SGD adds to the gradient. The penalty is the loss-side
    view; the decay is the update-side view of the same regularizer.
    """
    raise NotImplementedError("l2_penalty() is unwritten")


class Dropout:
    """Inverted dropout.

    Train mode: each element is kept with probability 1 - p (mask = rng.random(x.shape) >= p)
    and the kept elements are scaled by 1 / (1 - p), so the EXPECTED output equals
    the input and nothing needs to change at test time. Eval mode: identity.

    backward multiplies dout by the SAME scaled mask that forward used: units
    that were dropped receive no gradient, units that were kept pass it scaled.
    Cache the scaled mask in forward. ``__init__`` and ``zero_grad`` are written.
    """

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
        """Train: x * mask / (1 - p) with a fresh mask each call. Eval (or p == 0): return x unchanged."""
        raise NotImplementedError("Dropout.forward() is unwritten")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout * (the same scaled mask), or dout unchanged if forward was the identity."""
        raise NotImplementedError("Dropout.backward() is unwritten")

    def zero_grad(self) -> None:
        pass


# --------------------------------------------------------------------------- phase 2


def count_heads(model, X_train: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray) -> int:
    """Number of i with y_noisy[i] != y_clean[i] AND argmax(model(X_train))[i] == y_noisy[i]. A Python int.

    Predict in eval mode (dropout off) and restore the model's mode afterwards.
    A flipped example the model gets *right* (predicts y_clean) is not a head:
    the model refused the lie.
    """
    raise NotImplementedError("count_heads() is unwritten")


# --------------------------------------------------------------------------- phase 3


def slay_the_hydra(
    X_train: np.ndarray,
    y_noisy: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: np.random.Generator,
):
    """Train and return a model (a Sequential, in eval mode) that refuses to memorize the lies.

    Requirements checked by the trial:
        accuracy(model, X_val, y_val) >= 0.80
        count_heads(model, X_train, y_noisy, y_clean) <= 15      (there are 60 lies)
        accuracy(model, X_train, y_noisy) - accuracy(model, X_val, y_val) <= 0.20

    Everything you need is imported above. Weapons: SGD(weight_decay=...),
    Dropout, and train_with_validation(..., patience=...) so the mirror picks
    the epoch. Draw every random number from ``rng``. A [2, 64, 64, 3] ReLU MLP
    with He init is plenty of capacity; the question is how you restrain it.
    Make the wild, unregularized version first (dungeon trial output shows you
    its heads), then add one weapon at a time and watch the count.
    """
    raise NotImplementedError("slay_the_hydra() is unwritten")
