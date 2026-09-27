"""ROOM 3.4 - THE MIRROR OF VALIDATION  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Training loss tells you how well the model fits what it has seen. Only data it
has never touched tells you whether it learned the pattern or the examples.
"""

from __future__ import annotations

import copy

import numpy as np

from .room_1_the_anvil import SoftmaxCrossEntropy
from .room_3_the_bellows import SGD, Sequential, train

OVERFIT_RISE = 0.10  # val loss has climbed this fraction above its minimum
UNDERFIT_FLOOR = 0.5  # train loss never even halved from where it started


def evaluate_loss(model: Sequential, loss: SoftmaxCrossEntropy, X: np.ndarray, y: np.ndarray) -> float:
    """Mean loss over X in eval mode, no gradient, mode restored afterwards."""
    was_training = model.training
    model.eval()
    value = loss.forward(model.forward(X), y)
    if was_training:
        model.train()
    return float(value)


def train_with_validation(
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
    patience: int | None = None,
) -> tuple[list[float], list[float]]:
    """Room 3's loop plus a mirror: after every epoch, measure the loss on data the model never trains on.

    With ``patience`` set, stop when early_stopping says so and restore the
    parameters of the best epoch before returning.
    """
    train_hist: list[float] = []
    val_hist: list[float] = []
    best_state: dict[str, np.ndarray] | None = None
    for _ in range(epochs):
        train_hist.extend(train(model, loss, opt, X_train, y_train, 1, batch_size, rng))
        val_hist.append(evaluate_loss(model, loss, X_val, y_val))
        if patience is not None:
            best, stop = early_stopping(val_hist, patience)
            if best == len(val_hist) - 1:
                best_state = copy.deepcopy(model.params())
            if stop:
                break
    if patience is not None and best_state is not None:
        for name, p in model.params().items():
            p[...] = best_state[name]
    return train_hist, val_hist


def early_stopping(val_losses: list[float], patience: int) -> tuple[int, bool]:
    """(index of the lowest val loss, whether ``patience`` epochs have passed since it)."""
    best = int(np.argmin(val_losses))  # the FIRST minimum on ties
    epochs_since_best = len(val_losses) - 1 - best
    return best, epochs_since_best >= patience


def diagnose(train_losses: list[float], val_losses: list[float]) -> str:
    """Read the two curves. Returns "overfitting", "underfitting" or "healthy".

    1. overfitting:  val_losses[-1] > (1 + OVERFIT_RISE) * min(val_losses) and the train
                     loss is lower at the end than it was at the val minimum.
    2. underfitting: train_losses[-1] > UNDERFIT_FLOOR * train_losses[0].
    3. healthy:      anything else.
    """
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    best = int(np.argmin(val_losses))
    val_climbed = val_losses[-1] > (1.0 + OVERFIT_RISE) * val_losses[best]
    train_kept_falling = train_losses[-1] < train_losses[best]
    if val_climbed and train_kept_falling:
        return "overfitting"
    if train_losses[-1] > UNDERFIT_FLOOR * train_losses[0]:
        return "underfitting"
    return "healthy"


# Three pairs of curves scratched into the mirror's frame. Diagnose them by eye, then let the trial judge.
MIRROR_CURVES: dict[str, tuple[list[float], list[float]]] = {
    "the_eager_apprentice": (
        [1.10, 0.80, 0.55, 0.35, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03],
        [1.08, 0.85, 0.70, 0.62, 0.60, 0.61, 0.66, 0.72, 0.80, 0.88],
    ),
    "the_blunt_hammer": (
        [1.10, 1.05, 1.02, 1.00, 0.99, 0.98, 0.98, 0.97, 0.97, 0.97],
        [1.11, 1.06, 1.03, 1.01, 1.00, 0.99, 0.99, 0.98, 0.98, 0.98],
    ),
    "the_tempered_blade": (
        [1.10, 0.75, 0.52, 0.40, 0.33, 0.29, 0.26, 0.24, 0.23, 0.22],
        [1.09, 0.78, 0.56, 0.45, 0.39, 0.35, 0.33, 0.32, 0.31, 0.31],
    ),
}

MIRROR_PROPHECY: dict[str, str] = {
    "the_eager_apprentice": "overfitting",  # train -> 0.03 while val climbs from 0.60 to 0.88
    "the_blunt_hammer": "underfitting",  # neither curve gets anywhere: 1.10 -> 0.97
    "the_tempered_blade": "healthy",  # both fall, the gap stays small, val never climbs
}
