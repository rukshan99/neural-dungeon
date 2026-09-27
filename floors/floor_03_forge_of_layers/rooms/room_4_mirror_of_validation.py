"""ROOM 3.4 - THE MIRROR OF VALIDATION

    A tall mirror of polished steel, bolted to the forge wall. The smith
    never judges a blade by how it looks in the fire. She holds it up to the
    mirror, where the glow cannot flatter it.

The training loss measures how well the model fits the examples it was shown.
It cannot tell you whether the model learned the pattern or memorized the
examples, because from inside the training set those look identical. The only
honest measurement uses data the model has never touched: a VALIDATION set,
split off before training and never trained on.

Three things you can read from the two curves, train loss and val loss:

* Healthy: both fall, the gap between them stays small.
* Overfitting: train keeps falling while val bottoms out and climbs. The model
  has started spending its capacity on the noise in the training set. Past the
  val minimum, every extra epoch makes it WORSE at the actual task.
* Underfitting: neither goes anywhere. The model cannot even fit the training
  set: too small, too slow a learning rate, too few epochs, broken features.

Early stopping is the simplest regularizer there is: keep the parameters from
the epoch with the lowest val loss, and stop when ``patience`` epochs have
passed without a new low. It costs nothing and it works on every model.

The diagnose() rule on this floor, stated exactly so the trial can check it:
    1. overfitting  if val[-1] > (1 + OVERFIT_RISE) * min(val)  AND  train[-1] < train[argmin(val)]
    2. underfitting if train[-1] > UNDERFIT_FLOOR * train[0]
    3. healthy      otherwise
Rules like this are heuristics; the constants are honest defaults, not laws.
"""

from __future__ import annotations

import numpy as np

from .room_1_the_anvil import SoftmaxCrossEntropy
from .room_3_the_bellows import SGD, Sequential, train  # noqa: F401  (train is the loop you built)

OVERFIT_RISE = 0.10  # val loss has climbed this fraction above its minimum
UNDERFIT_FLOOR = 0.5  # train loss never even halved from where it started


def evaluate_loss(model: Sequential, loss: SoftmaxCrossEntropy, X: np.ndarray, y: np.ndarray) -> float:
    """Mean loss over X in eval mode, as a Python float. No backward, no update.

    Switch to eval mode first (dropout off), and restore the previous mode
    afterwards, the way Room 3's accuracy() does.
    """
    raise NotImplementedError("evaluate_loss() is unwritten")


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
    """Room 3's loop with the mirror held up: returns (train_losses, val_losses), one entry per epoch.

    For each epoch: run Room 3's ``train(...)`` for ONE epoch (so the training
    curve is identical to what train() alone would produce), then append
    ``evaluate_loss`` on the validation set. Validation data is looked at, never
    trained on.

    If ``patience`` is given, also do early stopping: after each epoch call
    ``early_stopping(val_losses, patience)``; whenever this epoch is the new best,
    snapshot a COPY of every parameter; when it says stop, break. Before
    returning, copy the best snapshot back into the model's parameters, so the
    caller gets the best model rather than the last one. Both lists then have
    as many entries as epochs actually run.
    """
    raise NotImplementedError("train_with_validation() is unwritten")


def early_stopping(val_losses: list[float], patience: int) -> tuple[int, bool]:
    """(index of the lowest val loss so far, whether to stop).

    best = the FIRST index of the minimum (np.argmin does this). Stop when at
    least ``patience`` epochs have passed since best, i.e. when
    ``len(val_losses) - 1 - best >= patience``. A list of one entry never stops.
    """
    raise NotImplementedError("early_stopping() is unwritten")


def diagnose(train_losses: list[float], val_losses: list[float]) -> str:
    """"overfitting", "underfitting" or "healthy", by the three-step rule in the module docstring.

    Use OVERFIT_RISE and UNDERFIT_FLOOR; the trial builds its curves around them.
    """
    raise NotImplementedError("diagnose() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY
# Three pairs of curves (train_losses, val_losses) scratched into the mirror's
# frame by earlier smiths. Read them by eye and write your diagnosis for each
# into MIRROR_PROPHECY before running anything. The trial applies the rule.
# Do not edit the curves.
# ---------------------------------------------------------------------------
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

MIRROR_PROPHECY: dict[str, str | None] = {
    "the_eager_apprentice": None,
    "the_blunt_hammer": None,
    "the_tempered_blade": None,
}
