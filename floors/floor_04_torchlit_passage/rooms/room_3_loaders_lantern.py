"""ROOM 4.3 - THE LOADER'S LANTERN

    A lantern-bearer walks ahead of you through a hall of rune tablets, lifting
    them into the light a handful at a time, in an order only the lantern
    knows. On Floor 3 you wrote that loop by hand. Here it has a name.

Two classes do the work:

* ``torch.utils.data.Dataset`` answers two questions: ``__len__`` (how many
  examples) and ``__getitem__(i)`` (the i-th example, as tensors).
* ``torch.utils.data.DataLoader`` turns a Dataset into an iterator over
  batches. It picks indices (in order, or shuffled), fetches each example,
  and *collates* them: a list of ``(x, y)`` pairs becomes ``(stack of x,
  stack of y)``. Scalars become 0-d tensors and then a 1-d batch.

Shuffling draws from a ``torch.Generator``. Pass your own, seeded, and the
order is reproducible without touching the global RNG. The generator's state
advances each epoch, so epoch 2 has a different order from epoch 1, and two
loaders seeded identically walk identical orders. The last batch is smaller
when the length does not divide evenly (``drop_last=False`` is the default).

The floor's data: ``make_runes`` from ``dungeon.artifacts.toydata``, 8x8 glyph
images in 10 classes. A miniature MNIST that needs no download.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RuneDataset(Dataset):
    """Rune glyphs and labels from numpy arrays.

    ``X`` is ``(N, 8, 8)`` (float32 usually, but float64 must be tolerated) and
    ``y`` is ``(N,)`` integer labels. ``__getitem__(i)`` returns a tuple
    ``(image, label)`` with ``image`` a ``float32`` tensor of shape ``(8, 8)`` and
    ``label`` an ``int64`` 0-d tensor (a Python int is also acceptable; the
    loader collates both into an ``int64`` batch).

    Convert once in ``__init__`` (``torch.as_tensor(..., dtype=...)``) rather than
    per item; the floor's data fits in memory many thousand times over.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("RuneDataset.__init__() is unwritten")

    def __len__(self) -> int:
        raise NotImplementedError("RuneDataset.__len__() is unwritten")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("RuneDataset.__getitem__() is unwritten")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """A DataLoader whose shuffle is reproducible from ``seed``.

    Build ``torch.Generator()`` seeded with ``manual_seed(seed)`` and pass it as
    ``generator=``. Keep ``drop_last=False`` and ``num_workers=0`` (the default:
    worker processes have their own seeding story, told in the README).
    """
    raise NotImplementedError("make_loader() is unwritten")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """One pass over ``loader``. Return the mean of the per-batch losses as a Python float.

    The five lines every torch training loop is made of, in this order:
    zero the gradients, forward, loss, backward, step. Before the loop, put the
    model in training mode (``model.train()``); Dropout and BatchNorm behave
    differently otherwise. Accumulate ``loss.item()``, never the loss tensor.
    """
    raise NotImplementedError("train_one_epoch() is unwritten")


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Classification accuracy of ``model`` over ``loader``, in [0, 1], as a Python float.

    Evaluation mode (``model.eval()``) and no graph (``torch.no_grad()``): you are
    measuring, not learning, and a graph you never backward through is memory
    thrown away. Predictions are ``logits.argmax(dim=1)``. Count correct
    examples across all batches and divide by the total number of examples,
    not by the number of batches (the last batch may be smaller).
    """
    raise NotImplementedError("evaluate() is unwritten")
