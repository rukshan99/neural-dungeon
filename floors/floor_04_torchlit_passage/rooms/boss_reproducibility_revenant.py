"""BOSS - THE REPRODUCIBILITY REVENANT

                    .-~~~~-.
                   /  _  _  \\        "You trained me yesterday. You wrote
                  |  (o)(o)  |        down the accuracy. Train me again.
                  |    __    |        Go on. I will be someone else."
                   \\  '--'  /
                 .-'`-....-'`-.       The Revenant comes back DIFFERENT every
                /   |  ||  |   \\      time. Weight init. Shuffle order. Dropout
               /    |  ||  |    \\     masks. Every die in the building rolls
              /_____|__||__|_____\\    without you, and it counts on that.

WEAKNESS: someone who seeds every die and checkpoints every piece of state,
so that a run can be replayed, or interrupted and resumed, to the bit.

Sources of randomness in a torch training run, in the order you meet them:

    torch.manual_seed(s)     weight init, dropout masks, torch.randn, and the
                             DataLoader's shuffle IF you did not give it a
                             generator (it then seeds itself from the global RNG)
    torch.Generator          the loader's private die, if you hand it one
    random / numpy           anything YOUR code does with them (augmentation)
    the hardware             some CUDA kernels are non-deterministic by design;
                             torch.use_deterministic_algorithms(True) trades
                             speed for a guarantee. CPU kernels are deterministic
                             for a fixed thread count.

A checkpoint that can resume EXACTLY holds: model.state_dict(),
optimizer.state_dict() (momentum buffers, Adam moments, step counts), the
epoch, and the RNG states: torch.get_rng_state() and the loader generator's
get_state(). Miss any one and the resumed run drifts from the straight one.

The fight has three phases:

  Phase 1  seed_everything             - every die, one call.
  Phase 2  train_deterministically     - two runs, bitwise identical.
  Phase 3  save/load_checkpoint and train_with_resume - interrupt after
           2 epochs, rebuild everything from scratch, resume, and match a
           3-epoch straight run exactly.

Run:  dungeon fight 4
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# You will also want, once the stubs are gone:
#     import random
#     import numpy as np
#     import torch.nn.functional as F
#     from torch.utils.data import TensorDataset
#     from dungeon.artifacts.toydata import make_runes, train_val_split

DATA_SEED = 0  # the runes themselves are the dungeon's; only the training dice are yours


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python's ``random``, numpy's global RNG and torch (all devices) with ``seed``.

    If ``deterministic`` is True also call ``torch.use_deterministic_algorithms(True)``
    (and ``False`` otherwise, so the flag never leaks between calls). It must
    not break anything on CPU.
    """
    raise NotImplementedError("seed_everything() is unwritten")


def build_run(seed: int) -> tuple[nn.Module, DataLoader, torch.optim.Optimizer]:
    """Everything a run needs, built from ``seed`` alone.

    1. ``seed_everything(seed)``.
    2. The model: ``nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
       nn.Dropout(0.1), nn.Linear(64, 10))``. The Dropout is deliberate: its
       masks come from the global torch RNG, which is exactly the die the
       Revenant hides behind.
    3. The data: ``make_runes(n_per_class=100, seed=DATA_SEED)``, then
       ``train_val_split(..., val_fraction=0.2, seed=DATA_SEED)``; keep the
       training split as a ``TensorDataset`` of float32 images and int64 labels.
    4. The loader: ``batch_size=32, shuffle=True`` with its own
       ``torch.Generator().manual_seed(seed)`` passed as ``generator=``.
    5. The optimizer: ``torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)``.
    """
    raise NotImplementedError("build_run() is unwritten")


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """One training epoch with ``F.cross_entropy``; mean per-batch loss as a Python float.

    The same five lines as room 4.3, in training mode.
    """
    raise NotImplementedError("run_epoch() is unwritten")


def train_deterministically(seed: int, epochs: int = 2) -> tuple[nn.Module, list[float]]:
    """``build_run(seed)`` then ``epochs`` calls of ``run_epoch``. Return ``(model, losses)``.

    Two calls with the same seed must produce bitwise-identical parameters and
    identical loss lists. That is the whole of Phase 2.
    """
    raise NotImplementedError("train_deterministically() is unwritten")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rng_state: dict[str, torch.Tensor],
) -> None:
    """``torch.save`` a dict with keys ``"model"``, ``"optimizer"``, ``"epoch"``, ``"rng_state"``.

    ``"model"`` and ``"optimizer"`` are the two ``state_dict()``s. ``rng_state`` is a
    dict of TENSORS, e.g. ``{"torch": torch.get_rng_state(), "loader":
    loader.generator.get_state()}``. Keep the whole checkpoint to tensors, numbers,
    strings and plain containers so that it loads under ``torch.load``'s safe
    default (``weights_only=True``). numpy's ``get_state()`` would not.
    """
    raise NotImplementedError("save_checkpoint() is unwritten")


def load_checkpoint(path: str | Path) -> dict:
    """Read a ``save_checkpoint`` file back: ``torch.load(path, map_location="cpu", weights_only=True)``."""
    raise NotImplementedError("load_checkpoint() is unwritten")


def train_with_resume(
    seed: int,
    epochs: int = 3,
    stop_after: int = 2,
    path: str | Path = "revenant.pt",
) -> tuple[nn.Module, list[float]]:
    """Train ``stop_after`` epochs, checkpoint, REBUILD everything, restore, finish. Return ``(model, losses)``.

    1. ``build_run(seed)``; run ``stop_after`` epochs, collecting losses.
    2. ``save_checkpoint(path, ...)`` with epoch = ``stop_after`` and the RNG
       states of both the global torch RNG and the loader's generator.
    3. Pretend the process died: ``build_run(seed)`` again, fresh model, fresh
       loader, fresh optimizer. Their state is now wrong in three ways.
    4. ``load_checkpoint(path)``: load the model and optimizer state_dicts,
       ``torch.set_rng_state(...)`` and ``loader.generator.set_state(...)``.
    5. Run the remaining ``epochs - stop_after`` epochs, appending losses.

    The result must equal ``train_deterministically(seed, epochs)`` exactly:
    same losses, ``torch.equal`` on every tensor of the state_dict.
    """
    raise NotImplementedError("train_with_resume() is unwritten")
