"""BOSS - THE REPRODUCIBILITY REVENANT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Three ideas defeat the Revenant:

1. Seed every source of randomness, once, at the start of a run.
2. Give the DataLoader its own Generator so the shuffle order does not depend on
   how many other random numbers the rest of the program drew.
3. A checkpoint is *all* the state: weights, optimizer buffers, the epoch, and
   the RNG states (global torch RNG for dropout, the loader's generator for the
   shuffle). With those restored, a resumed run is bitwise identical to an
   uninterrupted one on CPU.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dungeon.artifacts.toydata import make_runes, train_val_split

DATA_SEED = 0


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Python, numpy and torch (CPU and every CUDA device) from one seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA generators if any exist
    torch.use_deterministic_algorithms(deterministic)


def build_run(seed: int) -> tuple[nn.Module, DataLoader, torch.optim.Optimizer]:
    """Model, loader and optimizer, all reproducible from ``seed``."""
    seed_everything(seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 10),
    )
    X, y = make_runes(n_per_class=100, seed=DATA_SEED)
    X_train, y_train, _, _ = train_val_split(X, y, val_fraction=0.2, seed=DATA_SEED)
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.int64),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, loader, optimizer


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """The canonical five lines, mean per-batch loss as a float."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_deterministically(seed: int, epochs: int = 2) -> tuple[nn.Module, list[float]]:
    """Two calls with the same seed give bitwise-identical weights and identical losses."""
    model, loader, optimizer = build_run(seed)
    losses = [run_epoch(model, loader, optimizer) for _ in range(epochs)]
    return model, losses


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rng_state: dict[str, torch.Tensor],
) -> None:
    """Everything needed to continue as if nothing had happened."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "rng_state": rng_state,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict:
    """Read it back under the safe default: tensors and plain containers only."""
    return torch.load(path, map_location="cpu", weights_only=True)


def train_with_resume(
    seed: int,
    epochs: int = 3,
    stop_after: int = 2,
    path: str | Path = "revenant.pt",
) -> tuple[nn.Module, list[float]]:
    """Train, checkpoint, forget everything, restore, finish. Exactly."""
    model, loader, optimizer = build_run(seed)
    losses = [run_epoch(model, loader, optimizer) for _ in range(stop_after)]
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=stop_after,
        rng_state={"torch": torch.get_rng_state(), "loader": loader.generator.get_state()},
    )

    # The process dies here. A new one starts with fresh, wrong state everywhere.
    model, loader, optimizer = build_run(seed)
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])  # momentum buffers come back
    torch.set_rng_state(checkpoint["rng_state"]["torch"])  # dropout masks continue
    loader.generator.set_state(checkpoint["rng_state"]["loader"])  # shuffle order continues

    for _ in range(checkpoint["epoch"], epochs):
        losses.append(run_epoch(model, loader, optimizer))
    return model, losses
