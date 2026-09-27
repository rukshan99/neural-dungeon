"""ROOM 4.3 - THE LOADER'S LANTERN  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RuneDataset(Dataset):
    """(N, 8, 8) images and (N,) labels as float32 / int64 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # Convert once. as_tensor with an explicit dtype cures numpy's float64 at the border.
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.int64)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """A private, seeded die for the shuffle: reproducible without touching the global RNG."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """zero_grad, forward, loss, backward, step. Mean per-batch loss as a float."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()  # .item() detaches; summing tensors would keep every graph alive
    return total / len(loader)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Accuracy over every example, in eval mode, without building a graph."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.shape[0]
    return correct / total
