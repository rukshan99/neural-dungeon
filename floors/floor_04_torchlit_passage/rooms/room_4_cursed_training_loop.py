"""ROOM 4.4 - THE CURSED TRAINING LOOP

    A training loop lies on the altar, complete and still running. It has been
    running for a very long time. It has learned nothing. Seven curses were
    laid on it by seven different adventurers, each of whom believed their
    version worked.

THIS ROOM HAS NO STUBS. The code below is complete and wrong in exactly SEVEN
places. Do not rewrite it from scratch; find each curse and lift it. Every
trial in ``trials/test_room_4.py`` names a *symptom*. Run the trial, read the
symptom, find the line. The last trial trains the whole loop on the runes and
expects at least 90% validation accuracy in three epochs, which the fixed
loop reaches in a couple of seconds.

The curses are all real bugs that real people ship. Some crash. Most do not.
The ones that do not crash are the ones that cost weeks.

Hints for the shape of the hunt (not for the answers):

  * one curse is a wrong dtype       * one curse is a wrong axis
  * one curse is a wrong scale       * one curse is something applied twice
  * one curse is a memory leak       * one curse is something never called
  * one curse is a mode never switched

Run:  dungeon trial 4 room_4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dungeon.artifacts.toydata import make_runes, train_val_split


def make_model(hidden: int = 64, dropout: float = 0.2) -> nn.Module:
    """A small MLP over flattened 8x8 runes, with dropout, returning 10 logits."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    )


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """SGD with momentum over the model's parameters."""
    return torch.optim.SGD(model.parameters(), lr=10.0, momentum=0.9)


def prepare_batch(xb, yb) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn whatever the loader yields into ``(x, y)`` tensors the model and loss accept."""
    x = torch.as_tensor(xb, dtype=torch.float32)
    y = torch.as_tensor(yb, dtype=torch.float32)
    return x, y


def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between ``logits`` of shape (B, C) and class labels ``y`` of shape (B,)."""
    probs = torch.softmax(logits, dim=1)
    return F.cross_entropy(probs, y)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows whose highest logit is the true class. Python float."""
    preds = logits.argmax(dim=0)
    return (preds == y).float().mean().item()


def train_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer) -> float:
    """One pass over ``loader``. Return the mean per-batch loss as a Python float."""
    total_loss = 0.0
    for xb, yb in loader:
        x, y = prepare_batch(xb, yb)
        logits = model(x)
        loss = compute_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader) -> float:
    """Accuracy of ``model`` over every example in ``loader``."""
    correct = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            x, y = prepare_batch(xb, yb)
            logits = model(x)
            correct += accuracy(logits, y) * len(y)
            total += len(y)
    return correct / total


def train_cursed(epochs: int = 3, seed: int = 0, batch_size: int = 32) -> dict:
    """Train on the runes and report ``{"train_loss": [per-epoch floats], "val_acc": float}``."""
    torch.manual_seed(seed)
    X, y = make_runes(n_per_class=100, seed=seed)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_fraction=0.2, seed=seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=100,
    )
    model = make_model()
    optimizer = make_optimizer(model)
    history = [train_epoch(model, train_loader, optimizer) for _ in range(epochs)]
    return {"train_loss": history, "val_acc": evaluate(model, val_loader)}
