"""ROOM 4.4 - THE CURSED TRAINING LOOP  (reference solution: the seven curses lifted)

Spoilers below. The cursed file you are meant to fix is in ../rooms/.

The seven curses, and the line that lifted each:

  1. lr=10.0                      -> lr=0.05. The loss got worse every step.
  2. labels cast to float32       -> int64. CrossEntropyLoss wants class indices.
  3. softmax before cross_entropy -> pass logits. The loss applies log_softmax itself.
  4. argmax(dim=0)                -> dim=1. One winner per example, not per class.
  5. no optimizer.zero_grad()     -> added. .grad accumulates across batches.
  6. total_loss += loss           -> loss.item(). Summing tensors keeps every graph alive.
  7. never model.train()/eval()   -> added. Dropout was active while measuring accuracy.
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
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)  # curse 1: was 10.0


def prepare_batch(xb, yb) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 inputs, int64 class labels."""
    x = torch.as_tensor(xb, dtype=torch.float32)
    y = torch.as_tensor(yb, dtype=torch.int64)  # curse 2: was float32
    return x, y


def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy straight from logits; the loss does its own log_softmax."""
    return F.cross_entropy(logits, y)  # curse 3: was softmax(logits) first


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows whose highest logit is the true class."""
    preds = logits.argmax(dim=1)  # curse 4: was dim=0
    return (preds == y).float().mean().item()


def train_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer) -> float:
    """One pass over ``loader``. Return the mean per-batch loss as a Python float."""
    model.train()  # curse 7 (half of it): dropout on while learning
    total_loss = 0.0
    for xb, yb in loader:
        x, y = prepare_batch(xb, yb)
        optimizer.zero_grad()  # curse 5: was missing
        logits = model(x)
        loss = compute_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # curse 6: was += loss
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader) -> float:
    """Accuracy of ``model`` over every example in ``loader``."""
    model.eval()  # curse 7 (other half): dropout off while measuring
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
