"""Training loop template (torch) - loot from Floor 4, the Torchlit Passage.

A clean, device-agnostic, fully seeded training loop with checkpointing and
exact resume. Copy it into a project and delete what you do not need. Every
choice in here was a trial on Floor 4.

    seed_everything      every die, once
    pick_device          cuda > mps > cpu, or an override
    to_device            nested batches across the river
    make_loaders         a Generator per loader, so the shuffle is yours
    train_one_epoch      zero_grad, forward, loss, backward, step; .item() for logging
    evaluate             eval mode AND no_grad
    save/load_checkpoint model, optimizer, epoch, RNG states; loads under weights_only=True
    main                 --resume picks up where the last run died, bit for bit on CPU

Run the demo (trains on the dungeon's runes, needs the repo on sys.path):

    python floors/floor_04_torchlit_passage/loot/training_loop_template_torch.py --epochs 3
    python floors/floor_04_torchlit_passage/loot/training_loop_template_torch.py --epochs 5 --resume
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------------- seeds and devices


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Python, numpy, torch (CPU and every CUDA device). Call once, before building the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)


def pick_device(prefer: str | None = None) -> torch.device:
    """An override is trusted; otherwise the best bank available."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device) -> Any:
    """Move every tensor in a nested list / tuple / dict; leave everything else alone."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


# ----------------------------------------------------------------------------- model and data


def build_model(in_features: int = 64, hidden: int = 64, n_classes: int = 10, dropout: float = 0.1) -> nn.Module:
    """Swap in your own. Returns logits; the loss does the squashing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """float32 features, int64 labels, and a private seeded Generator for the shuffle."""

    def dataset(X, y):
        return TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.int64))

    train_loader = DataLoader(
        dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        # num_workers=4, pin_memory=True,  # on a GPU box; then also seed numpy per worker (see the checklist)
    )
    val_loader = DataLoader(dataset(X_val, y_val), batch_size=max(batch_size, 256), shuffle=False)
    return train_loader, val_loader


# ----------------------------------------------------------------------------- the loop


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """The canonical five lines. Returns the mean per-batch loss as a float."""
    model.train()
    total = 0.0
    for batch in loader:
        x, y = to_device(batch, device)
        optimizer.zero_grad()  # 1. .grad accumulates; clear it
        logits = model(x)  # 2. forward
        loss = F.cross_entropy(logits, y)  # 3. raw logits in, int64 labels in
        loss.backward()  # 4. gradients into every .grad
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # here, if you clip
        optimizer.step()  # 5. in-place update under no_grad
        total += loss.item()  # detach for logging; never sum loss tensors
    return total / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """(mean loss, accuracy) over every example. eval mode AND no_grad; counts examples, not batches."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        x, y = to_device(batch, device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    return total_loss / n, correct / n


# ----------------------------------------------------------------------------- checkpoints


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loader: DataLoader) -> None:
    """Everything needed to continue as if nothing happened. Tensors and primitives only."""
    rng_state = {"torch": torch.get_rng_state(), "loader": loader.generator.get_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "rng_state": rng_state},
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader) -> int:
    """Restore model, optimizer and RNG states in place. Returns the epoch to continue FROM."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_state"]["torch"])
    loader.generator.set_state(checkpoint["rng_state"]["loader"])
    if "cuda" in checkpoint["rng_state"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
    return int(checkpoint["epoch"])


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a small MLP on the dungeon's runes, reproducibly.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps; default: best available")
    parser.add_argument("--checkpoint", default="runs/torchlit_template.pt")
    parser.add_argument("--resume", action="store_true", help="continue from --checkpoint if it exists")
    args = parser.parse_args(argv)

    # The demo data lives in the repo; a real project imports its own.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dungeon.artifacts.toydata import make_runes, train_val_split

    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}   seed: {args.seed}")

    X, y = make_runes(n_per_class=100, seed=0)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_fraction=0.2, seed=0)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, args.batch_size, args.seed)

    model = build_model().to(device)  # move BEFORE the optimizer exists and before any step
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    start_epoch = 0
    if args.resume and Path(args.checkpoint).is_file():
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer, train_loader)
        print(f"resumed from {args.checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1:>3}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val acc {val_acc:.3f}")
        save_checkpoint(args.checkpoint, model, optimizer, epoch + 1, train_loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
