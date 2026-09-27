"""ROOM 8.3 - THE ADAPTATION  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The fine-tuning loop is the pretraining loop with two differences: the model
starts from the checkpoint instead of random weights, and the optimizer is
handed *only* the parameters that are still trainable. Everything the model
needs to keep is decided by ``requires_grad`` before this file runs.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer


def make_batches(
    text: str,
    tokenizer: CharTokenizer,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """An endless stream of ``(x, y)`` batches of random windows from ``text``.

    ``x`` and ``y`` are ``(batch_size, block_size)`` int64 tensors and ``y`` is
    ``x`` shifted one character to the right (next-character targets). Window
    starts are drawn with ``generator`` so the same seed gives the same stream.
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(data) < block_size + 2:
        raise ValueError(f"text has {len(data)} tokens; need at least block_size + 2 = {block_size + 2}")
    high = len(data) - block_size - 1
    while True:
        ix = torch.randint(0, high + 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        yield x, y


def finetune(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    batch_size: int = 16,
    block_size: int = 64,
    generator: torch.Generator | None = None,
    weight_decay: float = 0.0,
) -> list[float]:
    """Train the *trainable* parameters of ``model`` on ``text`` with AdamW; return per-step losses.

    The model is put in train mode for the steps and returned to its previous
    mode afterwards. Frozen parameters are neither handed to the optimizer nor
    touched by it.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Nothing to train: every parameter is frozen.")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    batches = make_batches(text, tokenizer, block_size, batch_size, generator)
    was_training = model.training
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        x, y = next(batches)
        _logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.train(was_training)
    return losses


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    n_batches: int = 8,
    block_size: int = 64,
    batch_size: int = 16,
    generator: torch.Generator | None = None,
) -> float:
    """Mean cross-entropy over ``n_batches`` random batches, in eval mode, no gradients."""
    was_training = model.training
    model.eval()
    batches = make_batches(text, tokenizer, block_size, batch_size, generator)
    total = 0.0
    for _ in range(n_batches):
        x, y = next(batches)
        _logits, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / n_batches
