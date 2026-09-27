"""ROOM 7.4 - THE TRAINING  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Turning text into shifted (x, y) batches, a warmup-then-cosine learning-rate
schedule, and a training loop with AdamW and gradient clipping.
"""

from __future__ import annotations

import math

import torch


def encode_corpus(text: str, tokenizer) -> torch.Tensor:
    """The whole corpus as one 1-D int64 tensor of token ids."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator):
    """Random windows of ``data``: x is (B, T) tokens, y is the same windows shifted one step right.

    Every start index i satisfies i + block_size < len(data), so y never runs
    off the end of the corpus.
    """
    ix = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def lr_schedule(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup to lr_max over ``warmup`` steps, then cosine decay to lr_min at ``total``.

    step < warmup:   lr_max * (step + 1) / warmup          (step warmup-1 -> lr_max)
    step >= total:   lr_min
    otherwise:       progress = (step - warmup) / (total - warmup)
                     lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
    """
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train(
    model,
    data: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    generator: torch.Generator,
    grad_clip: float = 1.0,
) -> list[float]:
    """Train ``model`` on ``data`` for ``steps`` AdamW steps. Returns the loss at every step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, steps // 10)
    block_size = model.cfg.block_size
    model.train()
    losses: list[float] = []
    for step in range(steps):
        lr_now = lr_schedule(step, warmup, steps, lr, lr / 10)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        x, y = get_batch(data, block_size, batch_size, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, batches: int, generator: torch.Generator, batch_size: int = 32) -> float:
    """Mean loss over ``batches`` random batches, in eval mode, with no gradients."""
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, model.cfg.block_size, batch_size, generator)
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / batches
