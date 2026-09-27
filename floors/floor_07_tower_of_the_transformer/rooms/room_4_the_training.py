"""ROOM 7.4 - THE TRAINING

    A tower that has never read anything predicts every character with the
    same shrug: loss ln(72) = 4.28. Feed it the dungeon's own chronicles for
    a hundred and twenty steps and listen to the shrug turn into a guess.

BATCHES. A language model's training data is the text itself. Pick random
windows of ``block_size`` tokens as ``x``; the target ``y`` is the same
window shifted one token to the right, so y[b, t] = x[b, t + 1]: at every
position the model predicts the NEXT token. A start index i is valid when
i + block_size < len(data), so that y has a last token to point at.

THE SCHEDULE. Warm up linearly, then decay along a cosine:

    step < warmup:   lr_max * (step + 1) / warmup        step warmup-1 is lr_max
    step >= total:   lr_min
    otherwise:       progress = (step - warmup) / (total - warmup)
                     lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))

Warmup protects the fresh Adam moment estimates from a large first step;
the cosine lets the model settle into a minimum instead of bouncing around it.

THE LOOP, every step:

    set every optimizer param_group's "lr" from lr_schedule(step, ...)
    x, y = get_batch(...)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

Clipping rescales the whole gradient vector so its L2 norm is at most
``grad_clip``; one bad batch cannot fling the weights across the landscape.
Build the optimizer as ``torch.optim.AdamW(model.parameters(), lr=lr,
betas=(0.9, 0.95), weight_decay=0.1)`` (the trial swaps that class for a spy
that checks the norm and the learning rate at each step). Warmup length is
yours; a tenth of the steps is customary. lr_min = lr / 10 is customary too.
"""

from __future__ import annotations

import torch


def encode_corpus(text: str, tokenizer) -> torch.Tensor:
    """The whole text as one 1-D ``torch.long`` tensor of token ids (``tokenizer.encode``)."""
    raise NotImplementedError("encode_corpus() is unwritten")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator):
    """Random windows: x (B, T) and y (B, T) with y[b, t] = x[b, t + 1].

    Draw the start indices with ``torch.randint(..., generator=generator)``
    so that the batch is reproducible from the seed.
    """
    raise NotImplementedError("get_batch() is unwritten")


def lr_schedule(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup then cosine decay; see the module docstring for the exact formula."""
    raise NotImplementedError("lr_schedule() is unwritten")


def train(
    model,
    data: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    generator: torch.Generator,
    grad_clip: float = 1.0,
) -> list[float]:
    """AdamW for ``steps`` steps with the schedule and clipping. Returns every step's loss as a float.

    ``model.cfg.block_size`` is the window length. Put the model in
    ``train()`` mode first.
    """
    raise NotImplementedError("train() is unwritten")


def estimate_loss(model, data: torch.Tensor, batches: int, generator: torch.Generator, batch_size: int = 32) -> float:
    """Mean loss over ``batches`` batches drawn with ``get_batch``, in eval mode, under ``torch.no_grad()``.

    Restore the model's previous train/eval mode before returning.
    """
    raise NotImplementedError("estimate_loss() is unwritten")
