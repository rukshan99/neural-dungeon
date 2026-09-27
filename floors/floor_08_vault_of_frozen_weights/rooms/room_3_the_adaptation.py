"""ROOM 8.3 - THE ADAPTATION

    The goblin quartermaster's ledger is nothing like the chronicles: pipes,
    colons, prices in copper. The Chronicler reads it at a loss of 6.8, which
    is to say it cannot. You have sixty steps and the sigils. Teach it.

A fine-tuning loop is the pretraining loop with two changes: the weights start
from the checkpoint, and the optimizer is handed **only the trainable
parameters**. The second part is not cosmetic. AdamW keeps two moment tensors
per parameter it is given, so handing it frozen weights wastes memory; and a
weight frozen after a backward pass still carries a stale ``.grad`` that the
optimizer would step on. Give it nothing it must not touch.

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps):
        x, y = next(batches)
        _, loss = model(x, y)          # the GPT computes cross-entropy for you
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

Batches: random windows of ``block_size`` characters from the text, targets are
the same windows shifted one character right. Draw the window starts with
``torch.randint(..., generator=generator)`` so the trial can reproduce them.

Modes: the Chronicler was trained with dropout 0.1, so ``model.train()`` makes
it stochastic and ``model.eval()`` makes it deterministic. Train in train mode,
evaluate in eval mode, and put things back the way you found them.
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
    """An endless iterator of ``(x, y)`` batches of random windows from ``text``.

    Encode the text once. Each ``next()`` yields ``x`` and ``y`` of shape
    ``(batch_size, block_size)`` and dtype int64, where ``y[:, t] == x[:, t + 1]``.
    Window starts ``i`` must satisfy ``i + block_size + 1 <= len(data)``.
    (Write it as a generator function: a ``while True:`` with a ``yield``.)
    """
    raise NotImplementedError("make_batches() is unwritten")


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
    """Train the trainable parameters of ``model`` on ``text`` for ``steps`` steps with ``torch.optim.AdamW``.

    Return the per-step training losses as Python floats (``loss.item()``).
    Put the model in train mode for the steps and restore its previous mode
    afterwards. Only parameters with ``requires_grad=True`` go to the optimizer.
    """
    raise NotImplementedError("finetune() is unwritten")


def evaluate_loss(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    n_batches: int = 8,
    block_size: int = 64,
    batch_size: int = 16,
    generator: torch.Generator | None = None,
) -> float:
    """Mean cross-entropy over ``n_batches`` random batches of ``text``, as a float.

    Under ``torch.no_grad()``, in eval mode, restoring the model's mode after.
    Must leave no gradients and change no weights.
    """
    raise NotImplementedError("evaluate_loss() is unwritten")
