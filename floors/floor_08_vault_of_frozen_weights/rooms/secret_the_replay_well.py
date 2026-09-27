"""SECRET - THE REPLAY WELL   (optional)

    Under the Forgetter's antechamber, a well. Drop a memory in and it comes
    back up on schedule, exactly as often as you asked. Carved on the rim:
    "F_i (w_i - w*_i)^2". Someone down here fought forgetting with a spring.

Two more mitigations, each with an exact contract:

REPLAY ON A SCHEDULE. Room 4 mixed rows inside a batch. Here whole batches
alternate between corpora on a fixed ratio: batch ``i`` (from 0) is a replay
batch exactly when ``floor((i + 1) * ratio) > floor(i * ratio)``, so among any
first N batches exactly ``floor(N * ratio)`` are replay. No coin flips.

ELASTIC WEIGHT CONSOLIDATION (Kirkpatrick et al., 2017). After training on the
old task, estimate how much each weight mattered with the diagonal Fisher
information, approximated from gradients on old data:

    F_i = mean over batches of (dL/dw_i)^2          (the "empirical Fisher")

Then, while learning the new task, add a quadratic penalty that pulls every
weight back toward its old value ``w*``, harder where F is large:

    penalty = (lam / 2) * sum_i F_i * (w_i - w*_i)^2

Weights that carried the old knowledge become stiff springs; weights that did
not are free to move. On this model F averages about 1e-6, so ``lam`` has to be
around 1e5 before the penalty competes with the task loss.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_3_the_adaptation import make_batches  # noqa: F401


def mixed_batches(
    new_text: str,
    old_text: str,
    ratio: float,
    tokenizer: CharTokenizer,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
    """An endless iterator of ``(x, y, source)``; ``source`` is ``"old"`` or ``"new"``.

    Build one ``make_batches`` stream per corpus and pull from the right one
    according to the schedule above. ``ratio=0`` never replays, ``ratio=1``
    always does. Raise ValueError outside [0, 1].
    """
    raise NotImplementedError("mixed_batches() is unwritten")


def fisher_diagonal(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    n_batches: int = 8,
    block_size: int = 64,
    batch_size: int = 16,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """``{name: mean of grad**2 over n_batches}`` for every trainable parameter, same shapes.

    Evaluate in eval mode (no dropout), ``zero_grad`` before each backward, and
    leave no gradients behind. The returned tensors are detached.
    """
    raise NotImplementedError("fisher_diagonal() is unwritten")


def ewc_penalty(
    model: nn.Module,
    snap: dict[str, torch.Tensor],
    fisher_diag: dict[str, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """``(lam / 2) * sum_i F_i * (w_i - w*_i)^2`` as a scalar tensor attached to the graph.

    ``snap`` is a Room 1 snapshot (``w*``). Sum over the parameters present in
    ``fisher_diag``. Exactly zero at the snapshot.
    """
    raise NotImplementedError("ewc_penalty() is unwritten")


def finetune_with_ewc(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    snap: dict[str, torch.Tensor],
    fisher_diag: dict[str, torch.Tensor],
    lam: float,
    batch_size: int = 16,
    block_size: int = 64,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Full fine-tune on ``text`` with ``ewc_penalty`` added to the loss at every step.

    Return the per-step TASK losses (without the penalty) as floats. ``lam=0``
    must behave exactly like a plain fine-tune. Restore the model's mode after.
    """
    raise NotImplementedError("finetune_with_ewc() is unwritten")
