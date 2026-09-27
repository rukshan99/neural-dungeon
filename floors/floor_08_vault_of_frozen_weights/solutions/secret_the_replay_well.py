"""SECRET - THE REPLAY WELL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Two more ways to make the loss remember:

* Replay with an exact schedule. Instead of mixing rows inside a batch, whole
  batches alternate between corpora on a fixed ratio, so among any first N
  batches exactly floor(N * ratio) come from the old text.
* Elastic Weight Consolidation (Kirkpatrick et al., 2017). Estimate how much
  each weight mattered to the old task with the diagonal Fisher information
  F_i = E[(dL/dw_i)^2] on old data, then penalise moving it:

      penalty = (lam / 2) * sum_i F_i * (w_i - w*_i)^2

  Important weights become stiff springs; unimportant ones stay free.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_3_the_adaptation import make_batches


def mixed_batches(
    new_text: str,
    old_text: str,
    ratio: float,
    tokenizer: CharTokenizer,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
    """An endless stream of ``(x, y, source)`` batches; ``source`` is ``"old"`` or ``"new"``.

    Batch ``i`` (counting from 0) is an old-text batch exactly when
    ``floor((i + 1) * ratio) > floor(i * ratio)``, so among the first N batches
    exactly ``floor(N * ratio)`` are replay. ``ratio=0`` never replays,
    ``ratio=1`` always does.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    new_stream = make_batches(new_text, tokenizer, block_size, batch_size, generator)
    old_stream = make_batches(old_text, tokenizer, block_size, batch_size, generator)
    i = 0
    while True:
        is_old = int((i + 1) * ratio + 1e-9) > int(i * ratio + 1e-9)
        x, y = next(old_stream) if is_old else next(new_stream)
        yield x, y, ("old" if is_old else "new")
        i += 1


def fisher_diagonal(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    n_batches: int = 8,
    block_size: int = 64,
    batch_size: int = 16,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """``{name: E[grad^2]}`` over ``n_batches`` of ``text`` for every trainable parameter."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fisher = {n: torch.zeros_like(p) for n, p in params}
    batches = make_batches(text, tokenizer, block_size, batch_size, generator)
    was_training = model.training
    model.eval()
    for _ in range(n_batches):
        x, y = next(batches)
        model.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        loss.backward()
        for n, p in params:
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 / n_batches
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return fisher


def ewc_penalty(
    model: nn.Module,
    snap: dict[str, torch.Tensor],
    fisher_diag: dict[str, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """``(lam / 2) * sum_i F_i (w_i - w*_i)^2`` over the parameters present in ``fisher_diag``.

    Returns a scalar tensor attached to the graph so it can be added to the loss.
    """
    total = torch.zeros(())
    for name, param in model.named_parameters():
        if name in fisher_diag and name in snap:
            total = total + (fisher_diag[name] * (param - snap[name]) ** 2).sum()
    return 0.5 * lam * total


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
    """Full fine-tune on ``text`` with the EWC penalty added to every step's loss.

    Returns the per-step *task* losses (without the penalty) so they compare
    directly with ``finetune``. ``lam=0`` is plain fine-tuning.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    batches = make_batches(text, tokenizer, block_size, batch_size, generator)
    was_training = model.training
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        x, y = next(batches)
        _logits, loss = model(x, y)
        total = loss + ewc_penalty(model, snap, fisher_diag, lam)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        losses.append(loss.item())
    model.train(was_training)
    return losses
