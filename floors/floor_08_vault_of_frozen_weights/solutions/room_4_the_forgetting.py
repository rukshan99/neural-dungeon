"""ROOM 8.4 - THE FORGETTING  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Catastrophic forgetting: gradient descent on the new data has no term that
cares about the old data, so it will happily overwrite whatever it needs to.
The three standard mitigations, each of which appears below or in the boss:

  1. parameter-efficient tuning  - move fewer weights (LoRA), less to break;
  2. a lower learning rate        - move every weight, but less far;
  3. replay / mixing              - put old data back into the batches, so the
                                    loss *does* have a term that cares.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_2_the_low_rank_sigil import inject_lora
from .room_3_the_adaptation import evaluate_loss, finetune, make_batches

STRATEGIES: tuple[str, ...] = (
    "full_lr_1e-3",
    "full_lr_1e-4",
    "lora_r8_lr_1e-3",
    "full_lr_1e-3_replay_50",
)


def _fresh_seed(generator: torch.Generator | None) -> int:
    """Draw one seed from ``generator`` so several evaluations can share identical batches."""
    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())


def forgetting_report(
    base_model: nn.Module,
    adapted_model: nn.Module,
    tokenizer: CharTokenizer,
    old_text: str,
    new_text: str,
    generator: torch.Generator | None = None,
    n_batches: int = 8,
    block_size: int = 64,
) -> dict[str, float]:
    """Loss on the old and new corpora before (base) and after (adapted), on identical batches.

    Keys: ``old_before, old_after, new_before, new_after, delta_old, delta_new``
    where ``delta_* = *_after - *_before``. A positive ``delta_old`` is forgetting.
    """
    seed = _fresh_seed(generator)

    def loss_of(model: nn.Module, text: str) -> float:
        gen = torch.Generator().manual_seed(seed)
        return evaluate_loss(model, text, tokenizer, n_batches, block_size, generator=gen)

    report = {
        "old_before": loss_of(base_model, old_text),
        "old_after": loss_of(adapted_model, old_text),
        "new_before": loss_of(base_model, new_text),
        "new_after": loss_of(adapted_model, new_text),
    }
    report["delta_old"] = report["old_after"] - report["old_before"]
    report["delta_new"] = report["new_after"] - report["new_before"]
    return report


def full_finetune_copy(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    generator: torch.Generator | None = None,
    **finetune_kwargs,
) -> nn.Module:
    """Deep-copy ``model``, make *every* parameter trainable, fine-tune the copy on ``text``."""
    copy_ = copy.deepcopy(model)
    for p in copy_.parameters():
        p.requires_grad_(True)
    finetune(copy_, text, tokenizer, steps, lr, generator=generator, **finetune_kwargs)
    return copy_


def lora_finetune_copy(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    r: int = 8,
    alpha: float = 16.0,
    generator: torch.Generator | None = None,
    **finetune_kwargs,
) -> nn.Module:
    """Deep-copy ``model``, freeze it, inject LoRA (rank ``r``) and fine-tune only the adapters."""
    copy_ = copy.deepcopy(model)
    for p in copy_.parameters():
        p.requires_grad_(False)
    inject_lora(copy_, r=r, alpha=alpha)
    finetune(copy_, text, tokenizer, steps, lr, generator=generator, **finetune_kwargs)
    return copy_


def replay_finetune_copy(
    model: nn.Module,
    new_text: str,
    old_text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    replay_fraction: float = 0.5,
    batch_size: int = 16,
    block_size: int = 64,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Full fine-tune of a deep copy where each batch mixes rows from both corpora.

    ``round(batch_size * replay_fraction)`` rows of every batch are windows of
    ``old_text`` (the replay), the rest are windows of ``new_text``.
    """
    copy_ = copy.deepcopy(model)
    for p in copy_.parameters():
        p.requires_grad_(True)
    n_old = int(round(batch_size * replay_fraction))
    n_new = batch_size - n_old
    if not 0 < n_new <= batch_size:
        raise ValueError(f"replay_fraction={replay_fraction} leaves {n_new} new rows per batch")
    new_batches = make_batches(new_text, tokenizer, block_size, n_new, generator)
    old_batches = make_batches(old_text, tokenizer, block_size, max(n_old, 1), generator)

    opt = torch.optim.AdamW(copy_.parameters(), lr=lr)
    copy_.train()
    for _ in range(steps):
        x_new, y_new = next(new_batches)
        if n_old > 0:
            x_old, y_old = next(old_batches)
            x, y = torch.cat([x_new, x_old]), torch.cat([y_new, y_old])
        else:
            x, y = x_new, y_new
        _logits, loss = copy_(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    copy_.eval()
    return copy_


# ---------------------------------------------------------------------------
# THE PROPHECY OF FORGETTING
# Four strategies, same steps, same seed, fine-tuned on the goblin ledger.
# "forgets" means the chronicles loss rose by more than FORGETTING_THRESHOLD.
# ---------------------------------------------------------------------------
FORGETTING_THRESHOLD = 0.35

FORGETTING_PROPHECY: dict = {
    "forgets_most": "full_lr_1e-3",
    "forgets_least": "full_lr_1e-3_replay_50",
    "lora_forgets_less_than_full_at_same_lr": True,
    "forgets": {
        "full_lr_1e-3": True,
        "full_lr_1e-4": True,
        "lora_r8_lr_1e-3": True,
        "full_lr_1e-3_replay_50": False,
    },
}
