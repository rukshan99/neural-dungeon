"""ROOM 8.4 - THE FORGETTING

    The Chronicler speaks goblin now. Ask it about the chronicles. It hesitates.
    Ask again. Somewhere in the vault, something pale is smiling.

Catastrophic forgetting is not a bug in the optimizer; it is the optimizer
doing its job. The loss on the ledger has no term that cares about the
chronicles, so gradient descent will overwrite whatever it needs to. Measured
honestly it looks like this: old-text loss before and after, on the SAME
batches (otherwise sampling noise masquerades as forgetting).

Three standard mitigations, all of which you will now measure:

  1. Parameter-efficient tuning (LoRA): move fewer weights, in a low-rank
     subspace, so there is less to break. It limits the damage; it does not
     abolish it.
  2. A lower learning rate: move every weight, but less far. It forgets less
     and learns less; you trade one for the other.
  3. Replay (mixing): put old data back into every batch so the loss has a
     reason to keep the old behaviour. Usually the strongest of the three.

This room gives you three ways to produce a fine-tuned COPY (the original must
stay pristine), a report that measures the damage, and a prophecy to fill in
BEFORE you run the trial. Same steps, same seeds, same batches for all four.
"""

from __future__ import annotations

import copy  # noqa: F401

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_2_the_low_rank_sigil import inject_lora  # noqa: F401
from .room_3_the_adaptation import evaluate_loss, finetune, make_batches  # noqa: F401

STRATEGIES: tuple[str, ...] = (
    "full_lr_1e-3",
    "full_lr_1e-4",
    "lora_r8_lr_1e-3",
    "full_lr_1e-3_replay_50",
)


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
    """Loss on old and new text, before (``base_model``) and after (``adapted_model``).

    Keys: ``old_before, old_after, new_before, new_after, delta_old, delta_new``
    with ``delta_* = *_after - *_before``. Before and after MUST be evaluated on
    identical batches: draw one seed from ``generator`` and build a fresh
    ``torch.Generator().manual_seed(seed)`` for each of the four evaluations.
    """
    raise NotImplementedError("forgetting_report() is unwritten")


def full_finetune_copy(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Deep-copy ``model``, set EVERY parameter trainable, ``finetune`` the copy on ``text``, return it."""
    raise NotImplementedError("full_finetune_copy() is unwritten")


def lora_finetune_copy(
    model: nn.Module,
    text: str,
    tokenizer: CharTokenizer,
    steps: int,
    lr: float,
    r: int = 8,
    alpha: float = 16.0,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Deep-copy ``model``, freeze everything, ``inject_lora(r, alpha)``, ``finetune`` only the adapters, return the copy."""
    raise NotImplementedError("lora_finetune_copy() is unwritten")


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
    """Full fine-tune of a deep copy where every batch mixes both corpora.

    ``n_old = round(batch_size * replay_fraction)`` rows of each batch are windows
    of ``old_text`` (the replay), the other ``batch_size - n_old`` rows are windows
    of ``new_text``. Two ``make_batches`` streams and a ``torch.cat`` per step.
    All parameters trainable, AdamW, return the copy in eval mode.
    """
    raise NotImplementedError("replay_finetune_copy() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY OF FORGETTING
# The trial fine-tunes the Chronicler on the goblin ledger four ways, 30 steps
# each, and measures delta_old (the rise in chronicles loss) for each:
#
#     "full_lr_1e-3"            every weight, lr 1e-3
#     "full_lr_1e-4"            every weight, lr 1e-4
#     "lora_r8_lr_1e-3"         LoRA r=8, alpha=16, lr 1e-3 (same lr as the first)
#     "full_lr_1e-3_replay_50"  every weight, lr 1e-3, half of each batch is chronicles
#
# A strategy "forgets" when delta_old > FORGETTING_THRESHOLD. Predict FIRST.
# ---------------------------------------------------------------------------
FORGETTING_THRESHOLD = 0.35

FORGETTING_PROPHECY: dict = {
    "forgets_most": None,   # a strategy name
    "forgets_least": None,  # a strategy name
    "lora_forgets_less_than_full_at_same_lr": None,  # True / False
    "forgets": {
        "full_lr_1e-3": None,            # True / False
        "full_lr_1e-4": None,
        "lora_r8_lr_1e-3": None,
        "full_lr_1e-3_replay_50": None,
    },
}
