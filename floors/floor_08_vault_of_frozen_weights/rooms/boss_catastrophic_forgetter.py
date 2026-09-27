"""BOSS - THE CATASTROPHIC FORGETTER

                 .  .   .    .
              .  ( \\ | / )  .          "Teach them anything you like.
            .    (  o  o  )    .          I will keep what they lose."
           .   . (   __   ) .   .
             . . .`-----'. . .          It drifts between the shelves of the
            .  :  :  :  :  :  .         vault touching checkpoints. Where its
           :   :  :  :  :  :   :        fingers pass, the models forget. Every
          :    :  :  :  :  :    :       gradient step you take on the ledger is
               `--'  `--'  `--'         a door it walks through.

WEAKNESS: parameter-efficient tuning (few weights move), replay (the loss still
cares about the old text), and adapters that can be switched off, swapped, or
merged, so that the base weights are never touched at all.

The fight has four phases:

  Phase 1  adapt_without_forgetting  - ledger loss <= 1.6 AND chronicles loss
           within 0.35 of the pristine model, in 25 steps. A naive full
           fine-tune fails this by a mile; LoRA alone fails it too. LoRA plus
           replaying chronicles rows in every batch passes. The returned model
           must carry LoRALinear adapters (phases 2 and 3 depend on it).
  Phase 2  AdapterSwitch - save the adapter as its own small state, load it
           onto a pristine copy, disable it (pristine outputs again, to 1e-6),
           enable it (the ledger is back).
  Phase 3  merge_and_export - fold every sigil into its Linear and return a
           state_dict a fresh GPT(cfg) loads with strict=True.
  Phase 4  FORGETTER_PROPHECY - which strategies satisfy both constraints?

Run:  dungeon fight 8
"""

from __future__ import annotations

import copy  # noqa: F401

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_2_the_low_rank_sigil import (  # noqa: F401
    LoRALinear,
    inject_lora,
    lora_modules,
    lora_state_dict,
)
from .room_3_the_adaptation import make_batches  # noqa: F401


def adapt_without_forgetting(
    model: nn.Module,
    tokenizer: CharTokenizer,
    new_text: str,
    old_text: str,
    steps: int,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Return a NEW model that reads ``new_text`` and still reads ``old_text``.

    ``model`` must come back untouched (deep-copy it). The result must contain
    LoRALinear adapters over the frozen base and be in eval mode. Reference
    recipe: freeze, inject_lora(r=8, alpha=16), AdamW at lr 3e-3 over the
    adapter parameters, batches of 16 x 64 with 8 rows from new_text and 8
    from old_text every step.
    """
    raise NotImplementedError("adapt_without_forgetting() is unwritten")


class AdapterSwitch:
    """Adapters as a detachable component. All methods are static."""

    @staticmethod
    def save_adapter(model: nn.Module) -> dict:
        """``{"r": int, "alpha": float, "targets": tuple[str, ...], "weights": lora_state_dict(model)}``.

        ``targets`` are the attribute names of the wrapped layers (e.g. ``("c_attn", "c_proj", "fc", "proj")``),
        so that ``load_adapter`` can re-inject on a pristine model.
        """
        raise NotImplementedError("AdapterSwitch.save_adapter() is unwritten")

    @staticmethod
    def load_adapter(model: nn.Module, state: dict) -> nn.Module:
        """Copy the saved adapter weights into ``model`` and return it.

        If ``model`` has no LoRALinear modules yet, freeze it and ``inject_lora``
        with the saved r, alpha and targets first. Then copy every tensor of
        ``state["weights"]`` into the parameter of the same name (under ``torch.no_grad()``).
        """
        raise NotImplementedError("AdapterSwitch.load_adapter() is unwritten")

    @staticmethod
    def disable_adapters(model: nn.Module) -> None:
        """Set ``enabled = False`` on every LoRALinear: the model computes exactly the pristine base."""
        raise NotImplementedError("AdapterSwitch.disable_adapters() is unwritten")

    @staticmethod
    def enable_adapters(model: nn.Module) -> None:
        """Set ``enabled = True`` on every LoRALinear."""
        raise NotImplementedError("AdapterSwitch.enable_adapters() is unwritten")


def merge_and_export(model: nn.Module) -> dict[str, torch.Tensor]:
    """A plain-GPT ``state_dict`` with every adapter merged into its Linear's weight.

    Work on a deep copy (the input keeps its adapters): replace each LoRALinear
    with ``child.merge()`` via ``setattr`` on its parent, then return the copy's
    ``state_dict()`` with detached, cloned tensors. The keys must be exactly
    those of a fresh ``GPT(cfg).state_dict()``.
    """
    raise NotImplementedError("merge_and_export() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 4: WHICH STRATEGIES SURVIVE THE FORGETTER?
# The trial fine-tunes the Chronicler on the ledger four ways for 25 steps and
# checks BOTH constraints: ledger loss <= 1.6 and chronicles loss rise <= 0.35.
# True = survives (both hold), False = devoured. Predict first.
#
#     "full_lr_1e-3"               every weight, lr 1e-3
#     "lora_r8_lr_3e-3"            LoRA r=8, alpha=16, lr 3e-3, ledger only
#     "full_lr_1e-3_replay_50"     every weight, lr 1e-3, half of each batch is chronicles
#     "lora_r8_lr_3e-3_replay_50"  LoRA r=8, lr 3e-3, half of each batch is chronicles
# ---------------------------------------------------------------------------
FORGETTER_PROPHECY: dict[str, bool | None] = {
    "full_lr_1e-3": None,
    "lora_r8_lr_3e-3": None,
    "full_lr_1e-3_replay_50": None,
    "lora_r8_lr_3e-3_replay_50": None,
}
