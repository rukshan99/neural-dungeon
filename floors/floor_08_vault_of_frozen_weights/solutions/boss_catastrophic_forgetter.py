"""BOSS - THE CATASTROPHIC FORGETTER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Forgetter is defeated by not giving it anything to erase:

  Phase 1  LoRA adapters (few weights move) trained on batches that *mix* the
           ledger with replayed chronicles (the loss still cares about the old
           text). The base weights are never touched.
  Phase 2  Adapters are a separate, small state. Save them, load them onto a
           pristine model, switch them off to get the original model back
           bit for bit.
  Phase 3  When you want a plain model with no adapter code, merge:
           W' = W + (alpha / r) B A, and export an ordinary state_dict.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer

from .room_2_the_low_rank_sigil import LoRALinear, inject_lora, lora_modules, lora_state_dict
from .room_3_the_adaptation import make_batches


def adapt_without_forgetting(
    model: nn.Module,
    tokenizer: CharTokenizer,
    new_text: str,
    old_text: str,
    steps: int,
    generator: torch.Generator | None = None,
    r: int = 8,
    alpha: float = 16.0,
    lr: float = 3e-3,
    batch_size: int = 16,
    block_size: int = 64,
    replay_fraction: float = 0.5,
) -> nn.Module:
    """A deep copy of ``model`` that speaks the new tongue and still remembers the old one.

    Recipe: freeze everything, inject LoRA, and train the adapters on batches
    that are half ``new_text`` and half replayed ``old_text``. The input model
    is not modified.
    """
    adapted = copy.deepcopy(model)
    for p in adapted.parameters():
        p.requires_grad_(False)
    inject_lora(adapted, r=r, alpha=alpha)
    params = [p for p in adapted.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    n_old = int(round(batch_size * replay_fraction))
    n_new = batch_size - n_old
    new_batches = make_batches(new_text, tokenizer, block_size, n_new, generator)
    old_batches = make_batches(old_text, tokenizer, block_size, n_old, generator)

    adapted.train()
    for _ in range(steps):
        x_new, y_new = next(new_batches)
        x_old, y_old = next(old_batches)
        x, y = torch.cat([x_new, x_old]), torch.cat([y_new, y_old])
        _logits, loss = adapted(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    adapted.eval()
    return adapted


class AdapterSwitch:
    """Adapters as a detachable component: save, load onto a pristine model, switch on and off."""

    @staticmethod
    def save_adapter(model: nn.Module) -> dict:
        """``{"r", "alpha", "targets", "weights"}``: everything needed to rebuild the adapters elsewhere."""
        modules = lora_modules(model)
        if not modules:
            raise ValueError("This model has no LoRA adapters to save.")
        first = modules[0][1]
        return {
            "r": first.r,
            "alpha": first.alpha,
            "targets": tuple(sorted({name.rsplit(".", 1)[-1] for name, _m in modules})),
            "weights": lora_state_dict(model),
        }

    @staticmethod
    def load_adapter(model: nn.Module, state: dict) -> nn.Module:
        """Copy saved adapter weights into ``model``, injecting adapters first if it has none."""
        if not lora_modules(model):
            for p in model.parameters():
                p.requires_grad_(False)
            inject_lora(model, r=state["r"], alpha=state["alpha"], targets=tuple(state["targets"]))
        own = dict(model.named_parameters())
        missing = [k for k in state["weights"] if k not in own]
        if missing:
            raise KeyError(f"Adapter tensors have nowhere to go: {missing[:3]}...")
        with torch.no_grad():
            for key, value in state["weights"].items():
                own[key].copy_(value)
        return model

    @staticmethod
    def disable_adapters(model: nn.Module) -> None:
        """Route every LoRALinear straight through its frozen base: the pristine model again."""
        for _n, m in lora_modules(model):
            m.enabled = False

    @staticmethod
    def enable_adapters(model: nn.Module) -> None:
        for _n, m in lora_modules(model):
            m.enabled = True


def merge_and_export(model: nn.Module) -> dict[str, torch.Tensor]:
    """A plain-GPT ``state_dict`` with every adapter folded into its Linear's weight.

    Works on a deep copy, so ``model`` keeps its adapters. Load the result into a
    fresh ``GPT(cfg)`` with ``strict=True``.
    """
    exported = copy.deepcopy(model)
    for parent_name, parent in list(exported.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.merge())
    return {k: v.detach().clone() for k, v in exported.state_dict().items()}


# ---------------------------------------------------------------------------
# PHASE 4: WHICH STRATEGIES SURVIVE THE FORGETTER?
# A strategy passes if, after the trial's fine-tune on the ledger, the ledger
# loss is <= LEDGER_GOAL and the chronicles loss rose by <= FORGETTING_BUDGET.
# ---------------------------------------------------------------------------
FORGETTER_PROPHECY: dict[str, bool | None] = {
    "full_lr_1e-3": False,
    "lora_r8_lr_3e-3": False,
    "full_lr_1e-3_replay_50": True,
    "lora_r8_lr_3e-3_replay_50": True,
}
