"""ROOM 8.2 - THE LOW-RANK SIGIL

    You cannot carve new runes into a frozen door. But you can draw a sigil on
    the ice in front of it, and the sigil can be anything you like, as long as
    it starts out invisible.

LoRA (Low-Rank Adaptation) leaves a pretrained weight ``W`` (out x in) frozen
and adds a trainable correction of rank ``r``:

    y = W x + b  +  (alpha / r) * B (A x)          A: (r, in)    B: (out, r)

``A`` is random, ``B`` is **zero**, so at initialisation ``B A = 0`` and the
wrapped layer computes exactly the original layer: fine-tuning starts from the
pretrained function, not from a perturbed one. Only ``A`` and ``B`` train.
They hold ``r * (in + out)`` weights instead of ``in * out``: for a 128 -> 512
layer with r = 4 that is 2,560 instead of 65,536.

``alpha / r`` is a fixed scale, not a parameter. Holding ``alpha`` fixed while
you change ``r`` keeps the size of the update roughly constant, which is why it
exists at all. A common default is ``alpha = 2 * r``.

Merging: ``W' = W + (alpha / r) * B @ A`` is an ordinary weight matrix. Fold it
in and the adapter disappears, leaving a plain ``nn.Linear`` with no runtime
cost. Keep it separate and you can switch it off, swap it, or ship it as a
tiny file next to a shared base. The boss uses both.

Where to put sigils: on the ``nn.Linear`` layers inside the blocks. The
Chronicler's are ``c_attn`` (fused q, k, v), ``c_proj``, ``fc`` and ``proj``,
16 in all. ``lm_head`` is not a target (its weight is tied to ``wte``).
"""

from __future__ import annotations

import math  # noqa: F401

import torch
import torch.nn as nn

DEFAULT_TARGETS: tuple[str, ...] = ("c_attn", "c_proj", "fc", "proj")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` (``self.base``) plus a trainable rank-``r`` correction.

    Attributes the trials and the boss rely on:
      base      the wrapped nn.Linear, frozen (requires_grad=False on weight and bias)
      lora_A    nn.Parameter of shape (r, in_features), random init (e.g.
                ``nn.init.kaiming_uniform_(A, a=math.sqrt(5))`` or normal with std 1/r)
      lora_B    nn.Parameter of shape (out_features, r), initialised to ZEROS
      scaling   alpha / r (a float, not a parameter)
      dropout   nn.Dropout(p) on the INPUT of the LoRA path only (nn.Identity when p == 0)
      enabled   bool, True by default; when False, forward returns base(x) exactly
      r, alpha  kept for bookkeeping (save/load needs them)
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("LoRALinear.__init__() is unwritten")

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``base(x) + (dropout(x) @ lora_A.T @ lora_B.T) * scaling``, or just ``base(x)`` when disabled.

        ``x`` may have any leading shape (..., in_features); the result is (..., out_features).
        """
        raise NotImplementedError("LoRALinear.forward() is unwritten")

    def delta_weight(self) -> torch.Tensor:
        """The (out_features, in_features) correction ``scaling * lora_B @ lora_A``."""
        raise NotImplementedError("LoRALinear.delta_weight() is unwritten")

    def merge(self) -> nn.Linear:
        """A NEW plain ``nn.Linear`` computing the same function as this layer with the adapter on.

        weight = base.weight + delta_weight(); bias = base.bias (or none if the base has none).
        Do not modify ``self.base`` in place.
        """
        raise NotImplementedError("LoRALinear.merge() is unwritten")


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> list[str]:
    """Replace, in place, every ``nn.Linear`` whose attribute name is in ``targets`` with a ``LoRALinear``.

    Walk ``model.named_modules()``; for each parent, look at ``named_children()``
    and ``setattr(parent, child_name, LoRALinear(child, r, alpha, dropout))``
    when the child is an ``nn.Linear`` and ``child_name in targets``. Return the
    dotted names of the replaced modules (e.g. ``"blocks.0.attn.c_attn"``) in
    the order they were met. Do not wrap a LoRALinear twice.
    """
    raise NotImplementedError("inject_lora() is unwritten")


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    """``[(name, module), ...]`` for every ``LoRALinear`` in ``model``, in ``named_modules`` order."""
    raise NotImplementedError("lora_modules() is unwritten")


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors: keys ending in ``lora_A`` or ``lora_B``, values detached clones.

    This is the file you would ship alongside a shared base checkpoint.
    """
    raise NotImplementedError("lora_state_dict() is unwritten")


def lora_parameter_count(model: nn.Module) -> int:
    """Total scalar weights in all adapters: ``sum(r * (in + out))`` over wrapped layers. 0 if none."""
    raise NotImplementedError("lora_parameter_count() is unwritten")
