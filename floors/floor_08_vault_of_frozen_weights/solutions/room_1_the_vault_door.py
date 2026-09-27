"""ROOM 8.1 - THE VAULT DOOR  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A checkpoint is a dict of tensors plus the config needed to rebuild the module
tree that owns them. ``requires_grad`` is the per-tensor switch that decides
whether autograd records operations on it: a frozen parameter never receives a
``.grad`` and the optimizer therefore never moves it. Everything else on this
floor (snapshot, changed_parameters) is bookkeeping for *proving* that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dungeon.artifacts.tiny_gpt import CharTokenizer, load_pretrained


def load_chronicler() -> tuple[nn.Module, CharTokenizer]:
    """The pretrained Chronicler and its tokenizer, in eval mode, every parameter trainable."""
    model, tokenizer, _extra = load_pretrained()
    return model, tokenizer


def freeze(model: nn.Module, patterns: list[str] | tuple[str, ...] | None = None) -> int:
    """Set ``requires_grad=False`` on every parameter whose name contains any pattern.

    ``patterns=None`` freezes everything. Returns the number of parameter tensors
    that matched (and are therefore frozen after the call).
    """
    frozen = 0
    for name, param in model.named_parameters():
        if patterns is None or any(p in name for p in patterns):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """``(name, parameter)`` pairs with ``requires_grad=True``, in ``named_parameters`` order."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total number of scalar weights (``numel``), optionally only the trainable ones.

    ``named_parameters()`` yields tied weights once, so the Chronicler's tied
    ``lm_head``/``wte`` matrix is counted a single time.
    """
    return sum(p.numel() for _n, p in model.named_parameters() if p.requires_grad or not trainable_only)


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """``{name: detached clone}`` of every parameter, so later edits cannot reach it."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def changed_parameters(snap: dict[str, torch.Tensor], model: nn.Module, atol: float = 0.0) -> list[str]:
    """Names of parameters that differ from ``snap`` by more than ``atol`` anywhere.

    A parameter that is new (not in the snapshot), missing (in the snapshot but
    not in the model) or of a different shape counts as changed. ``atol=0`` means
    bit-for-bit equality is required.
    """
    changed: list[str] = []
    current = dict(model.named_parameters())
    for name, param in current.items():
        before = snap.get(name)
        if before is None or before.shape != param.shape:
            changed.append(name)
            continue
        if not torch.all((param.detach() - before).abs() <= atol):
            changed.append(name)
    changed.extend(name for name in snap if name not in current)
    return changed
