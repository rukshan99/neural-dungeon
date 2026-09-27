"""ROOM 8.2 - THE LOW-RANK SIGIL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

LoRA (Hu et al., 2021) keeps the pretrained weight W (out x in) frozen and adds
a trainable low-rank correction:

    W' = W + (alpha / r) * B @ A        A: (r, in)   B: (out, r)

B starts at zero, so at initialisation the wrapped layer is *exactly* the
original layer and training begins from the pretrained function. Because
r << min(in, out), the correction costs r * (in + out) weights instead of
in * out. Merging is just adding the product back into W: a plain nn.Linear
falls out, with no runtime overhead left behind.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS: tuple[str, ...] = ("c_attn", "c_proj", "fc", "proj")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` correction on the side."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        # Same init as the reference implementation: A random, B zero, so B @ A == 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The (out, in) correction ``scaling * B @ A`` that ``merge`` folds into W."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> nn.Linear:
        """A fresh plain ``nn.Linear`` computing the same function as this module (adapter on)."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, enabled={self.enabled}"


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every ``nn.Linear`` whose attribute name is in ``targets`` with a ``LoRALinear``.

    Returns the dotted names of the modules replaced, in ``named_modules`` order.
    Already-wrapped layers are left alone, so calling twice does not double-wrap.
    """
    replaced: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(parent, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return replaced


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    """``(name, module)`` for every LoRALinear in the model."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors (``...lora_A`` and ``...lora_B``), as detached clones."""
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.endswith("lora_A") or k.endswith("lora_B")
    }


def lora_parameter_count(model: nn.Module) -> int:
    """How many scalar weights live in the adapters: sum of r * (in + out) over wrapped layers."""
    return sum(m.lora_A.numel() + m.lora_B.numel() for _n, m in lora_modules(model))
