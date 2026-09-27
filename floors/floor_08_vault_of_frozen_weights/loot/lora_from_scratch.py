"""LoRA from scratch: loot from Floor 8 of the Neural Dungeon.

Three functions' worth of code that turns any PyTorch model with ``nn.Linear``
layers into a parameter-efficiently fine-tunable one, and back.

    model = ...                                   # any nn.Module, pretrained
    freeze(model)                                 # every parameter requires_grad=False
    names = inject_lora(model, r=8, alpha=16, targets=("q_proj", "v_proj"))
    params = [p for p in model.parameters() if p.requires_grad]   # only lora_A / lora_B
    opt = torch.optim.AdamW(params, lr=2e-4)
    ...train...
    torch.save({"config": {"r": 8, "alpha": 16, "targets": [...]},
                "weights": lora_state_dict(model)}, "adapter.pt")    # kilobytes, not gigabytes
    merge_lora(model)                             # plain nn.Linear layers again, zero overhead

Only torch is required. Nothing here depends on the dungeon.

The math (Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"):

    y = W x + b + (alpha / r) * B (A x)      W: (out, in) frozen
                                            A: (r, in)   random init
                                            B: (out, r)  ZERO init  ->  B A == 0 at step 0
    merged:  W' = W + (alpha / r) * B @ A   bias unchanged
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` correction on the side.

    ``enabled=False`` routes straight through the base layer, which gives you the
    pristine model back at runtime without reloading anything.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got r={r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # the door stays frozen
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r                     # a constant, not a parameter
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init nn.Linear uses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        # (..., in) @ (in, r) -> (..., r) @ (r, out) -> (..., out). Never materialise B @ A here:
        # two small matmuls are cheaper than one (out, in) product per forward.
        return out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The (out, in) correction that merge() folds into the base weight."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> nn.Linear:
        """A new plain nn.Linear that computes exactly this layer with the adapter on."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device, self.base.weight.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, enabled={self.enabled}"


def freeze(model: nn.Module) -> None:
    """requires_grad=False on every parameter. Do this BEFORE injecting."""
    for p in model.parameters():
        p.requires_grad_(False)


def _replace_children(model: nn.Module, should_replace, make):
    """Walk the tree; for each child that ``should_replace(name, child)``, ``setattr`` ``make(child)``."""
    replaced: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if should_replace(child_name, child):
                setattr(parent, child_name, make(child))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return replaced


def inject_lora(model: nn.Module, r: int, alpha: float, targets: tuple[str, ...], dropout: float = 0.0) -> list[str]:
    """Replace every nn.Linear whose attribute name is in ``targets`` with a LoRALinear.

    Returns the dotted names replaced. Typical targets: ("q_proj", "v_proj") for
    Llama-style models, ("c_attn",) for GPT-2-style fused qkv, plus the MLP
    projections if you can afford them. Never wrap a layer whose weight is tied
    to an embedding (GPT-2's lm_head).
    """
    return _replace_children(
        model,
        lambda name, child: isinstance(child, nn.Linear) and name in targets,
        lambda child: LoRALinear(child, r, alpha, dropout),
    )


def merge_lora(model: nn.Module) -> list[str]:
    """Replace every LoRALinear with its merged plain nn.Linear, in place. Returns the names."""
    return _replace_children(model, lambda _n, child: isinstance(child, LoRALinear), lambda child: child.merge())


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors, as detached clones. Save these; the base ships separately."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith(("lora_A", "lora_B"))}


def load_lora_state_dict(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Copy saved adapter tensors into an already-injected model (same r, same targets)."""
    own = dict(model.named_parameters())
    missing = [k for k in weights if k not in own]
    if missing:
        raise KeyError(f"no parameter for adapter tensors {missing[:3]}... inject_lora first, with the same r and targets")
    with torch.no_grad():
        for k, v in weights.items():
            own[k].copy_(v)


def set_adapters_enabled(model: nn.Module, enabled: bool) -> None:
    """Switch every adapter on or off at runtime. Off == the pristine base model."""
    for _n, m in lora_modules(model):
        m.enabled = enabled


def lora_parameter_count(model: nn.Module) -> int:
    return sum(m.lora_A.numel() + m.lora_B.numel() for _n, m in lora_modules(model))


if __name__ == "__main__":
    # Smoke test on a toy: inject, check nothing changed, train one step, merge, check equality.
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8))
    x = torch.randn(4, 32)
    before = net(x)
    freeze(net)
    print("injected:", inject_lora(net, r=4, alpha=8, targets=("0", "2")))
    assert torch.equal(net(x), before), "B is zero: the function must be unchanged"
    trainable = [p for p in net.parameters() if p.requires_grad]
    print("trainable tensors:", len(trainable), "weights:", lora_parameter_count(net))
    with torch.no_grad():
        for _n, m in lora_modules(net):
            m.lora_B.normal_()                             # pretend we trained
    adapted = net(x)
    merge_lora(net)
    assert not lora_modules(net), "plain Linears after merge"
    assert torch.allclose(net(x), adapted, atol=1e-5), "merge must reproduce the adapted function"
    print("merged ok; max diff", (net(x) - adapted).abs().max().item())
