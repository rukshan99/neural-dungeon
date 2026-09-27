"""ROOM 4.2 - THE MODULE FORGE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Linear, act, Linear, act, ..., Linear over flattened inputs. Returns logits."""

    def __init__(self, sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()  # before any assignment, or registration silently fails
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.flatten(1)  # (B, ...) -> (B, in_features)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # no activation after the last layer
                h = self.act(h)
        return h


def count_parameters(model: nn.Module) -> int:
    """Trainable scalars only."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model: nn.Module) -> None:
    """Kaiming-normal weights and zero biases for every Linear, via apply()."""

    def init_linear(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    model.apply(init_linear)


def manual_forward(model: MLP, x: torch.Tensor) -> torch.Tensor:
    """The same computation as model(x), written with the parameters and matmul."""
    h = x.flatten(1)
    n = len(model.layers)
    for i, layer in enumerate(model.layers):
        h = h @ layer.weight.T + layer.bias  # Linear stores weight as (out, in)
        if i < n - 1:
            h = model.act(h)
    return h


def serialize(model: nn.Module) -> bytes:
    """state_dict -> bytes through an in-memory buffer."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_into(model: nn.Module, blob: bytes) -> nn.Module:
    """bytes -> state_dict -> model, strictly."""
    state = torch.load(io.BytesIO(blob), map_location="cpu")  # weights_only=True is the default
    model.load_state_dict(state)  # strict=True: a mismatched architecture raises
    return model


def freeze(model: nn.Module, prefix: str) -> int:
    """requires_grad=False for every parameter whose name starts with prefix."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad_(False)
            frozen += 1
    return frozen
