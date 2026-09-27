"""ROOM 4.5 - THE DEVICE FERRY  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer: str | None = None) -> torch.device:
    """An override is trusted; otherwise cuda > mps > cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Recurse through lists, tuples and dicts; move tensors; leave everything else alone."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def ensure_float32(batch: Any) -> Any:
    """Floating tensors and arrays -> float32; integers untouched; same recursion as to_device."""
    if isinstance(batch, np.ndarray):
        batch = torch.as_tensor(batch)
    if isinstance(batch, torch.Tensor):
        return batch.float() if batch.is_floating_point() else batch
    if isinstance(batch, list):
        return [ensure_float32(b) for b in batch]
    if isinstance(batch, tuple):
        return tuple(ensure_float32(b) for b in batch)
    if isinstance(batch, dict):
        return {k: ensure_float32(v) for k, v in batch.items()}
    return batch


def model_device(model: nn.Module) -> torch.device:
    """Parameters first, buffers second, otherwise the module has no device."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    raise ValueError("This module owns no parameters or buffers, so it lives nowhere in particular.")
