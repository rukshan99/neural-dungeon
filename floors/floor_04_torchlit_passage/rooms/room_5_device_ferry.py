"""ROOM 4.5 - THE DEVICE FERRY

    A black river crosses the passage. On the far bank the torches burn white
    and a thousand times brighter; that bank is called ``cuda`` and you may not
    have one. The ferryman does not care which bank you are on. He cares that
    everything in the boat gets off on the same side.

Every tensor lives on a *device*: ``cpu``, ``cuda:0``, ``mps`` (Apple), or the
strange ``meta`` device that stores shapes and no data. Two tensors can only
meet in an operation if they share a device. A model on the GPU and a batch
on the CPU is a RuntimeError, one you will meet approximately weekly.

The rules of the ferry:

* ``torch.device("cuda")`` is just a label; constructing it does not need a GPU.
* ``tensor.to(device)`` returns a NEW tensor (or the same one if already there).
  ``module.to(device)`` moves the module IN PLACE and returns it.
* Devices do not change dtype, and dtype has its own trap: numpy defaults to
  float64, so ``torch.from_numpy(np.random.rand(...))`` is a float64 tensor,
  and ``nn.Linear`` (float32) refuses it: "mat1 and mat2 must have the same
  dtype". Fix the dtype at the border, once.
* A module has no ``.device``. Ask its parameters.

The trials run on CPU. Where they need a "second device" they use ``meta``,
which exists on every build.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def pick_device(prefer: str | None = None) -> torch.device:
    """The best available device, or the one the caller insists on.

    If ``prefer`` is given (e.g. ``"cpu"``, ``"cuda:1"``), return ``torch.device(prefer)``
    without checking availability: an override is trusted, and the first
    tensor moved there will complain if you were wrong. Otherwise: ``cuda`` if
    ``torch.cuda.is_available()``, else ``mps`` if ``torch.backends.mps.is_available()``,
    else ``cpu``. Always return a ``torch.device``, never a string.
    """
    raise NotImplementedError("pick_device() is unwritten")


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor inside ``obj`` to ``device``, preserving the container structure.

    ``obj`` may be a tensor, or a list / tuple / dict nesting tensors at any
    depth. Lists stay lists, tuples stay tuples, dicts keep their keys. Anything
    that is not a tensor or one of those containers (strings, numbers, None,
    arbitrary objects) is returned as-is, the very same object.
    """
    raise NotImplementedError("to_device() is unwritten")


def ensure_float32(batch: Any) -> Any:
    """Cure the float64 that numpy smuggled in. Same recursion as ``to_device``.

    * a floating-point tensor of any width -> ``float32``
    * a numpy array -> a tensor (floating arrays become ``float32``; integer
      arrays keep their integer dtype)
    * integer / bool tensors are returned unchanged: labels must stay ``int64``
    * containers are rebuilt like in ``to_device``; other objects pass through
    """
    raise NotImplementedError("ensure_float32() is unwritten")


def model_device(model: nn.Module) -> torch.device:
    """The device a model lives on, read from its first parameter.

    A module with no parameters may still own buffers (``model.buffers()``);
    use those next. A module with neither has no device: raise ``ValueError``.
    """
    raise NotImplementedError("model_device() is unwritten")
