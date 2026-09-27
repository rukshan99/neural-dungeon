"""ROOM 12.2 - THE QUANTIZER'S BENCH  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Symmetric absmax int8: scale = max|w| / 127, q = round(w / scale), w ~ q * scale.
Per-channel keeps one scale per output row so a single large row cannot ruin
the resolution of every other row.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 127  # int8 symmetric range is [-127, 127]; -128 is left unused so the grid is symmetric


def quantize_int8(w: torch.Tensor, per_channel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax quantization of a 2-D weight (out, in) to int8.

    Returns (q int8 (out, in), scale float32). scale is (out, 1) per-channel or
    (1, 1) per-tensor, so ``q.float() * scale`` broadcasts back to (out, in).
    """
    w = w.detach().to(torch.float32)
    if per_channel:
        absmax = w.abs().amax(dim=1, keepdim=True)  # (out, 1): one scale per output row
    else:
        absmax = w.abs().amax().reshape(1, 1)  # (1, 1): one scale for the whole matrix
    scale = (absmax / QMAX).clamp(min=1e-8)  # an all-zero row would otherwise divide by zero
    q = torch.round(w / scale).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 codes back to float32: q * scale (broadcast)."""
    return q.to(torch.float32) * scale


def quantization_error(w: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    """max |w - deq| and the relative Frobenius error ||w - deq|| / ||w||."""
    diff = w.detach().float() - dequantize(q, scale)
    return {
        "max_abs_error": float(diff.abs().max()),
        "relative_error": float(diff.norm() / w.detach().float().norm().clamp(min=1e-12)),
    }


class QuantizedLinear(nn.Module):
    """An nn.Linear whose weight is stored as int8 codes plus float32 scales.

    forward dequantizes on the fly and runs the matmul in float32. That saves
    memory (4x on the weight) but not compute: real int8 kernels multiply the
    int8 codes directly and apply the scale to the accumulated result.
    """

    def __init__(self, linear: nn.Linear, per_channel: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.per_channel = per_channel
        q, scale = quantize_int8(linear.weight, per_channel)
        self.register_buffer("weight_q", q)  # int8 (out, in)
        self.register_buffer("scale", scale)  # float32 (out, 1) or (1, 1)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dequantize(self.weight_q, self.scale), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8, per_channel={self.per_channel}"


def quantize_model_linears(model, per_channel: bool = True):
    """Deep-copy ``model`` and replace every nn.Linear except ``lm_head`` with a QuantizedLinear.

    lm_head is skipped because its weight IS wte.weight (tied). Quantizing it
    would either quantize the embedding lookup too or silently break the tie,
    and the output projection is the layer whose error lands straight on the
    logits. Returns (quantized_copy, number_of_layers_replaced).
    """
    qmodel = copy.deepcopy(model)  # deepcopy keeps the wte/lm_head tie intact
    targets = [
        (name, module)
        for name, module in qmodel.named_modules()
        if isinstance(module, nn.Linear) and name != "lm_head"
    ]
    for name, module in targets:  # collect first, then replace: never mutate while iterating
        parent_name, _, attr = name.rpartition(".")
        parent = qmodel.get_submodule(parent_name) if parent_name else qmodel
        setattr(parent, attr, QuantizedLinear(module, per_channel))
    return qmodel, len(targets)


# The prophecy of the bench, answered. Each line is measured by the trial.
QUANT_PROPHECY: dict[str, str | int | None] = {
    "int8_weights_per_fp32_weight": 4,  # 1 byte vs 4 bytes
    "larger_error_on_the_quiet_rows": "per-tensor",  # the loud row sets everyone's scale
    "code_of_each_rows_absmax": 127,  # absmax / scale == 127 exactly, by construction
    "tensor_tied_to_lm_head": "wte",  # lm_head.weight is wte.weight
    "dequant_on_the_fly_saves": "memory",  # the matmul is still float32 of the same size
}
