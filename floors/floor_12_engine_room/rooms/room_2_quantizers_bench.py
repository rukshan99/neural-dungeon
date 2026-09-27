"""ROOM 12.2 - THE QUANTIZER'S BENCH

    A workbench under a single lamp. On it: a crucible, a set of moulds with
    exactly 255 notches, and a ledger of weights that no longer fit in the
    vault. The quantizer melts each weight down and pours it into the nearest
    notch. Something is lost. The question is always how much, and where.

Symmetric absmax int8 quantization of a weight matrix ``w`` of shape (out, in):

    scale = max|w| / 127          one number per row (per-channel) or per matrix
    q     = round(w / scale)      an integer in [-127, 127], stored as int8
    w_hat = q * scale             what the model will actually multiply with

The largest-magnitude weight lands exactly on +-127, so nothing is clipped and
every weight is within ``scale / 2`` of its original. That bound is the whole
story of per-channel vs per-tensor: one row with a large outlier forces a large
scale on every row of a per-tensor scheme, and every other row loses precision
it did not have to lose. Per-channel gives each output row its own scale.

Storage drops 4x (1 byte instead of 4 per weight). In this room the matmul
still runs in float32 on the dequantized weight - that saves memory, not
compute. Real int8 kernels multiply the int8 codes and apply the scales to the
accumulated result; the algebra is the same because a per-row scale factors
out of ``x @ W.T``.

One layer is left alone: ``lm_head``. Its weight IS ``wte.weight`` (tied). The
room asks you to understand why that matters before you quantize the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

QMAX = 127  # symmetric int8 uses [-127, 127]; -128 is left unused so the grid is symmetric


def quantize_int8(w: torch.Tensor, per_channel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax quantization of a 2-D weight (out, in).

    Returns ``(q, scale)``: q is int8 of shape (out, in); scale is float32 of shape
    (out, 1) when ``per_channel`` (one scale per output row, kept as a column so it
    broadcasts against w) or (1, 1) when per-tensor. scale = absmax / 127, clamped
    to a tiny positive minimum so an all-zero row does not divide by zero.
    Round to nearest, clamp to [-127, 127], cast to ``torch.int8``. Detach w first.
    """
    raise NotImplementedError("quantize_int8() is unwritten")


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 codes back to float32: ``q.float() * scale`` (scale broadcasts)."""
    raise NotImplementedError("dequantize() is unwritten")


def quantization_error(w: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    """``{"max_abs_error": max |w - w_hat|, "relative_error": ||w - w_hat||_F / ||w||_F}`` as floats."""
    raise NotImplementedError("quantization_error() is unwritten")


class QuantizedLinear(nn.Module):
    """A drop-in replacement for an ``nn.Linear`` that stores its weight as int8 codes.

    Store ``weight_q`` (int8, (out, in)) and ``scale`` (float32) as buffers via
    ``self.register_buffer``; keep the bias as float32 (a Parameter or buffer; None if
    the linear had none). Expose ``in_features`` and ``out_features``.

    ``forward(x)`` dequantizes on the fly: ``F.linear(x, dequantize(weight_q, scale), bias)``.
    (Yes, this does the matmul in float32. Say so in a comment; the trial is about
    storage and error, and the real int8 kernel is a topic for the cheat sheet.)
    """

    def __init__(self, linear: nn.Linear, per_channel: bool = True):
        raise NotImplementedError("QuantizedLinear.__init__() is unwritten")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QuantizedLinear.forward() is unwritten")


def quantize_model_linears(model, per_channel: bool = True):
    """Deep-copy ``model`` and replace every ``nn.Linear`` EXCEPT ``lm_head`` with a QuantizedLinear.

    Returns ``(quantized_copy, n_replaced)``. The original model must be untouched.
    Walk ``named_modules()``, collect the targets first, then ``setattr`` on each
    parent (``model.get_submodule(parent_name)``): never mutate a module tree while
    iterating over it. ``copy.deepcopy`` preserves the wte/lm_head weight tie; your
    copy must too.

    Why skip lm_head: its weight is the very same tensor as ``wte.weight``.
    Quantizing it would quantize the embedding lookup as well (or silently break the
    tie), and the head's error lands straight on the logits.
    """
    raise NotImplementedError("quantize_model_linears() is unwritten")


# ---------------------------------------------------------------------------
# THE PROPHECY OF THE BENCH
# Replace every None with your prediction BEFORE running the trial. The trial
# measures each one on the Chronicler and on a small matrix with one loud row.
# ---------------------------------------------------------------------------
QUANT_PROPHECY: dict[str, str | int | None] = {
    # How many int8 weights fit in the bytes of ONE fp32 weight? (an int)
    "int8_weights_per_fp32_weight": None,
    # A (8, 16) matrix where one row is 10x louder than the rest. Which scheme has the
    # larger max abs error on the QUIET rows? "per-tensor" | "per-channel" | "same"
    "larger_error_on_the_quiet_rows": None,
    # Per-channel: the largest-magnitude entry of every row maps to which |code|? (an int)
    "code_of_each_rows_absmax": None,
    # Quantizing lm_head would also quantize which tensor? (the attribute name on the model)
    "tensor_tied_to_lm_head": None,
    # Dequantize-on-the-fly (this room's QuantizedLinear) saves: "memory" | "compute" | "both"
    "dequant_on_the_fly_saves": None,
}
