"""ROOM 4.6 - THE HALF-LIGHT

    Past the ferry the passage narrows and the torches hang twice as far
    apart. Half the light, half the oil. You can still read the runes cut
    into the walls; you can no longer read the small print beneath them.
    Somewhere ahead, a number that used to be 70000 has quietly become
    infinity, and a gradient that used to be 1e-8 has quietly become nothing.

A float32 spends 32 bits on every number: 1 sign, 8 exponent, 23 mantissa.
Mixed precision keeps the weights in float32 and does the expensive
arithmetic (matmuls, convolutions) in a 16-bit format: half the memory
traffic, and on tensor-core hardware several times the throughput. There are
two 16-bit formats, and they spend their bits differently:

              sign  exponent  mantissa       max        eps          tiny
    float32    1       8         23       3.40e+38   1.19e-07     1.18e-38
    float16    1       5         10       65504      9.77e-04     6.10e-05
    bfloat16   1       8          7       3.39e+38   7.81e-03     1.18e-38

``eps`` is the gap between 1 and the next number up (2 ** -mantissa_bits);
``tiny`` is the smallest normal number (2 ** (2 - 2 ** (exponent_bits - 1))).
Below ``tiny`` come the subnormals, and below those, zero.

float16 keeps precision (11 significant bits, about three decimal digits) and
throws away RANGE: it overflows above 65504 and flushes to zero below about
6e-8. bfloat16 keeps float32's RANGE (the same 8 exponent bits) and throws
away precision: 8 significant bits, so ``1 + 0.001 == 1``. Every rule of
mixed precision follows from that one trade.

Three tools, in order of how often you will reach for them:

* ``torch.autocast(device_type=..., dtype=...)``: a context manager that runs
  each op in the dtype an allow-list prescribes. Matmul-family ops in the
  low-precision dtype; the loss functions in float32; everything else in
  whatever dtype its inputs already have. Parameters are not touched: they
  stay float32, and autocast casts copies on the way into each op.
* ``model.to(torch.bfloat16)`` for INFERENCE: the weights themselves become
  16-bit and the model takes half the memory. Not for training: a weight
  update of 1e-4 on a weight near 1 vanishes in 8 significant bits, which is
  why training keeps a float32 "master" copy of every weight.
* Dynamic loss scaling, for float16 TRAINING: multiply the loss by a large S
  so that small gradients survive the trip through float16 (the chain rule is
  linear, so every gradient is multiplied by S too), divide the gradients by
  S before the step, and when a gradient comes back inf or nan, skip the step
  and halve S. ``torch.amp.GradScaler`` does this on CUDA. Here you write the
  CPU version by hand, so that you know exactly what it does.

Then the Half-Light Prophecy: eight snippets, eight predictions. Commit
before you run them; the arithmetic is small enough to do in your head with
the table above.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def dtype_report(dtype: torch.dtype) -> dict[str, int | float]:
    """Describe a floating-point format. Return a dict with exactly these keys:

        bits            total width (32 or 16)
        exponent_bits   8 for float32 and bfloat16, 5 for float16
        mantissa_bits   23 for float32, 10 for float16, 7 for bfloat16
        max             largest finite value
        eps             the gap between 1.0 and the next representable number
        tiny            the smallest positive NORMAL number

    ``torch.finfo(dtype)`` knows ``bits``, ``max``, ``eps`` and ``tiny``; use it
    rather than typing the numbers. The exponent and mantissa widths it does
    not know: keep a small table. Return Python ``int``s and ``float``s. For any
    dtype outside the table (integers, bools) raise ``ValueError``.

    The trial checks the arithmetic that ties the columns together:
    ``bits == 1 + exponent_bits + mantissa_bits``, ``eps == 2 ** -mantissa_bits``
    and ``tiny == 2 ** (2 - 2 ** (exponent_bits - 1))``.
    """
    raise NotImplementedError("dtype_report() is unwritten")


def overflow_demo(x_fp16: torch.Tensor) -> torch.Tensor:
    """The sum of squares of a float16 vector, computed NAIVELY in float16.

    ``(x * x).sum()`` with nothing cast: a 0-d ``float16`` tensor, inf and all.
    You are being asked to write this one badly on purpose, so that the next
    function has something to fix. If the input has 1000 entries of 10.0, every
    square is a harmless 100 and the sum, 100000, is past 65504 and comes back
    ``inf``. If one entry is 300, its square alone is past 65504. Do not upcast.
    """
    raise NotImplementedError("overflow_demo() is unwritten")


def safe_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """The same quantity, accumulated in float32. Return a 0-d ``float32`` tensor.

    Cast ``x`` to float32 BEFORE squaring. ``(x * x).sum(dtype=torch.float32)``
    is not enough: it accumulates in float32 but still squares in float16, and
    ``300.0 ** 2`` has already overflowed by the time the sum begins. The rule
    generalises: reductions, norms and variances belong in float32, which is
    exactly what autocast's float32 allow-list encodes.
    """
    raise NotImplementedError("safe_sum_of_squares() is unwritten")


def forward_autocast_bf16(model: nn.Module, x: torch.Tensor):
    """Run ``model(x)`` under ``torch.autocast(device_type="cpu", dtype=torch.bfloat16)``.

    Return whatever the model returns (a tensor, or a tuple if the model makes
    one). Do not cast the model or the input yourself: autocast decides per op.
    Inside the block, ``nn.Linear`` and matmuls produce bfloat16; the loss
    functions (``F.cross_entropy``, ``F.mse_loss``, ...) produce float32; the
    parameters stay float32 throughout. Use the context manager, so that
    autocast is switched off again when the call returns.
    """
    raise NotImplementedError("forward_autocast_bf16() is unwritten")


def cast_for_inference(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A deep copy of ``model`` with every floating-point parameter and buffer in ``dtype``, in eval mode.

    ``copy.deepcopy`` first, so the caller's float32 model (and its training
    flag) is left exactly as it was: that copy is the master. Then ``.to(dtype)``
    (a module's ``.to`` accepts a dtype as well as a device; integer buffers are
    left alone) and ``.eval()``. Inputs must be cast to the same dtype by whoever
    calls the copy; a bfloat16 Linear refuses a float32 batch.
    """
    raise NotImplementedError("cast_for_inference() is unwritten")


def parameter_bytes(model: nn.Module) -> int:
    """How many bytes the parameters of ``model`` occupy. Return a Python ``int``.

    ``p.numel() * p.element_size()`` summed over ``model.parameters()``: a
    float32 element is 4 bytes, float16 and bfloat16 are 2. Do not assume 4.
    """
    raise NotImplementedError("parameter_bytes() is unwritten")


class LossScaler:
    """Dynamic loss scaling, by hand: what ``torch.amp.GradScaler`` does on CUDA.

    State: the current scale ``S`` (start at ``init_scale``) and a count of
    consecutive successful steps. Three methods:

    ``scale(loss)``
        ``loss * S``. The result stays in the graph, so ``scaled.backward()``
        leaves ``S * grad`` in every ``.grad``.
    ``unscale_(grads)``
        Divide each gradient tensor by ``S`` IN PLACE (``g.div_(S)``) and return
        ``True`` if any of them holds an ``inf`` or a ``nan``, else ``False``.
        (``torch.isfinite`` is your friend.) Skip ``None`` entries.
    ``step(optimizer, params)``
        ``unscale_`` the ``.grad`` of every parameter in ``params``. If anything
        was not finite: do NOT call ``optimizer.step()``, multiply ``S`` by
        ``backoff_factor``, reset the success count, return ``False``. Otherwise
        call ``optimizer.step()``, count one success, and once ``growth_interval``
        consecutive successes are reached multiply ``S`` by ``growth_factor``
        and reset the count. Return ``True``.

    ``get_scale()`` returns the current ``S`` as a Python float. With the
    defaults (``2 ** 16``, doubling every 2000 clean steps, halving on every
    overflow) the scale settles just below the largest value the network's
    gradients can bear, which is the whole point: as large as possible so
    nothing underflows, no larger so nothing overflows.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        raise NotImplementedError("LossScaler.__init__() is unwritten")

    def get_scale(self) -> float:
        raise NotImplementedError("LossScaler.get_scale() is unwritten")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LossScaler.scale() is unwritten")

    def unscale_(self, grads: Iterable[torch.Tensor | None]) -> bool:
        raise NotImplementedError("LossScaler.unscale_() is unwritten")

    def step(self, optimizer: torch.optim.Optimizer, params: Iterable[torch.Tensor]) -> bool:
        raise NotImplementedError("LossScaler.step() is unwritten")


# ---------------------------------------------------------------------------
# THE HALF-LIGHT PROPHECY
# Eight snippets. Each ends by assigning ``result``: the string "inf" or
# "finite", the string "zero" or "nonzero", or a bool. Predict each one from
# the table in the docstring: 65504, eps of 2**-10 and 2**-7, and where the
# subnormals of float16 give out (about 6e-8). Do not run them first.
# ---------------------------------------------------------------------------
HALF_LIGHT_SNIPPETS: dict[str, str] = {
    "70000_in_float16": """
x = torch.tensor(70000.0).to(torch.float16)
result = "inf" if torch.isinf(x) else "finite"
""",
    "70000_in_bfloat16": """
x = torch.tensor(70000.0).to(torch.bfloat16)
result = "inf" if torch.isinf(x) else "finite"
""",
    "1_plus_1e-3_equals_1_in_float16": """
one = torch.tensor(1.0, dtype=torch.float16)
small = torch.tensor(1e-3, dtype=torch.float16)
result = bool((one + small) == one)
""",
    "1_plus_1e-3_equals_1_in_bfloat16": """
one = torch.tensor(1.0, dtype=torch.bfloat16)
small = torch.tensor(1e-3, dtype=torch.bfloat16)
result = bool((one + small) == one)
""",
    "1e-4_squared_in_float16": """
x = torch.tensor(1e-4, dtype=torch.float16)
result = "zero" if (x * x) == 0 else "nonzero"
""",
    "1e-4_squared_in_bfloat16": """
x = torch.tensor(1e-4, dtype=torch.bfloat16)
result = "zero" if (x * x) == 0 else "nonzero"
""",
    "a_gradient_of_1e-8_in_float16": """
g = torch.tensor(1e-8).to(torch.float16)
result = "zero" if g == 0 else "nonzero"
""",
    "the_same_gradient_scaled_by_2_to_16": """
g = (torch.tensor(1e-8) * 2.0 ** 16).to(torch.float16)
result = "zero" if g == 0 else "nonzero"
""",
}

HALF_LIGHT_PROPHECY: dict[str, str | bool | None] = {
    "70000_in_float16": None,
    "70000_in_bfloat16": None,
    "1_plus_1e-3_equals_1_in_float16": None,
    "1_plus_1e-3_equals_1_in_bfloat16": None,
    "1e-4_squared_in_float16": None,
    "1e-4_squared_in_bfloat16": None,
    "a_gradient_of_1e-8_in_float16": None,
    "the_same_gradient_scaled_by_2_to_16": None,
}
