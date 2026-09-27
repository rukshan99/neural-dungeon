"""ROOM 4.6 - THE HALF-LIGHT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The prophecy, worked out from the table rather than by running it:

    70000 in float16          "inf"      the largest float16 is 65504
    70000 in bfloat16         "finite"   bfloat16 has float32's exponent; it lands on 70144
    1 + 1e-3 == 1, float16    False      eps is 2**-10 = 0.00098 < 0.001, so 1.001 rounds to 1.000977
    1 + 1e-3 == 1, bfloat16   True       eps is 2**-7 = 0.0078 > 0.001, so 1.001 rounds to 1.0
    (1e-4)**2 in float16      "zero"     1e-8 is below the last float16 subnormal, 2**-24 = 6e-8
    (1e-4)**2 in bfloat16     "nonzero"  1e-8 is comfortably inside float32's range
    1e-8 in float16           "zero"     the same underflow, wearing a gradient's clothes
    1e-8 * 2**16 in float16   "nonzero"  6.6e-4 is a normal float16 number: that is loss scaling
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
import torch.nn as nn

# exponent and mantissa widths; torch.finfo knows everything else
_FORMAT_BITS: dict[torch.dtype, tuple[int, int]] = {
    torch.float32: (8, 23),
    torch.float16: (5, 10),
    torch.bfloat16: (8, 7),
    torch.float64: (11, 52),
}


def dtype_report(dtype: torch.dtype) -> dict[str, int | float]:
    """finfo for the numbers, a table for the two widths finfo does not know."""
    if dtype not in _FORMAT_BITS:
        raise ValueError(f"{dtype} is not a floating-point format this room knows about.")
    exponent_bits, mantissa_bits = _FORMAT_BITS[dtype]
    info = torch.finfo(dtype)
    return {
        "bits": int(info.bits),
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits,
        "max": float(info.max),
        "eps": float(info.eps),
        "tiny": float(info.tiny),
    }


def overflow_demo(x_fp16: torch.Tensor) -> torch.Tensor:
    """Deliberately naive: squares and sums in whatever dtype the input has."""
    return (x_fp16 * x_fp16).sum()


def safe_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """Cast first, then square, then sum: every intermediate has float32's range."""
    x32 = x.to(torch.float32)
    return (x32 * x32).sum()


def forward_autocast_bf16(model: nn.Module, x: torch.Tensor):
    """The context manager picks the dtype per op; nothing else changes."""
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        return model(x)


def cast_for_inference(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Copy, cast, eval. The original stays float32 and untouched."""
    return copy.deepcopy(model).to(dtype).eval()


def parameter_bytes(model: nn.Module) -> int:
    """numel * element_size, so a half-precision copy reports half the bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


class LossScaler:
    """Dynamic loss scaling: grow while the gradients are finite, back off the moment they are not."""

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self._scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._good_steps = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        # A plain multiplication, so autograd multiplies every gradient by the same S.
        return loss * self._scale

    def unscale_(self, grads: Iterable[torch.Tensor | None]) -> bool:
        found_inf = False
        for g in grads:
            if g is None:
                continue
            g.div_(self._scale)
            if not bool(torch.isfinite(g).all()):
                found_inf = True
        return found_inf

    def step(self, optimizer: torch.optim.Optimizer, params: Iterable[torch.Tensor]) -> bool:
        found_inf = self.unscale_(p.grad for p in params)
        if found_inf:
            # The gradients are garbage: leave the weights alone and try a smaller S next time.
            self._scale *= self.backoff_factor
            self._good_steps = 0
            return False
        optimizer.step()
        self._good_steps += 1
        if self._good_steps >= self.growth_interval:
            # Nothing has overflowed for a while; probe a larger S so that fewer gradients underflow.
            self._scale *= self.growth_factor
            self._good_steps = 0
        return True


# ---------------------------------------------------------------------------
# THE HALF-LIGHT PROPHECY
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
    "70000_in_float16": "inf",
    "70000_in_bfloat16": "finite",
    "1_plus_1e-3_equals_1_in_float16": False,
    "1_plus_1e-3_equals_1_in_bfloat16": True,
    "1e-4_squared_in_float16": "zero",
    "1e-4_squared_in_bfloat16": "nonzero",
    "a_gradient_of_1e-8_in_float16": "zero",
    "the_same_gradient_scaled_by_2_to_16": "nonzero",
}
