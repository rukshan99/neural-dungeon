"""SECRET - THE CUSTOM WHISPER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
"""

from __future__ import annotations

import torch


class SoftThreshold(torch.autograd.Function):
    """sign(x) * max(|x| - lam, 0), differentiated by hand."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        mask = x.abs() > lam  # where the output is not pinned at zero
        ctx.save_for_backward(mask)
        return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (mask,) = ctx.saved_tensors
        grad_x = grad_output * mask.to(grad_output.dtype)  # slope 1 outside the band, 0 inside
        return grad_x, None  # lam is a float: no gradient for it


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Always go through .apply(); calling forward() directly bypasses autograd."""
    return SoftThreshold.apply(x, lam)


def soft_threshold_composed(x: torch.Tensor, lam: float) -> torch.Tensor:
    """The witness: ordinary ops that autograd differentiates on its own."""
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)
