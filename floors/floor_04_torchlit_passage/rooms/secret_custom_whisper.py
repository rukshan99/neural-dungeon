"""SECRET - THE CUSTOM WHISPER   (optional)

    Behind the crypt wall a voice repeats two words in turn: "forward",
    "backward". All floor long autograd has whispered the second word for you.
    Down here you whisper it yourself.

``torch.autograd.Function`` is how you add an operation autograd does not know
how to differentiate (or knows, but too slowly). You write two static methods:

    forward(ctx, *inputs)      compute the output; stash what backward needs
                               with ctx.save_for_backward(...)
    backward(ctx, grad_output) return d(loss)/d(input) for EACH input, in
                               order; None for inputs that need no gradient

and you *call* it with ``MyFunction.apply(...)``, never directly.

The op here is the soft threshold (shrinkage), the heart of L1-regularised
solvers and of sparse coding:

    soft_threshold(x, lam) = sign(x) * max(|x| - lam, 0)

    d/dx = 1 where |x| > lam, else 0   (the kink at |x| == lam has no derivative;
                                        either 0 or 1 is a valid subgradient)

The trial compares your Function with the same op composed from ordinary
tensor ops (which autograd differentiates on its own), then runs
``torch.autograd.gradcheck`` in float64: finite differences against your
hand-written backward, at points safely away from the kinks.
"""

from __future__ import annotations

import torch


class SoftThreshold(torch.autograd.Function):
    """sign(x) * max(|x| - lam, 0) with a hand-written backward. ``lam`` is a plain float."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute the output; save the mask ``|x| > lam`` for backward.

        ``ctx.save_for_backward`` takes tensors only; read them back in backward
        as ``ctx.saved_tensors``. Return a tensor of the same shape and dtype as ``x``.
        """
        raise NotImplementedError("SoftThreshold.forward() is unwritten")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return ``(grad_x, None)``: one entry per forward input, ``None`` for ``lam``.

        grad_x = grad_output where the mask is True, else 0. Match grad_output's dtype.
        """
        raise NotImplementedError("SoftThreshold.backward() is unwritten")


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """The user-facing op: ``SoftThreshold.apply(x, lam)``."""
    raise NotImplementedError("soft_threshold() is unwritten")


def soft_threshold_composed(x: torch.Tensor, lam: float) -> torch.Tensor:
    """The same function from ordinary ops: ``torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)``.

    Autograd differentiates this one by itself. It is the witness your custom
    backward is checked against.
    """
    raise NotImplementedError("soft_threshold_composed() is unwritten")
