"""ROOM 6.2 - THE VEIL OF CAUSALITY

    A gauze veil hangs between each head and the voices further down the
    hall. A head may hear itself and everything said before it. Nothing
    after. The veil is what makes the hall an oracle rather than an echo.

A language model is trained to predict token t+1 from tokens 0..t. If the
head at position t can hear position t+1, the answer is sitting right there:
the loss drops to nearly zero, the model learns to copy, and at generation
time, where there is no future to copy from, it produces nonsense. The causal
mask is the single most consequential correctness property of the model, and
it fails silently. You cannot see a leak in the loss curve. You can only test
for it.

The veil is a lower-triangular boolean matrix:

    mask[i, j] = (j <= i)          query i may attend key j iff j is not after i

    T = 4:   [[1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 1]]        torch.tril(torch.ones(T, T, dtype=torch.bool))

Applied before the softmax as -inf (Room 6.1), every weight strictly above
the diagonal is exactly zero and every row still sums to 1.

The third function is the important one. ``leaks_future`` does not read
anybody's code. It perturbs the future and checks whether the past noticed:
for each t, overwrite positions t+1..T-1 of q, k and v with fresh random
values and demand that out[:, :t+1] is unchanged. A correct causal attention
passes bitwise (the masked weights are exactly 0.0, and 0.0 * anything is
0.0). A mask applied after the softmax fails at once: the denominator of the
softmax still contains the future. The boss on this floor is a bigger
version of this function.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .room_1_the_single_gaze import scaled_dot_product_attention  # noqa: F401 - use it below


def causal_mask(T: int) -> torch.Tensor:
    """The veil: a (T, T) bool tensor with ``mask[i, j] == (j <= i)``.

    Lower triangle and diagonal True, strictly upper triangle False.
    dtype must be ``torch.bool``.
    """
    raise NotImplementedError("causal_mask() is unwritten")


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-attention behind the veil. q, k, v: (B, T, d). Returns ``(out, weights)``.

    out: (B, T, d). weights: (B, T, T) with weights[b, i, j] == 0 exactly for j > i.
    Build the mask from T and hand everything to Room 6.1's attention.
    """
    raise NotImplementedError("causal_attention() is unwritten")


def leaks_future(
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    B: int,
    T: int,
    d: int,
    seed: int,
) -> bool:
    """The leakage test. True if ``attn_fn`` lets any position hear its future.

    ``attn_fn(q, k, v) -> out`` takes three (B, T, d) tensors and returns an
    (B, T, d) tensor (a tensor, not a tuple).

    Recipe:
      1. ``g = torch.Generator().manual_seed(seed)``; draw q, k, v with
         ``torch.randn(B, T, d, generator=g)``; ``base = attn_fn(q, k, v)``.
      2. For each t in 0..T-2: clone q, k, v; overwrite positions t+1..T-1 of
         all three clones with fresh ``randn`` from the same generator; run
         attn_fn again.
      3. If ``out[:, :t+1]`` differs from ``base[:, :t+1]`` anywhere by more
         than 1e-6 (``torch.allclose(..., atol=1e-6, rtol=0)``), the veil is
         torn: return True.
      4. If every t survives, return False.

    Return a Python ``bool``. Never look at attn_fn's source: the boss's
    oracles all *claim* to be causal.
    """
    raise NotImplementedError("leaks_future() is unwritten")
