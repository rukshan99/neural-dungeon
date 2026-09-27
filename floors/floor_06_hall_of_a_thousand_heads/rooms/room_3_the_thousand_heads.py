"""ROOM 6.3 - THE THOUSAND HEADS

    Look up. The hall is not one head but a thousand, carved in rows along
    the vault. Each listens to the same voices and each hears something
    different: one tracks who is speaking, one tracks what was said two
    voices ago, one only cares about the punctuation.

One head with d_model-dimensional queries computes one weighting of the keys
per query. Multi-head attention runs H heads *in parallel*, each in its own
head_dim = d_model / H dimensional slice, so each query gets H different
weightings. Their outputs are concatenated and mixed by one final linear map.

    x                     (B, T, d_model)
    qkv = Linear(x)       (B, T, 3 * d_model)     one fused matmul for q, k, v
    q, k, v = split       (B, T, d_model) each
    split_heads           (B, H, T, head_dim) each     H is a batch axis now
    scores = q @ k^T / sqrt(head_dim)         (B, H, T, T)
    weights = softmax(scores, -1)             (B, H, T, T)   rows sum to 1
    heads = weights @ v                       (B, H, T, head_dim)
    merge_heads                               (B, T, d_model)
    out = proj(...)                           (B, T, d_model)

Note the scaling: sqrt(head_dim), because each head's dot product runs over
head_dim entries. Note also that the attention itself is Room 6.1's function
with two batch axes instead of one, which is why you wrote it to touch only
the last two dims.

Parameter count: qkv has 3*d_model*d_model weights + 3*d_model biases, proj
has d_model*d_model + d_model. Total 4*d_model^2 + 4*d_model, and that is
exactly what torch.nn.MultiheadAttention has too. The trial copies your
weights into torch's module and demands the same output, so the layer names
and the head layout below are a contract, not a suggestion. The same layout
(``c_attn`` / ``c_proj`` in the Chronicler) carries you through Floor 7.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .room_1_the_single_gaze import scaled_dot_product_attention  # noqa: F401 - use it in forward


class MultiHeadAttention(nn.Module):
    """H heads of scaled dot-product self-attention with fused qkv and an output projection."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build the layers. Contract (the trial reads these attributes by name):

            self.d_model  = d_model
            self.n_heads  = n_heads
            self.head_dim = d_model // n_heads
            self.qkv      = nn.Linear(d_model, 3 * d_model)    fused q, k, v projection
            self.proj     = nn.Linear(d_model, d_model)        output projection

        Raise ``ValueError`` if d_model is not divisible by n_heads. Remember
        ``super().__init__()`` first; nn.Module insists.
        """
        raise NotImplementedError("MultiHeadAttention.__init__() is unwritten")

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, head_dim).

        Head h owns columns ``h*head_dim : (h+1)*head_dim`` of the model
        dimension: ``x.view(B, T, H, head_dim).transpose(1, 2)``. This
        contiguous-chunk layout is what torch.nn.MultiheadAttention uses, so
        the weights will be interchangeable.
        """
        raise NotImplementedError("split_heads() is unwritten")

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, d_model). The exact inverse of split_heads.

        ``transpose(1, 2)`` puts T back before H, then ``.contiguous().view(B, T, -1)``
        (or ``reshape``) glues the heads side by side.
        """
        raise NotImplementedError("merge_heads() is unwritten")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d_model) -> out (B, T, d_model), or (out, weights) if return_weights.

        mask: optional bool tensor broadcastable to (B, H, T, T), True = may attend.
        A (T, T) causal mask and a (B, 1, T, T) combined mask (Room 6.4) both
        broadcast; apply it before the softmax.
        weights: (B, H, T, T), one row per (batch, head, query), rows sum to 1.

        1. ``self.qkv(x).split(self.d_model, dim=-1)`` gives q, k, v in that order.
        2. split_heads each.
        3. Room 6.1's scaled_dot_product_attention on the 4-D tensors (its
           sqrt(d) is now sqrt(head_dim), because d is the last axis).
        4. merge_heads, then self.proj.
        """
        raise NotImplementedError("forward() is unwritten")
