"""SECRET - THE TILED GAZE   (optional)

    Behind the Oracle's dais, a slit in the wall too narrow for the whole
    hall to fit through. A head passes it one tile of voices at a time, and
    somehow still weighs them all correctly.

Plain attention materialises the (Tq, Tk) score matrix. For T = 32k that is
a billion floats per head. Memory-efficient attention (FlashAttention's
forward pass) never builds it: it walks the keys in blocks and keeps, per
query, three running quantities:

    m    the largest score seen so far                    (B, Tq, 1)
    l    the sum of exp(score - m) seen so far            (B, Tq, 1)
    acc  the sum of exp(score - m) * v seen so far        (B, Tq, dv)

For each block of keys k_blk, v_blk (block_size keys):

    s      = q @ k_blk^T / sqrt(d)                        (B, Tq, block_size)
    m_new  = max(m, s.max over the block)
    alpha  = exp(m - m_new)          rescale everything accumulated under the OLD max
    p      = exp(s - m_new)                               (B, Tq, block_size)
    l      = alpha * l + p.sum(-1)
    acc    = alpha * acc + p @ v_blk
    m      = m_new

    out    = acc / l

Why it is exact: softmax is invariant to subtracting a constant from every
score, and exp(s - m_old) * exp(m_old - m_new) == exp(s - m_new), so every
earlier contribution is re-expressed under the new max with one multiply. The
largest intermediate is (B, Tq, block_size); the (Tq, Tk) matrix never exists.
Start m at -inf (so exp(m - m_new) is 0 for the first block) and l, acc at 0.

The trial checks the result against plain attention for several block sizes,
including ones that do not divide Tk (the last block is just shorter), and
then watches every tensor operation you perform: if any tensor with trailing
shape (Tq, Tk) or (Tk, Tq) appears, the slit was too narrow and you are
caught. Calling torch's own scaled_dot_product_attention is also caught.
"""

from __future__ import annotations

import torch


def blockwise_logsumexp(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """logsumexp over the last axis of ``x`` (..., N), computed block by block.

    The warm-up: the same (m, l) recurrence without the values. Walk
    ``x[..., start:start+block_size]``, keep running m and l with
    ``keepdim=True``, and return ``(m + log(l)).squeeze(-1)``, shape (...).
    Must equal ``torch.logsumexp(x, dim=-1)`` to 1e-5 for any block_size >= 1.
    """
    raise NotImplementedError("blockwise_logsumexp() is unwritten")


def online_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Attention without the (Tq, Tk) matrix. q: (B, Tq, d), k: (B, Tk, d), v: (B, Tk, dv).

    Returns out (B, Tq, dv) equal (to 1e-5) to plain, unmasked scaled
    dot-product attention. Iterate over key blocks of ``block_size`` (the last
    block may be shorter) with the recurrence in the module docstring. Scale
    by 1 / sqrt(d) where d = q.shape[-1]. Never create a tensor whose last two
    dims are (Tq, Tk).
    """
    raise NotImplementedError("online_softmax_attention() is unwritten")
