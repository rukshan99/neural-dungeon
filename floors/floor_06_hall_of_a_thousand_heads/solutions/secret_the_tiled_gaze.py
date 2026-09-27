"""SECRET - THE TILED GAZE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The online-softmax recurrence (the FlashAttention forward pass, minus the
hardware). Per query we keep the running max m, the running denominator l
and the running numerator acc. Each new block of keys may raise the max, in
which case everything accumulated so far is rescaled by exp(m_old - m_new).
"""

from __future__ import annotations

import math

import torch


def blockwise_logsumexp(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """logsumexp over the last axis, one block at a time. Equals torch.logsumexp(x, -1)."""
    N = x.shape[-1]
    m = torch.full(x.shape[:-1] + (1,), float("-inf"), dtype=x.dtype)
    l_sum = torch.zeros_like(m)  # "l" in the papers: the running sum of exp(x - m)
    for start in range(0, N, block_size):
        blk = x[..., start : start + block_size]
        m_new = torch.maximum(m, blk.amax(dim=-1, keepdim=True))
        l_sum = l_sum * torch.exp(m - m_new) + torch.exp(blk - m_new).sum(dim=-1, keepdim=True)
        m = m_new
    return (m + torch.log(l_sum)).squeeze(-1)


def online_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Exact attention that never materialises the (Tq, Tk) score matrix."""
    B, Tq, d = q.shape
    Tk = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    m = torch.full((B, Tq, 1), float("-inf"), dtype=q.dtype)  # running max per query
    l_sum = torch.zeros((B, Tq, 1), dtype=q.dtype)  # running sum of exp(s - m) ("l" in the papers)
    acc = torch.zeros((B, Tq, v.shape[-1]), dtype=q.dtype)  # running sum of exp(s - m) * v
    for start in range(0, Tk, block_size):
        k_blk = k[:, start : start + block_size]  # (B, blk, d); the last block may be short
        v_blk = v[:, start : start + block_size]  # (B, blk, dv)
        s = (q @ k_blk.transpose(-2, -1)) * scale  # (B, Tq, blk): the only score tile alive
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        alpha = torch.exp(m - m_new)  # 0 on the first block (m == -inf), <= 1 afterwards
        p = torch.exp(s - m_new)
        l_sum = alpha * l_sum + p.sum(dim=-1, keepdim=True)
        acc = alpha * acc + p @ v_blk
        m = m_new
    return acc / l_sum
