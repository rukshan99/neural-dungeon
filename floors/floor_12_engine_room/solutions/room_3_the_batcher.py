"""ROOM 12.3 - THE BATCHER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Left padding lines every prompt up at the RIGHT edge, so the last column of the
batch is "the newest token" for every row and decoding appends a column. The
padding is made harmless by (1) position ids that start at 0 at the first real
token and (2) an attention mask that hides padded keys from every query.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def left_pad(sequences: list[list[int]], pad_id: int):
    """Pad on the LEFT to a rectangle.

    Returns (idx (B, T) long, attention_mask (B, T) bool: True on real tokens,
    position_ids (B, T) long: 0, 1, 2, ... from each row's first real token; 0 on pads).
    """
    if not sequences or any(len(s) == 0 for s in sequences):
        raise ValueError("every sequence must have at least one token")
    B = len(sequences)
    T = max(len(s) for s in sequences)
    idx = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.bool)
    for b, seq in enumerate(sequences):
        idx[b, T - len(seq):] = torch.tensor(seq, dtype=torch.long)
        attention_mask[b, T - len(seq):] = True
    # Count real tokens seen so far, minus one; pads (before the first real token) clamp to 0.
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return idx, attention_mask, position_ids


def masked_attention(block, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """One block's self-attention with an explicit boolean mask.

    x: (B, T, C) after ln_1;  allowed: bool broadcastable to (B, H, T, T), True where
    query i may look at key j. Returns (B, T, C) after c_proj.
    """
    attn = block.attn
    B, T, C = x.shape
    H = attn.n_head
    hd = C // H
    q, k, v = attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T, H, hd).transpose(1, 2)
    k = k.view(B, T, H, hd).transpose(1, 2)
    v = v.view(B, T, H, hd).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    att = att.masked_fill(~allowed, float("-inf"))
    att = F.softmax(att, dim=-1)
    # A padded query whose every key is padded sees only -inf: softmax gives NaN.
    # Those rows are never read, but NaN spreads through matmuls, so zero them.
    att = torch.nan_to_num(att, nan=0.0)
    y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
    return attn.c_proj(y)


def forward_masked(model, idx: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """The GPT forward with explicit positions and a causal + key-padding mask. Returns logits (B, T, V)."""
    B, T = idx.shape
    if T > model.cfg.block_size:
        raise ValueError(f"sequence length {T} exceeds block_size {model.cfg.block_size}")
    x = model.wte(idx) + model.wpe(position_ids)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))  # (T, T)
    allowed = causal[None, None, :, :] & attention_mask[:, None, None, :]  # (B, 1, T, T)
    for block in model.blocks:
        x = x + masked_attention(block, block.ln_1(x), allowed)
        x = x + block.mlp(block.ln_2(x))
    return model.lm_head(model.ln_f(x))


@torch.no_grad()
def batched_greedy_generate(model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None = None) -> list[list[int]]:
    """Greedy-decode every prompt together (no cache: the whole padded batch is re-run each step).

    Returns only the generated tokens per prompt, cut at ``eos_id`` (excluded)
    or after ``max_new_tokens``. The batch stops when block_size is reached.
    """
    idx, mask, pos = left_pad(prompts, pad_id=0)  # any valid id works: the mask hides it
    B, T = idx.shape
    n = min(max_new_tokens, model.cfg.block_size - T)
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * B
    for _ in range(n):
        logits = forward_masked(model, idx, mask, pos)
        next_ids = logits[:, -1, :].argmax(dim=-1)  # (B,)
        for b in range(B):
            if finished[b]:
                continue
            tok = int(next_ids[b])
            if eos_id is not None and tok == eos_id:
                finished[b] = True
            else:
                generated[b].append(tok)
        if all(finished):
            break
        # Append one column: the batch stays rectangular, finished rows just carry on harmlessly.
        idx = torch.cat([idx, next_ids[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.bool)], dim=1)
        pos = torch.cat([pos, pos[:, -1:] + 1], dim=1)
    return generated
