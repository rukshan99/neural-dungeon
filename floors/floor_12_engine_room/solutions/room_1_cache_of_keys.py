"""ROOM 12.1 - THE CACHE OF KEYS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The whole trick: keys and values for a token depend only on that token and its
position, never on what comes after it. Once computed, they can be kept. A
decode step then feeds ONE new token through the network, computes its q, k, v,
appends k and v to the cache, and lets the single query attend over all of them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# One layer's cache: (k, v), each of shape (B, H, T_past, hd).
LayerCache = tuple[torch.Tensor, torch.Tensor]


def attention_with_cache(block, x: torch.Tensor, cache: LayerCache | None = None):
    """Causal self-attention for the NEW tokens ``x`` over cached + new keys and values.

    x:     (B, T_new, C)  already passed through block.ln_1
    cache: None on prefill, else (k_past, v_past) each (B, H, T_past, hd)
    returns (y (B, T_new, C) after c_proj,  new_cache (k_all, v_all) each (B, H, T_past+T_new, hd))
    """
    attn = block.attn
    B, T_new, C = x.shape
    H = attn.n_head
    hd = C // H

    # q, k, v for the new tokens only. This is the only place c_attn runs.
    q, k, v = attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T_new, H, hd).transpose(1, 2)  # (B, H, T_new, hd)
    k = k.view(B, T_new, H, hd).transpose(1, 2)
    v = v.view(B, T_new, H, hd).transpose(1, 2)

    if cache is not None:
        k_past, v_past = cache
        k = torch.cat([k_past, k], dim=2)  # (B, H, T_past + T_new, hd)
        v = torch.cat([v_past, v], dim=2)

    T_total = k.shape[2]
    T_past = T_total - T_new

    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T_new, T_total)
    if T_new > 1:
        # New query i sits at absolute position T_past + i and may see keys 0..T_past + i.
        # For a single decode token (T_new == 1) every key is in its past: no mask needed.
        q_pos = torch.arange(T_past, T_total, device=x.device)[:, None]
        k_pos = torch.arange(T_total, device=x.device)[None, :]
        att = att.masked_fill(k_pos > q_pos, float("-inf"))
    att = F.softmax(att, dim=-1)
    y = (att @ v).transpose(1, 2).contiguous().view(B, T_new, C)
    return attn.c_proj(y), (k, v)


def forward_with_cache(model, idx: torch.Tensor, past_len: int = 0, caches: list[LayerCache] | None = None):
    """Run the GPT on ``idx`` (B, T_new), whose first token sits at absolute position ``past_len``.

    Returns (logits (B, T_new, V), caches: one (k, v) per layer covering past + new tokens).
    """
    B, T_new = idx.shape
    if past_len + T_new > model.cfg.block_size:
        raise ValueError(
            f"positions {past_len}..{past_len + T_new - 1} exceed block_size {model.cfg.block_size}: "
            "the position table has no rows for them"
        )
    pos = torch.arange(past_len, past_len + T_new, dtype=torch.long, device=idx.device)
    x = model.wte(idx) + model.wpe(pos)
    if caches is None:
        caches = [None] * len(model.blocks)
    new_caches: list[LayerCache] = []
    for block, cache in zip(model.blocks, caches):
        y, cache = attention_with_cache(block, block.ln_1(x), cache)
        x = x + y
        x = x + block.mlp(block.ln_2(x))
        new_caches.append(cache)
    logits = model.lm_head(model.ln_f(x))
    return logits, new_caches


@torch.no_grad()
def generate_with_cache(model, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decoding with a KV cache: one prefill pass, then one-token decode steps.

    Block-size policy: STOP. The Chronicler has absolute positions 0..127, so a
    sequence can never be longer than block_size. We generate
    ``min(max_new_tokens, block_size - T)`` tokens and return. (The no-cache
    reference instead crops to the last block_size tokens, which shifts every
    position and would invalidate the cache; a real server reports
    finish_reason="length" here.)
    """
    B, T = idx.shape
    n = min(max_new_tokens, model.cfg.block_size - T)
    if n <= 0:
        return idx
    logits, caches = forward_with_cache(model, idx, 0, None)  # prefill: all prompt tokens at once
    for step in range(n):
        if step > 0:  # decode: only the token we just appended
            logits, caches = forward_with_cache(model, idx[:, -1:], idx.shape[1] - 1, caches)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
