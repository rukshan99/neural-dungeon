"""KV Cache Reference - loot from Floor 12, the Engine Room.

A clean, dependency-light reference for serving the dungeon's GPT (nanoGPT
attribute names: wte, wpe, blocks[i].ln_1 / attn.c_attn / attn.c_proj / ln_2 /
mlp, ln_f, lm_head). Two entry points:

    generate(model, idx, max_new_tokens)             one sequence, KV cache, greedy
    batched_generate(model, prompts, max_new_tokens) many sequences, left-padded,
                                                     KV cache, greedy

Both are token-exact with the no-cache ``model.generate(..., temperature=0)``.
Lift them into your own code; the only assumptions are the attribute names
above and absolute position embeddings (``wpe``) with ``model.cfg.block_size`` rows.

Run this file to see the cache in action:

    python floors/floor_12_engine_room/loot/kv_cache_reference.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

LayerCache = tuple[torch.Tensor, torch.Tensor]  # (k, v), each (B, H, T, hd)


# --------------------------------------------------------------------------- attention
def cached_attention(block, x: torch.Tensor, cache: LayerCache | None, key_mask: torch.Tensor | None = None):
    """Causal self-attention for the new tokens ``x`` (B, T_new, C) over cached + new keys.

    key_mask: optional (B, T_past + T_new) bool, True on real tokens. None means
    "no padding anywhere". Returns (y (B, T_new, C), (k_all, v_all)).
    """
    attn = block.attn
    B, T_new, C = x.shape
    H = attn.n_head
    hd = C // H
    q, k, v = attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T_new, H, hd).transpose(1, 2)
    k = k.view(B, T_new, H, hd).transpose(1, 2)
    v = v.view(B, T_new, H, hd).transpose(1, 2)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    T_total = k.shape[2]
    T_past = T_total - T_new

    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T_new, T_total)
    q_pos = torch.arange(T_past, T_total, device=x.device)[:, None]
    k_pos = torch.arange(T_total, device=x.device)[None, :]
    allowed = (k_pos <= q_pos)[None, None]  # (1, 1, T_new, T_total): causal for the new positions
    if key_mask is not None:
        allowed = allowed & key_mask[:, None, None, :]  # ...and never a padded key
    att = att.masked_fill(~allowed, float("-inf"))
    att = F.softmax(att, dim=-1)
    if key_mask is not None:
        att = torch.nan_to_num(att, nan=0.0)  # fully padded query rows: finite, never read
    y = (att @ v).transpose(1, 2).contiguous().view(B, T_new, C)
    return attn.c_proj(y), (k, v)


# --------------------------------------------------------------------------- forward
def cached_forward(model, idx: torch.Tensor, position_ids: torch.Tensor, caches: list[LayerCache] | None = None,
                   key_mask: torch.Tensor | None = None):
    """One forward over the new tokens ``idx`` (B, T_new) at ``position_ids`` (B, T_new).

    Returns (logits (B, T_new, V), caches). ``caches`` is None on prefill.
    """
    if int(position_ids.max()) >= model.cfg.block_size:
        raise ValueError(f"position {int(position_ids.max())} is beyond block_size {model.cfg.block_size}")
    x = model.wte(idx) + model.wpe(position_ids)
    caches = caches if caches is not None else [None] * len(model.blocks)
    new_caches: list[LayerCache] = []
    for block, cache in zip(model.blocks, caches):
        y, cache = cached_attention(block, block.ln_1(x), cache, key_mask)
        x = x + y
        x = x + block.mlp(block.ln_2(x))
        new_caches.append(cache)
    return model.lm_head(model.ln_f(x)), new_caches


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decoding for one or more equal-length sequences (B, T) with a KV cache.

    Stops at block_size (absolute positions run out). Returns (B, T + n).
    """
    B, T = idx.shape
    n = min(max_new_tokens, model.cfg.block_size - T)
    if n <= 0:
        return idx
    pos = torch.arange(T, device=idx.device)[None].expand(B, T)
    logits, caches = cached_forward(model, idx, pos)  # prefill
    for step in range(n):
        if step > 0:  # decode: the newest token only, at its absolute position
            past = idx.shape[1] - 1
            logits, caches = cached_forward(model, idx[:, -1:], torch.full((B, 1), past, dtype=torch.long), caches)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def left_pad(sequences: list[list[int]], pad_id: int = 0):
    """Ragged token lists -> (idx (B, T), key_mask (B, T) bool, position_ids (B, T))."""
    B, T = len(sequences), max(len(s) for s in sequences)
    idx = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for b, seq in enumerate(sequences):
        idx[b, T - len(seq):] = torch.tensor(seq, dtype=torch.long)
        mask[b, T - len(seq):] = True
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return idx, mask, position_ids


@torch.no_grad()
def batched_generate(model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None = None) -> list[list[int]]:
    """Left-padded, KV-cached greedy decoding for many prompts. Returns generated tokens per prompt."""
    idx, mask, pos = left_pad(prompts)
    B, T = idx.shape
    n = min(max_new_tokens, model.cfg.block_size - T)
    out: list[list[int]] = [[] for _ in prompts]
    done = [False] * B
    if n <= 0:
        return out
    logits, caches = cached_forward(model, idx, pos, None, mask)  # prefill the padded batch
    next_ids = logits[:, -1, :].argmax(dim=-1)
    for step in range(n):
        if step > 0:
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.bool)], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1], dim=1)
            logits, caches = cached_forward(model, next_ids[:, None], pos[:, -1:], caches, mask)
            next_ids = logits[:, -1, :].argmax(dim=-1)
        for b in range(B):
            if done[b]:
                continue
            tok = int(next_ids[b])
            if eos_id is not None and tok == eos_id:
                done[b] = True
            else:
                out[b].append(tok)
        if all(done):
            break
    return out


# --------------------------------------------------------------------------- memory
def kv_cache_bytes(n_layer: int, n_embd: int, seq_len: int, batch: int = 1, bytes_per_value: int = 4) -> int:
    """2 (k and v) x layers x tokens x n_embd x bytes, per sequence, times the batch."""
    return 2 * n_layer * seq_len * n_embd * bytes_per_value * batch


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    from dungeon.artifacts.tiny_gpt import load_pretrained

    torch.set_num_threads(1)
    model, tok, _ = load_pretrained()
    prompt = "Below them, the "
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    t0 = time.perf_counter()
    plain = model.generate(idx, 60, temperature=0)
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cached = generate(model, idx, 60)
    t_cached = time.perf_counter() - t0

    print(tok.decode(cached[0]))
    print(f"token-exact with the no-cache reference: {torch.equal(plain, cached)}")
    print(f"no cache {t_plain * 1e3:.0f} ms, cache {t_cached * 1e3:.0f} ms ({t_plain / t_cached:.1f}x)")
    cfg = model.cfg
    print(f"full-context KV cache: {kv_cache_bytes(cfg.n_layer, cfg.n_embd, cfg.block_size) / 1024:.0f} KiB per sequence")
    batch = batched_generate(model, [tok.encode(p) for p in ["the", "Kasimir waited.", "Below them, the "]], 20)
    for line in batch:
        print(repr(tok.decode(line)))
