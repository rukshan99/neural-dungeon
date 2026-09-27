"""BOSS - THE LATENCY LEVIATHAN  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Leviathan grows a segment for every token you recompute. Two weapons,
used together: a KV cache (never recompute a key) and left-padded batching
(compute eight requests' new tokens in one forward). The forward below combines
room 1's cached attention with room 3's key-padding mask.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .room_1_cache_of_keys import forward_with_cache
from .room_3_the_batcher import left_pad
from .room_4_the_meter import LatencyMeter

LayerCache = tuple[torch.Tensor, torch.Tensor]


def naive_serve(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """The baseline: one prompt at a time, no cache, the reference generate()."""
    out = []
    with torch.no_grad():
        for p in prompts:
            idx = torch.tensor([p], dtype=torch.long)
            full = model.generate(idx, max_new_tokens, temperature=0)
            out.append(full[0, len(p):].tolist())
    return out


def cached_masked_attention(block, x: torch.Tensor, cache: LayerCache | None, key_mask: torch.Tensor):
    """Cached attention for a padded batch.

    x: (B, T_new, C) after ln_1; cache: None or (k, v) (B, H, T_past, hd);
    key_mask: (B, T_past + T_new) bool, True on real tokens (pads included in the past).
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
    allowed = (k_pos <= q_pos)[None, None] & key_mask[:, None, None, :]  # causal AND not a pad
    att = att.masked_fill(~allowed, float("-inf"))
    att = torch.nan_to_num(F.softmax(att, dim=-1), nan=0.0)  # fully padded query rows -> 0
    y = (att @ v).transpose(1, 2).contiguous().view(B, T_new, C)
    return attn.c_proj(y), (k, v)


def batched_forward_with_cache(model, idx: torch.Tensor, position_ids: torch.Tensor,
                               key_mask: torch.Tensor, caches: list[LayerCache] | None = None):
    """Cached + masked forward. idx (B, T_new), position_ids (B, T_new), key_mask (B, T_past + T_new)."""
    x = model.wte(idx) + model.wpe(position_ids)
    if caches is None:
        caches = [None] * len(model.blocks)
    new_caches = []
    for block, cache in zip(model.blocks, caches):
        y, cache = cached_masked_attention(block, block.ln_1(x), cache, key_mask)
        x = x + y
        x = x + block.mlp(block.ln_2(x))
        new_caches.append(cache)
    return model.lm_head(model.ln_f(x)), new_caches


@torch.no_grad()
def serve(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Left-padded batching WITH a KV cache: one prefill, then one (B, 1) decode step per token."""
    idx, mask, pos = left_pad(prompts, pad_id=0)
    B, T = idx.shape
    n = min(max_new_tokens, model.cfg.block_size - T)
    generated: list[list[int]] = [[] for _ in prompts]
    if n <= 0:
        return generated
    logits, caches = batched_forward_with_cache(model, idx, pos, mask)  # prefill
    next_ids = logits[:, -1, :].argmax(dim=-1)
    for step in range(n):
        if step > 0:
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.bool)], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1], dim=1)
            logits, caches = batched_forward_with_cache(model, next_ids[:, None], pos[:, -1:], mask, caches)
            next_ids = logits[:, -1, :].argmax(dim=-1)
        for b in range(B):
            generated[b].append(int(next_ids[b]))
    return generated


@torch.no_grad()
def leviathan_report(model, prompts: list[list[int]], max_new_tokens: int, meter: LatencyMeter | None = None) -> dict:
    """Serve each prompt on its own with a cache and meter it honestly.

    prefill_ms: mean time of the prefill forward; decode_ms_per_token: mean time
    of one decode step; p50_ms / p95_ms: per-request latency; tokens_per_second.
    """
    meter = meter if meter is not None else LatencyMeter()
    prefill_times: list[float] = []
    decode_times: list[float] = []
    for p in prompts:
        idx = torch.tensor([p], dtype=torch.long)
        n = min(max_new_tokens, model.cfg.block_size - idx.shape[1])
        with meter.request() as req:
            t0 = time.perf_counter()
            logits, caches = forward_with_cache(model, idx, 0, None)
            prefill_times.append(time.perf_counter() - t0)
            req.first_token()  # the first token is decided by the prefill logits
            past = idx.shape[1]
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(n - 1):
                t0 = time.perf_counter()
                logits, caches = forward_with_cache(model, next_id, past, caches)
                decode_times.append(time.perf_counter() - t0)
                past += 1
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            req.tokens = n
    summary = meter.summary()
    return {
        "requests": len(prompts),
        "prefill_ms": 1000.0 * sum(prefill_times) / len(prefill_times),
        "decode_ms_per_token": 1000.0 * sum(decode_times) / max(len(decode_times), 1),
        "p50_ms": summary["latency_ms"]["p50"],
        "p95_ms": summary["latency_ms"]["p95"],
        "tokens": summary["tokens"],
        "tokens_per_second": summary["tokens_per_second"],
    }


# ---------------------------------------------------------------------------
# PHASE 3: THE PROPHECY, answered. Every line is measured by the trial.
# ---------------------------------------------------------------------------
LEVIATHAN_PROPHECY: dict[str, str | None] = {
    "fastest_of_four": "cache+batch",
    "cache_alone_vs_no_cache": "faster",
    "dominant_phase_60_60": "decode",  # 59 decode steps outweigh one 60-token prefill
    "forward_passes_with_cache": "same",  # one pass per token either way...
    "token_rows_with_cache": "fewer",  # ...but each pass embeds one token, not the whole prefix
}
