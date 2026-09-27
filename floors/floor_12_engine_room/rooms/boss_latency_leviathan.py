"""BOSS - THE LATENCY LEVIATHAN

              ___
          .-~~   ~~-.        _..--~~~--.._         _..--~~~--.._
        .'  o        `-.._.-~             ~-.._.-~             ~~-.._
       (      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
        `._  ____          _..--~~~--.._         _..--~~~--.._    _.-'
           ~~    ~~-.._.-~             ~~-.._.-~             ~~--~~

    "Every key you compute twice, I keep. Every petition you serve alone,
     I grow a segment. Look how long I have become while you recomputed."

    It lives in the flooded trench below the ferry landing, and it is exactly
    as long as the sum of every token the Chronicler has ever re-embedded.

WEAKNESS: the KV cache and left-padded batching, used TOGETHER, and measured
honestly with the meter from room 4.

  Phase 1  serve()  - batched (room 3) AND cached (room 1) greedy decoding for a
           list of prompts. Token-exact with the naive baseline, and much faster.
  Phase 2  leviathan_report() - prefill time, decode time per token, p50/p95 per
           request and tokens per second, from a real LatencyMeter.
  Phase 3  LEVIATHAN_PROPHECY - rank the four ways of serving before you measure.

The forward you need does not exist yet: room 1 caches but cannot mask, room 3
masks but cannot cache. Write the attention that does both: q, k, v for the new
tokens, k and v concatenated behind the cache, a mask that is causal for the
new positions AND hides every padded key (the pads are in the cache too), and a
finite result for fully padded query rows.

Run:  dungeon fight 12
"""

from __future__ import annotations

import torch

from .room_4_the_meter import LatencyMeter

LayerCache = tuple[torch.Tensor, torch.Tensor]


def naive_serve(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """The baseline the Leviathan feeds on: one prompt at a time, ``model.generate`` with
    ``temperature=0`` and no cache. Return only the generated tokens per prompt."""
    raise NotImplementedError("naive_serve() is unwritten")


def cached_masked_attention(block, x: torch.Tensor, cache: LayerCache | None, key_mask: torch.Tensor):
    """Cached attention for a left-padded batch.

    x: (B, T_new, C) after ``ln_1``. cache: None or (k, v) each (B, H, T_past, hd).
    key_mask: (B, T_past + T_new) bool over ALL keys, True on real tokens.
    allowed[b, :, i, j] = (j <= T_past + i) and key_mask[b, j]. -inf, softmax,
    nan_to_num, then c_proj. Returns (y (B, T_new, C), (k_all, v_all)).
    """
    raise NotImplementedError("cached_masked_attention() is unwritten")


def batched_forward_with_cache(model, idx: torch.Tensor, position_ids: torch.Tensor,
                               key_mask: torch.Tensor, caches: list[LayerCache] | None = None):
    """idx (B, T_new), position_ids (B, T_new), key_mask (B, T_past + T_new).

    Embed with ``wte(idx) + wpe(position_ids)``, run every block with
    ``cached_masked_attention``, finish with ``ln_f`` and ``lm_head``.
    Returns (logits (B, T_new, V), new_caches).
    """
    raise NotImplementedError("batched_forward_with_cache() is unwritten")


def serve(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Left-pad (room 3's ``left_pad``), ONE prefill over the padded batch, then one
    (B, 1) decode step per new token, appending to key_mask (True) and position_ids
    (previous + 1) each step. Greedy. Returns only the generated tokens per prompt;
    stops when block_size is reached. ``@torch.no_grad()``.
    """
    raise NotImplementedError("serve() is unwritten")


def leviathan_report(model, prompts: list[list[int]], max_new_tokens: int, meter: LatencyMeter | None = None) -> dict:
    """Serve each prompt on its own WITH a cache, timing every phase, and report:

      {"requests", "prefill_ms" (mean prefill forward), "decode_ms_per_token" (mean decode
       step), "p50_ms", "p95_ms" (per-request latency), "tokens", "tokens_per_second"}

    Use ``meter.request()`` around each request, ``req.first_token()`` right after the
    prefill, ``req.tokens = n`` at the end; the percentiles come from ``meter.summary()``.
    A fresh LatencyMeter if none is given.
    """
    raise NotImplementedError("leviathan_report() is unwritten")


# ---------------------------------------------------------------------------
# PHASE 3: THE PROPHECY
# Fill in every None BEFORE running the fight. The trial measures all of it.
# The four ways of serving 8 prompts: "no cache" (naive_serve), "cache"
# (room 1's generate_with_cache, one prompt at a time), "batch" (room 3's
# batched_greedy_generate), "cache+batch" (your serve()).
# ---------------------------------------------------------------------------
LEVIATHAN_PROPHECY: dict[str, str | None] = {
    # Which of the four has the highest tokens per second?
    "fastest_of_four": None,
    # Cache alone versus no cache, one prompt at a time: "faster" | "slower" | "same"
    "cache_alone_vs_no_cache": None,
    # One 60-token prompt generating 60 tokens WITH a cache: which phase takes longer in
    # total, "prefill" or "decode"?
    "dominant_phase_60_60": None,
    # Generating N tokens with a cache versus without: the NUMBER of forward passes is
    # "fewer" | "same" | "more"
    "forward_passes_with_cache": None,
    # ...and the number of token ROWS that go through the embedding is "fewer" | "same" | "more"
    "token_rows_with_cache": None,
}
