"""ROOM 12.1 - THE CACHE OF KEYS

    Pipes run along every wall of the Engine Room, and each one is labelled
    with a layer number and a letter: K or V. The Chronicler upstairs has been
    recomputing every key and value for every token, every step, since the
    day it was built. The pipes were installed for a reason. Nobody connected
    them.

The reference ``model.generate`` is honest and slow: to emit token number T+1 it
runs the whole prefix of T tokens through the network again. Keys and values for
a token depend only on that token and its position, never on what comes after,
so every step recomputes T-1 things it already knew. The step at length T costs
T times the per-token linear work plus a (T x T) attention; summed over N new
tokens that is quadratic in the sequence length.

A KV cache keeps k and v per layer, shape (B, H, T_past, hd), and feeds only the
NEW token(s) through the network. A decode step then costs one token of linear
work plus a (1 x T) attention over the cache: linear in T, and mostly constant
overhead at this model's size.

Two phases, which every serving system distinguishes:

    prefill  the whole prompt in one forward; fills the cache; sets time-to-first-token
    decode   one token per forward; appends one column to every cache

Reuse the pretrained weights: ``block.attn.c_attn`` (fused q, k, v projection),
``block.attn.c_proj``, ``block.ln_1`` / ``ln_2`` / ``mlp``, ``model.wte``, ``model.wpe``,
``model.ln_f``, ``model.lm_head``. Retrain nothing. Decorate the generation loop
with ``@torch.no_grad()``: autograd bookkeeping is pure overhead at inference.

Positions matter: the token you decode at step s sits at absolute position
past_len + s, and ``wpe`` has exactly ``block_size`` rows. Decide what happens
when the corridor ends, and say so in your docstring.
"""

from __future__ import annotations

import torch

# One layer's cache: (k, v), each of shape (B, H, T_past, hd).
LayerCache = tuple[torch.Tensor, torch.Tensor]


def attention_with_cache(block, x: torch.Tensor, cache: LayerCache | None = None):
    """Causal self-attention for the NEW tokens only, attending over cached + new keys.

    block:  one ``model.blocks[i]``; use ``block.attn.c_attn``, ``block.attn.c_proj``,
            ``block.attn.n_head``.
    x:      (B, T_new, C) - the new tokens' activations, already through ``block.ln_1``.
    cache:  None (prefill) or ``(k_past, v_past)`` each (B, H, T_past, hd).

    Returns ``(y, new_cache)``: y is (B, T_new, C) after ``c_proj``; new_cache is
    ``(k_all, v_all)`` each (B, H, T_past + T_new, hd) - the cache to pass next time.

    Compute q, k, v for x only, concatenate k and v behind the cached ones along the
    time axis, then attend. Causal mask for the new positions: new query i sits at
    absolute position T_past + i and may see keys 0..T_past + i. With T_new == 1
    every key is in its past and no mask is needed. Scale by 1/sqrt(hd).
    """
    raise NotImplementedError("attention_with_cache() is unwritten")


def forward_with_cache(model, idx: torch.Tensor, past_len: int = 0, caches: list[LayerCache] | None = None):
    """Run the GPT on ``idx`` (B, T_new) whose first token sits at absolute position ``past_len``.

    Position ids are ``past_len .. past_len + T_new - 1`` (that is what ``model.wpe``
    is indexed with). ``caches`` is None on prefill or one LayerCache per block.
    Run every block: ``x = x + attention_with_cache(block, block.ln_1(x), cache)``,
    then ``x = x + block.mlp(block.ln_2(x))``; finish with ``ln_f`` and ``lm_head``.

    Returns ``(logits (B, T_new, V), new_caches)`` with one entry per layer.
    Raise ValueError if ``past_len + T_new > model.cfg.block_size``: there are no
    position rows for those tokens.
    """
    raise NotImplementedError("forward_with_cache() is unwritten")


def generate_with_cache(model, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decoding with a KV cache. Returns idx with the new tokens appended, (B, T + n).

    One prefill pass over ``idx`` (all prompt tokens at once), then one-token
    decode steps: feed only the token you just appended, at position T_past.
    Greedy means argmax of the last position's logits; the result must equal
    ``model.generate(idx, n, temperature=0)`` token for token while the sequence
    fits in block_size.

    Block-size policy: the Chronicler has absolute positions 0..block_size-1, so a
    sequence can never be longer than block_size. Either STOP early (generate
    min(max_new_tokens, block_size - T) tokens) or CROP and re-prefill. Document
    your choice in this docstring; the trial accepts either. STOP must return
    exactly the tokens that fit and never exceed block_size; CROP must stay
    token-exact with ``model.generate``, which crops the same way.

    Use ``@torch.no_grad()``. Budget: exactly one embedding lookup per token of
    prompt and per generated token. The trial counts.
    """
    raise NotImplementedError("generate_with_cache() is unwritten")
