# Inference Optimization Cheat Sheet

*Loot from Floor 12. One page for the day the model has to actually run.*

## The two phases

| | Prefill | Decode |
|---|---|---|
| What | the whole prompt in one forward | one token per forward, with the KV cache |
| Cost | big matmul over P tokens; compute-bound | reads every weight and the whole cache to do a little arithmetic; **memory-bandwidth-bound** |
| Sets | time to first token (TTFT) | inter-token latency, tokens/s |
| Scales with | prompt length | number of generated tokens (and, gently, context length via attention) |

Decode dominates most requests: 59 cached decode steps outweigh one 60-token prefill by a wide margin. Without a cache, generating N tokens from P re-embeds `N * (P + (N - 1) / 2)` token-rows; with a cache, `P + N - 1`.

## KV cache

Per layer: `k, v : (B, H, T, hd)`, `H * hd == n_embd`. Keys and values of position j depend only on tokens `0..j`; once computed they never change.

```
cache_bytes = 2 * n_layer * T * n_embd * bytes_per_value * B
```

| Model | Per token | 4k context, one sequence |
|---|---|---|
| Chronicler (4 layers, 128, fp32) | 4 KiB | 512 KiB at its 128-token limit |
| 7B class (32 layers, 4096, fp16) | 512 KiB | 2 GiB |
| 70B class (80 layers, 8192, fp16) | 2.5 MiB | 10 GiB |

On big models the cache, not the weights, decides how many requests fit. Ways to shrink it: **grouped-query / multi-query attention** (several query heads share one k/v head: 4–8x smaller), **paged attention** (allocate the cache in fixed blocks so sequences do not reserve their maximum length up front and memory is not fragmented), **cache quantization** (int8/fp8 k and v), **sliding windows** (only the last W keys, with a model trained for it).

Decode-step mask: the new query at absolute position `T_past + i` may see keys `0..T_past + i`; with one new token no mask is needed. Positions are absolute in the dungeon's GPT (`wpe`, `block_size` rows), so a cached sequence stops at `block_size`. Rotary/relative positions relax that but not the cache's memory.

## Quantization at a glance

Symmetric absmax: `scale = max|W| / qmax`, `Q = round(W / scale)`, `W_hat = Q * scale`, error `<= scale / 2` per weight, RMS `~ scale / sqrt(12)`. Per-channel (one scale per output row) beats per-tensor whenever one row is louder than the others, which is always.

| Format | Bytes/weight | Typical use | Watch out for |
|---|---|---|---|
| fp32 | 4 | training, reference | memory |
| bf16 / fp16 | 2 | default serving precision | fp16 overflows; bf16 has fewer mantissa bits |
| int8 (weights) | 1 | 4x smaller than fp32, ~1% relative error, usually lossless in quality | activation outliers if you quantize activations too |
| int8 (weights + activations) | 1 | true int8 kernels, 2–4x faster matmuls | per-token activation scales; outlier channels kept in higher precision |
| int4 (weights, grouped) | 0.5 | fits big models on small GPUs | group-wise scales (e.g. 128 weights per scale); needs calibration data or error-compensating rounding; slower per-token dequant |
| fp8 (e4m3 / e5m2) | 1 | modern accelerators; weights, activations, KV cache | hardware support; scaling still needed |

Rules of thumb: keep embeddings, `lm_head` and LayerNorms in higher precision; quantize the big linears; measure loss/perplexity and a few greedy generations *before and after*; dequantize-on-the-fly saves memory only, a real kernel is what saves time. On bandwidth-bound decode, fewer weight bytes per token is directly lower latency.

## Batching

- **Static batching**: pad a fixed group, run it to completion. Simple; the group finishes at the pace of its longest member and short requests wait.
- **Left padding**: newest token in the last column for every row; `logits[:, -1]` is everyone's next-token distribution; decoding appends a column.
- **Position ids**: `(mask.cumsum(1) - 1).clamp(min=0)`, so a padded row is embedded exactly as it would be alone.
- **Mask**: `causal & key_is_real`, `-inf` before the softmax; `nan_to_num` for fully padded query rows; the pads stay in the KV cache and must stay masked there.
- **Continuous batching** (also called in-flight batching): the scheduler works at the granularity of a *decode step*, not a request. After every step it drops finished sequences and admits waiting ones into the freed slots, so the batch is always as full as memory allows and a long request never holds short ones hostage. Prefill of a new arrival is either run as its own step or chunked and interleaved with decode steps so TTFT and inter-token latency both stay bounded. Paged attention is what makes admitting a new sequence cheap: its cache is a list of blocks, not one contiguous reservation.
- Batching pays when the step is bandwidth-bound (GPU decode: eight sequences cost about the same per step as one). It does not pay for free: on a busy CPU with a tiny model, sequential can beat batched. Measure.

## Latency, throughput, SLOs

- **Nearest-rank percentile**: sort; take the element at 1-based rank `ceil(p/100 * n)`. Never interpolate: report a latency that actually happened.
- **p50** typical, **p95** one in twenty, **p99** the contract. Tail amplification: 10 fan-out calls hit a p99 on `1 - 0.99^10 ~ 10%` of requests.
- **TTFT** (prefill) is what a human feels; **inter-token latency** is reading speed (~50 ms/token is comfortable); **tokens/s** over the whole run is what you divide the bill by.
- **Latency vs throughput**: a bigger batch raises both. Find the knee, then set the SLO on the tail.
- **Honest measurement**: warm up first (allocator, kernels, caches), then time per request, several repeats, report min/median for micro-benchmarks and percentiles for load, same machine and thread count for any comparison, ratios rather than absolute numbers when hardware varies.
- **SLO**: name a percentile target and a throughput target; the report says pass/fail and *which* target broke.

## Streaming

- Token-by-token output turns a 3-second wait into a 300 ms wait plus reading.
- Wire formats: newline-delimited JSON, chunked transfer encoding, server-sent events (`data: ...\n\n`), WebSockets. HTTP/1.0 with no `Content-Length` ends the body when the connection closes, so a stdlib server can stream by flushing lines.
- Flush after every token; send a terminal `{"done": true}` (and usage/finish reason) so the client can tell "finished" from "connection dropped".
- Errors are JSON with a status code, never a traceback; validate the body before touching the model; serialize or batch model access, never let two requests fight over one accelerator.

## Speculative decoding

- **Draft** k tokens cheaply (a small model, an n-gram / prompt lookup, extra heads); **verify** all of them in one forward over `context + draft`; **accept** the longest matching prefix and append the model's own token after it.
- Output is *identical* to greedy (or, with rejection sampling, to the target distribution). Speedup = accepted tokens per pass; it is high on repetitive or predictable text and falls back to 1 token/pass otherwise.
- Prompt lookup: propose what followed the most recent earlier occurrence of the last n tokens; zero extra model cost.

## Pre-launch checklist

- [ ] Greedy generations from the serving path are token-exact with the reference model (cache, batching, quantization, speculation all on).
- [ ] Loss / perplexity of the quantized model measured on a fixed batch against fp32; a few generations eyeballed.
- [ ] KV-cache memory budgeted: `2 * n_layer * T_max * n_embd * bytes * B_max` fits, with headroom for prefill activations.
- [ ] Sequence-length and token-budget limits enforced at the door (400, not a crash), and `finish_reason` reported.
- [ ] Warm-up run before any measurement; percentiles (p50/p95/p99) and TTFT recorded per request; tokens/s over the run.
- [ ] SLO written down: p95 or p99 latency target, TTFT target, throughput target; the report names what fails.
- [ ] Streaming verified end-to-end: chunks arrive incrementally and concatenate to the non-streamed answer.
- [ ] Health endpoint, structured JSON errors, request validation, timeouts, and a lock or scheduler in front of the model.
- [ ] Determinism understood: greedy is deterministic; sampling is seeded or accepted as non-deterministic; batch composition can change floats at the 1e-5 level.
