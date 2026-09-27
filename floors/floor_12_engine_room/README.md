# Floor 12 — The Engine Room

> *Beneath everything, the machinery that makes the dungeon run. Pipes of keys and values, a crucible for weights, a ferry for petitions, a wall of gauges, and a door to the outside.*

```
            ▼ stairs from Floor 11
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │ 12.1 THE CACHE  │─────│ 12.2 THE QUANTI- │─────│ 12.3 THE BATCHER     │
   │     OF KEYS     │     │    ZER'S BENCH   │     │    (the ferry)       │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐     ┌──────────┴───────────┐
   │ ☠ THE LEVIATHAN │─────│  12.5 THE GATE   │─────│  12.4 THE METER      │
   │     TRENCH      │     │ (to the outside) │     │   (a wall of gauges) │
   └────────┬────────┘     └──────────────────┘     └──────────────────────┘
            ┆  ◇ a spark jumping the trench: the Speculative Spark
            ■  There is nothing below the Engine Room. This is the bottom.
```

The stairs end here. Every floor above taught you to build and train the machine; this one is about *running* it, when someone else is waiting on the other end of a socket. The Chronicler you meet here is the same 802,560-parameter GPT from `dungeon/artifacts/chronicler.pt`. You will not train it. You will make it fast, small, concurrent, measured, and reachable.

Why this matters: a model that is correct but serves one user at four tokens per second is a demo. Production is latency (how long one person waits), throughput (how many tokens per second the machine produces in total), memory (how many requests fit at once) and cost (all three, divided by money). The techniques on this floor are the ones every real inference engine is built from. They are also small enough to write in an afternoon on a laptop CPU, which is the point: you should never again treat "the serving layer" as a black box that happens to somebody else.

**You will learn:** the KV cache and prefill vs decode · int8 quantization and its error bounds · left-padded batching with position ids and masks · latency percentiles, throughput and SLOs · streaming inference over HTTP · speculative decoding.

**You need:** PyTorch (CPU is plenty). Floors 6 and 7 (attention, the GPT block) help; the model's attribute names are listed below so you can also come in cold.

---

## The lore of the Engine Room (read this before the rooms)

### The machine you are optimizing

The Chronicler is a character-level GPT: vocab 72, `block_size` 128, 4 layers, 4 heads, `n_embd` 128, head size 32. Its parts, in nanoGPT's names:

```
model.wte(idx)                 token embedding      (V, C)
model.wpe(pos)                 position embedding   (block_size, C)   absolute positions 0..127
model.blocks[i].ln_1           LayerNorm before attention
model.blocks[i].attn.c_attn    Linear(C, 3C)   fused q, k, v projection
model.blocks[i].attn.c_proj    Linear(C, C)    output projection
model.blocks[i].attn.n_head    4
model.blocks[i].ln_2, .mlp     LayerNorm, then fc -> GELU -> proj
model.ln_f, model.lm_head      final LayerNorm, Linear(C, V) with weight TIED to wte
```

One forward pass over `(B, T)` token ids: `x = wte(idx) + wpe(0..T-1)`, then for every block `x = x + attn(ln_1(x))` and `x = x + mlp(ln_2(x))`, then `lm_head(ln_f(x))`. The model has no hooks for caches, masks or custom positions, so on this floor you write new forward paths *around its weights*. `model.generate(idx, n, temperature=0)` is the honest, slow, greedy reference every room is judged against.

### Why decoding without a cache is quadratic

To emit token T+1, `model.generate` runs all T tokens through the network again. The step at length T costs T units of linear-layer work (every `nn.Linear` sees T rows) plus a `(T x T)` attention. Generating N tokens from a P-token prompt therefore pushes

```
sum_{t=P}^{P+N-1} t  =  N * (P + (N - 1) / 2)   token-rows through the network
```

For P = 60, N = 60 that is 5,370 token-rows. And yet almost all of it is repeated work: **the key and value of position j at every layer depend only on tokens 0..j** (causal masking guarantees nothing later leaks in), so once computed they never change. Recomputing them is the Leviathan's whole diet.

### The KV cache

Keep, per layer, the keys and values you have already computed:

```
k, v : (B, H, T_past, hd)       one pair per layer;  H * hd == n_embd
```

A **decode step** then feeds one new token through the network: `wte` + `wpe(past_len)`, and in every block compute q, k, v *for that token only*, append its k and v to the layer's cache, and let the single query attend over all `T_past + 1` keys. Linear work per step: 1 row instead of T. Attention: `(1 x T)` instead of `(T x T)`. The sequence-level cost drops from quadratic to linear in the length; per token it is essentially constant plus a light `O(T)` attention.

Two phases, and every serving system distinguishes them:

- **Prefill**: the whole prompt in one forward, positions `0..P-1`. Fills the cache. Its duration is the **time to first token** (TTFT). It is a large matmul and is compute-bound.
- **Decode**: one token per forward, at position `past_len`. Each step reads every weight (and the whole cache) to do a tiny amount of arithmetic. On real hardware it is **memory-bandwidth-bound**; on this floor's tiny model it is Python-and-kernel-launch-bound. Either way the arithmetic is not what you are waiting for.

The masking rule when more than one new token arrives at once (prefill, or speculative verification): new query `i` sits at absolute position `T_past + i` and may see keys `0..T_past + i`. With a single new token every key is in its past and no mask is needed.

**Memory.** Per token, per layer, k and v together hold `2 * n_embd` values. So

```
cache_bytes = 2 * n_layer * T * n_embd * bytes_per_value      (times B for a batch)
```

The Chronicler at full context: `2 * 4 * 128 * 128 * 4 = 524,288` bytes, half a MiB, nothing. A 7-billion-parameter class model (32 layers, `n_embd` 4096, fp16) needs `2 * 32 * 4096 * 2 = 524,288` bytes *per token*: half a MiB per token, 2 GiB for a 4,096-token conversation, 32 GiB for sixteen of them. On big models the cache, not the weights, decides how many requests fit on the machine. That is why the memory-shaving tricks in the cheat sheet (paging the cache, quantizing it, sharing keys across heads) exist.

**Positions and the end of the corridor.** The Chronicler's positions are absolute (`wpe` has exactly 128 rows), so a cached sequence can never grow past `block_size`. The reference `generate` crops to the last 128 tokens and re-runs; with a cache that would shift every position and invalidate everything. The reference solution *stops* at `block_size` (a real server would return `finish_reason="length"`). Cropping and re-prefilling is also acceptable; document your choice.

### The Quantizer's Bench: int8

A float32 weight takes 4 bytes. Symmetric absmax int8 quantization of a matrix `W` with shape `(out, in)`:

```
scale = max|W| / 127                 one scale per output row (per-channel) or per matrix (per-tensor)
Q     = round(W / scale)             integers in [-127, 127], stored as torch.int8
W_hat = Q * scale                    the weight the model actually uses
```

The largest-magnitude weight maps to exactly ±127, so nothing is clipped, and round-to-nearest guarantees **|W - W_hat| <= scale / 2 for every weight**. Rounding error is roughly uniform, so its RMS is `scale / sqrt(12)`. For Gaussian-looking weights with `absmax ~ 5 sigma`, the relative error is about `(5 sigma / 127) / (sigma * sqrt(12)) ~ 1.1%`. The Chronicler's linears measure 0.9–1.4%. Its loss on a batch of chronicles moves by about 0.0002, and greedy generations agree with fp32 for 40+ tokens.

**Per-channel vs per-tensor.** Suppose one row of `W` is ten times louder than the rest. Per-tensor, that row sets `scale` for everyone: every quiet row is quantized on a grid ten times coarser than it needed, and its error grows tenfold. Per-channel gives each output row its own scale and the loud row keeps its problem to itself. The trial builds exactly this matrix and measures.

**What the matmul looks like.** `y = x @ W_hat.T + b = x @ (Q * s).T + b = (x @ Q.T) * s.T + b`, because a per-output-row scale factors out of the product. This room's `QuantizedLinear` dequantizes `W_hat` on the fly and runs a float32 matmul: it saves *memory* (4x on the weight), not compute. A real int8 kernel also quantizes `x` (per token, on the fly), multiplies int8 by int8 into an int32 accumulator, and applies both scales at the end. On bandwidth-bound decode steps, reading 4x fewer weight bytes per token is a latency win as well as a size win, which is why quantization is a serving technique and not only a storage one.

**Why `lm_head` is spared.** `model.lm_head.weight is model.wte.weight`: one tensor, two jobs. Quantizing the head would quantize every embedding lookup too (or silently break the tie, giving you two diverging copies), and the head's error lands straight on the logits. Embeddings and heads are commonly kept in higher precision for this reason; the 16 block linears are where the bytes are anyway.

### The Batcher: left padding, positions, masks

Prompts of different lengths cannot share a `(B, T)` tensor without padding. Pad on the **left**: every row's newest token is then in the last column, `logits[:, -1]` is the next-token distribution for every row, and decoding appends one column to the whole batch. (Right padding would leave each row's last real token in a different column and force you to insert new tokens in the middle of the pads.)

Two things make the pads harmless:

1. **Position ids** that start at 0 at each row's first real token: `(mask.cumsum(1) - 1).clamp(min=0)`. A padded row is then embedded exactly as it would be alone.
2. **An attention mask** that is `causal AND key-is-real`, shape `(B, 1, T, T)`, applied as `-inf` *before* the softmax so padded keys receive weight exactly 0.

One trap: a padded *query* at the start of a row sees only padded keys. Every score is `-inf`, the softmax is NaN, and NaN spreads through every matmul after it. Those rows are never read, but they must be finite: `torch.nan_to_num(att, nan=0.0)` after the softmax is the simplest fix.

The batched forward's logits at real positions match the unpadded forward to about 1e-5, and greedy argmaxes agree token for token on this floor's prompts. On hardware where a decode step is bandwidth-bound, a batch of eight costs about the same per step as a batch of one: that is the economic root of batching, and it is *not* free everywhere. Batching a tiny model on a busy CPU can be slower than serving sequentially, because the pads are real work and the matmuls were never the bottleneck. The boss makes you measure this instead of assuming it.

### The Meter: percentiles, throughput, SLOs

A mean hides the tail. If 99 requests take 20 ms and one takes 4 s, the mean says 60 ms and the one user who waited four seconds did not experience a mean. Serving is judged by **percentiles**, and this floor uses **nearest rank**: sort the samples and take the element at 1-based rank `ceil(p/100 * n)`. It never interpolates, so the answer is always a latency that actually happened; p0 is the minimum, p100 the maximum.

```
p50   the median; a typical request
p95   one in twenty is slower
p99   one in a hundred; the number in the contract
```

Why the far needle matters more than it looks: a page that fans out to 10 backend calls hits at least one p99-slow call on `1 - 0.99^10 ~ 10%` of loads. Tails add up across services, so the tail is what you engineer for.

Two more gauges. **Throughput**, tokens per second over the whole run, is what the machine's cost is divided by. **TTFT** is what a human waiting on a stream feels. Latency and throughput pull against each other: batching raises throughput and every individual request's latency at the same time. An **SLO** (service level objective) names both a percentile target and a throughput target, and the honest way to check it is per request, after a warm-up, reporting percentiles rather than means, with a verdict that says *which* target failed.

### The Gate: streaming over HTTP

A model becomes a service the moment something else can call it over a socket. The shape of that service matters more than the framework, so this room uses the standard library:

```
GET  /health    -> {"status": "ok", "model": ...}
POST /generate  <- {"prompt": str, "max_new_tokens": int}  -> {"text", "tokens", "latency_ms"}
POST /stream    <- same body -> newline-delimited JSON: {"token": ...} per token, then {"done": true}
bad input       -> 400 {"error": "..."}        unknown route -> 404 {"error": "..."}
```

Streaming exists because TTFT is what people notice. The stream must actually stream: write and flush each line as the token is produced. With HTTP/1.0 (what `http.server` speaks by default) and no `Content-Length`, the end of the body is the close of the connection, so a client can read line by line as they arrive. Real servers achieve the same with chunked transfer encoding or server-sent events. Two other habits: errors are JSON with a status code, never a stack trace; and generation is serialized behind a lock, because two requests fighting over one CPU is worse for both than taking turns (the batcher is the real answer).

### The Speculative Spark

A decode step costs about the same whether it verifies one token or six. **Speculative decoding** spends that slack: *draft* k tokens cheaply, *verify* them all in one forward over `prompt + draft` (the logits at positions `T-1 .. T+k-1` are exactly what greedy would emit after each prefix), *accept* the longest prefix that matches, then append one token the model chose itself (the correction, or a bonus if the whole draft was right). Every accepted token is precisely greedy's token, so the output is identical; only the number of passes changes. **Prompt lookup** is the cheapest draft model: find the most recent earlier occurrence of the last n tokens in the context and propose what followed it. Text repeats; when it does not, the draft is empty and you are back to plain greedy at no extra cost.

---

## Rooms

Run `dungeon enter 12` to see your progress. Each room is a file in `rooms/`; replace every `raise NotImplementedError` with code and run that room's trial. Rooms may import earlier rooms with a relative import (`from .room_1_cache_of_keys import forward_with_cache`).

### 12.1 The Cache of Keys — `rooms/room_1_cache_of_keys.py`

Pipes labelled K and V run along every wall, and nobody ever connected them. Three functions: `attention_with_cache(block, x, cache)` computes q, k, v for the new tokens only and attends over cached + new keys with a correct causal mask for the new positions; `forward_with_cache(model, idx, past_len, caches)` runs every block with positions `past_len..`; `generate_with_cache(model, idx, max_new_tokens)` does one prefill and then one-token decode steps.

The trial demands token-exact agreement with `model.generate(..., temperature=0)` for 60 tokens from three prompts, checks the cache shapes, counts how many tokens pass through `model.wte` (one each, plus one per step), and races you against the reference on 60 prompt tokens plus 60 new ones: at least 1.5x, on your machine, with one thread, best of three.

```
dungeon trial 12 room_1
```

### 12.2 The Quantizer's Bench — `rooms/room_2_quantizers_bench.py`

`quantize_int8` and `dequantize`, `quantization_error`, a `QuantizedLinear` that stores int8 codes and float32 scales as buffers, and `quantize_model_linears` that deep-copies the Chronicler and replaces its 16 block linears while sparing the tied head. Then the **Prophecy of the Bench**: five predictions about bytes, loud rows, codes and ties, which the trial measures.

The trial checks the half-a-step bound on every weight, that per-channel keeps a loud row to itself, that the int8 Chronicler's loss on a fixed batch of chronicles is within 0.05 of fp32 and its greedy generations agree for 30 tokens, and that the codes take exactly a quarter of the bytes.

```
dungeon trial 12 room_2
```

### 12.3 The Batcher — `rooms/room_3_the_batcher.py`

The ferry takes eight petitions at once. `left_pad` builds `idx`, a boolean `attention_mask` and `position_ids` that start at 0 at each row's first real token; `masked_attention` and `forward_masked` re-implement the forward around the model's weights with a causal-and-key-padding mask; `batched_greedy_generate` decodes every prompt together, stops rows at `eos_id`, and returns only the generated tokens.

The key test: four prompts of different lengths, decoded together, must equal each prompt's individual greedy generation for all 40 tokens. Also: changing the pad id must not move a single real logit, padded rows must be finite, and eos must stop one row while the others carry on.

```
dungeon trial 12 room_3
```

### 12.4 The Meter — `rooms/room_4_the_meter.py`

A wall of gauges. `percentile` (nearest rank) and `latency_summary`; `throughput`; a `LatencyMeter` whose `request()` context manager records duration, token count and time-to-first-token; `run_load`, a sequential harness that drives a token *generator* per request; and `slo_report`, which passes only when p95 and tokens/s both meet their targets and otherwise names what failed.

No model in this room. Synthetic samples with known percentiles, a generator that sleeps before its first token so TTFT must land before the end, and a meter whose mean looks fine while its p95 does not.

```
dungeon trial 12 room_4
```

### 12.5 The Gate — `rooms/room_5_the_gate.py`

The door to the outside, on `http.server.ThreadingHTTPServer`. `greedy_token_stream` yields one token id at a time; `validate_request` turns any body into either a request or an error string; `InferenceServer.start()` binds port 0, reads back the real port and serves from a daemon thread; `stop()` shuts down cleanly; `make_handler` implements the routes.

The trial starts your server, hits it with `urllib`, checks `/health`, asks `/generate` the same question twice, reads `/stream` and checks the chunks add up to the non-streamed text, sends nonsense and expects a polite 400, and closes the door.

```
dungeon trial 12 room_5
```

---

## Boss: The Latency Leviathan

```
              ___
          .-~~   ~~-.        _..--~~~--.._         _..--~~~--.._
        .'  o        `-.._.-~             ~-.._.-~             ~~-.._
       (      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
        `._  ____          _..--~~~--.._         _..--~~~--.._    _.-'
           ~~    ~~-.._.-~             ~~-.._.-~             ~~--~~

     "Every key you compute twice, I keep. Every petition you serve alone,
      I grow a segment. Look how long I have become while you recomputed."
```

**Weakness:** the KV cache and left-padded batching, used *together*, and measured honestly.

- **Phase 1:** `serve(model, prompts, max_new_tokens)`: one prefill over the padded batch, then `(B, 1)` decode steps, with a mask that is causal for the new positions *and* hides every padded key in the cache. Token-exact with `naive_serve` for eight prompts, and at least 3x faster (measured on your machine: eight prompts, 24 tokens each, one thread, best of two).
- **Phase 2:** `leviathan_report`: prefill ms, decode ms per token, p50/p95 per request and tokens per second, from a real `LatencyMeter`. Prefilling a 40–70-token prompt must cost more than one cached decode step.
- **Phase 3:** `LEVIATHAN_PROPHECY`: which of {no cache, cache, batch, cache+batch} is fastest; cache alone vs no cache; whether prefill or decode dominates a 60-token prompt generating 60 tokens; and what the cache changes, the *number* of forward passes or the *rows* inside them. The trial measures all four ways of serving and counts the rows through `wte`.

```
dungeon fight 12
```

## Secret room: The Speculative Spark *(optional)*

Behind the trench, a spark jumps a gap it should not be able to jump. `propose_from_prompt(ids, n, k)` finds the last n-gram earlier in the context and proposes what followed; `speculative_generate` verifies the draft in one forward, accepts the matching prefix and appends the model's own next token. On a chronicle that repeats itself the reference needs 12 forward passes for 40 tokens, and the output is greedy's output to the token.

```
dungeon trial 12 --secret
```

## Loot

Clear the five rooms and defeat the Leviathan to unlock:

- **Inference Optimization Cheat Sheet** — `loot/inference_optimization_cheat_sheet.md`. The KV-cache memory formula, prefill vs decode, int8/int4/fp8 at a glance, batching strategies including continuous batching, percentiles and SLOs, streaming, speculative decoding, and a checklist to run before anything goes live.
- **KV Cache Reference** — `loot/kv_cache_reference.py`. A clean cached forward and left-padded batched greedy generation for the reference GPT, token-exact, ready to lift.

## Stuck?

- `dungeon hint 12 room_1` reveals one hint at a time (three per room).
- The failure messages say what shape or concept is wrong and what was observed. Read them before reading anything else.
- `solutions/` exists. After an honest attempt, compare; do not copy.

When `dungeon map` shows the Engine Room cleared, there are no more stairs. You built the machine, you taught it, and now it runs the way you would have built it. Go and serve something. Then climb back to the surface: `EPILOGUE.md` at the root of the dungeon is waiting for you.
