# Floor 6 — The Hall of a Thousand Heads

> *A thousand carved heads line the hall. Each one listens to every voice at once and weighs them. A veil keeps them from hearing what has not yet been said.*

```
            ▼ stairs from Floor 5
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │  6.1 THE SINGLE │─────│  6.2 THE VEIL OF │─────│  6.3 THE THOUSAND    │
   │      GAZE       │     │     CAUSALITY    │     │        HEADS         │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐     ┌──────────┴───────────┐
   │ ☠ THE ORACLE'S  │─────│  6.5 THE ROTARY  │─────│  6.4 THE PADDING     │
   │      DAIS       │     │     GALLERY      │     │        VEIL          │
   └────────┬────────┘     └──────────────────┘     └──────────────────────┘
            ┆  ◇ a slit behind the dais: the Tiled Gaze
            ▼  stairs down to Floor 7
```

The stairs open onto a hall so long its far end is lost in torchlight. Along both walls, carved into the stone in rows up to the vault, are heads. Hundreds of them. When you speak, every head turns a little, not towards you but towards each other, as though deciding together how much of what you said is worth keeping.

That is attention, and it is the operation the rest of the dungeon is built from. A transformer is a stack of these halls. Every token in a sequence asks every other token "how relevant are you to me?", gets a number back, and takes a weighted average of what the relevant tokens carry. It is two matrix multiplications with a softmax between them, and you could write it in five lines. The reason it gets a whole floor is that the five lines have three places to be silently wrong: the softmax over the wrong axis, a missing `sqrt(d)`, and, worst of all, a mask that lets a head hear the future. That last one does not crash. It does not even make the loss worse. It makes the loss *wonderful*, and then the model you trained cannot generate a sentence. The Oracle at the end of this hall is such a model, and your job is to catch it the only way it can be caught: empirically.

**You will learn:** scaled dot-product attention · the softmax over keys and why the scores are divided by `sqrt(d_k)` · causal masks applied before the softmax with `-inf` · key-padding masks and how to combine them · fully-masked rows and NaN · multi-head attention with split/merge heads and an output projection, interchangeable with `nn.MultiheadAttention` · detecting information leakage by perturbation · rotary position embeddings (RoPE) and why a rotation gives relative position through a dot product · grouped-query attention, multi-query attention and the KV cache they shrink · online-softmax (tiled) attention.

**You need:** PyTorch (CPU is plenty), `nn.Module` and `nn.Linear` from Floor 4, and the shape discipline of Floor 0. Every room here is shapes.

---

## The lore of attention (read this before the rooms)

### One query, many keys: a weighted average

Start with one head and one position. The position holds a **query** vector `q` of length `d`. Every position in the sequence (including this one) offers a **key** vector `k_j` and a **value** vector `v_j`. The key is what a position *advertises*; the value is what it *hands over* if asked.

```
score_j  = (q · k_j) / sqrt(d)          how relevant is position j to this query
weight_j = softmax(score)_j             = exp(score_j) / Σ_i exp(score_i)
out      = Σ_j weight_j · v_j           a weighted average of the values
```

The weights are non-negative and sum to 1, so `out` lives inside the convex hull of the values: attention *blends*, it never invents. In the hall, `q` is a head asking, the `k_j` are the voices answering, and `out` is what the head walks away having heard.

### Batched: two matmuls and a softmax

Stack the queries of all `Tq` positions into `q` of shape `(B, Tq, d)`, and the keys and values of all `Tk` positions into `k` `(B, Tk, d)` and `v` `(B, Tk, dv)`:

```
scores  = q @ k.transpose(-2, -1) / sqrt(d)     (B, Tq, d) x (B, d, Tk) -> (B, Tq, Tk)
weights = softmax(scores, dim=-1)               (B, Tq, Tk)   each row sums to 1
out     = weights @ v                           (B, Tq, Tk) x (B, Tk, dv) -> (B, Tq, dv)
```

Row `i` of `weights` belongs to query `i` and says how it splits its hearing across the `Tk` keys. That is why the softmax runs over the **last** axis, the keys. A softmax over `dim=-2` would make each *key* distribute itself over the queries, which is a different (and wrong) operation that produces no error.

In self-attention, `q`, `k`, `v` all come from the same sequence (`Tq == Tk == T`). In cross-attention they come from two different sequences. The code is identical.

### Why `sqrt(d)`

Suppose the entries of `q` and `k_j` are independent with mean 0 and variance 1. Then `q · k_j = Σ_i q_i k_{j,i}` is a sum of `d` uncorrelated terms, each with variance 1, so it has mean 0 and **variance `d`**, standard deviation `sqrt(d)`.

Softmax is exponential in its input. Scores spread over 2 units give a soft distribution; scores spread over 30 units give a distribution where one entry is `0.9999` and the rest are dust (`softmax([10, 0, 0])` puts `0.9999` on the first entry). Two things go wrong when that happens without your asking for it: the head can only ever hear one voice, and the gradient of the softmax, `p(1 - p)`, is nearly zero everywhere, so nothing upstream learns. Dividing by `sqrt(d)` makes the variance 1 for every `d`. Room 6.1 has you predict this effect and then measure it. The numbers are in the loot.

Inside multi-head attention, `d` is the **head dimension**, `d_model / H`, because that is the length of the vectors being dotted. Scaling by `sqrt(d_model)` is a common bug that no shape check catches.

### Masks: silence a voice *before* the softmax

To forbid query `i` from hearing key `j`, set `scores[i, j] = -inf` before the softmax. `exp(-inf)` is exactly `0.0`, so the key gets zero weight **and drops out of the denominator**: the surviving weights still sum to 1 and are exactly the softmax over the allowed keys alone.

```python
scores = scores.masked_fill(~mask, float("-inf"))   # mask: bool, True = may attend
weights = scores.softmax(dim=-1)
```

Multiplying the weights by the mask *after* the softmax looks similar and is wrong twice: the rows no longer sum to 1, and the forbidden keys are still inside the denominator, so their contents still change the visible weights. That second effect is a leak, and Room 6.2 has you catch it.

On this floor a boolean mask means **`True` = may attend**, which is also what `torch.nn.functional.scaled_dot_product_attention(attn_mask=...)` means. `torch.nn.MultiheadAttention` uses the **opposite** convention for its boolean `attn_mask` and `key_padding_mask` (`True` = ignore). Many codebases instead use *additive* float masks (`0` for attend, `-inf` for ignore) that are simply added to the scores. All three are equivalent; check which one you are holding.

### The causal veil

A language model at position `t` must predict token `t+1` using only tokens `0..t`. If it can see token `t+1` while training, it learns to copy it, the training loss collapses towards zero, and at generation time, when there is no `t+1` to copy, it produces nonsense. The **causal mask** is a lower-triangular boolean matrix:

```
mask[i, j] = (j <= i)            query i may hear key j iff j is not after i

T = 4:   1 0 0 0
         1 1 0 0                  torch.tril(torch.ones(T, T, dtype=torch.bool))
         1 1 1 0
         1 1 1 1
```

Applied as `-inf` before the softmax, every weight strictly above the diagonal is exactly zero. Position 0 hears only itself, so its output is exactly `v[0]`.

You cannot verify causality by reading the loss. You verify it by **perturbation**: rewrite the inputs at positions after `t` and check that the outputs at positions up to `t` did not move. Honest causal attention passes bitwise, because `0.0 * anything` is `0.0`. A mask applied after the softmax fails immediately. A mask shifted by one seat (`tril(..., diagonal=1)`) fails at exactly the pairs `(t, t+1)`. This test is Room 6.2's `leaks_future` and the boss's `leak_pairs`.

### The padding veil, and combining veils

Sequences of different lengths share a batch by being padded to a common `T`. The padded positions hold arbitrary numbers and must not be *heard*. That is a mask on the **key** axis that differs per batch element:

```python
padding  = torch.arange(T)[None, :] < lengths[:, None]      # (B, T)  True at real tokens
combined = causal[None, None, :, :] & padding[:, None, None, :]   # (B, 1, T, T)
```

Read the four axes of `combined`: batch, head, query, key. The head axis is a `1` so one mask broadcasts across every head. The padding enters along the *last* axis because a padded token is hidden as a key. Masking the *query* axis instead is a classic bug: it silences the padded rows (which nobody reads anyway) and leaves the real queries hearing the padding. The boss has an oracle that does exactly that.

The test that matters: run a padded batch, then run each sequence alone and unpadded. The real positions must agree. If they do not, the model you serve behaves differently from the model you trained.

### Fully masked rows

A row whose every key is masked is a row of all `-inf`, and `softmax` of that row is `NaN` (the max-subtraction computes `(-inf) - (-inf)`). One NaN row poisons the gradient of the whole batch. It happens with a length-0 sequence, or with any masking bug. Guard it: after the softmax, find rows with no `True` in the mask (`mask.any(-1, keepdim=True)`) and set them to zero. Silence, not poison. Torch's fused attention has changed its behaviour on such rows across versions; write the guard yourself.

### A thousand heads

One head computes one weighting of the keys per query. **Multi-head attention** computes `H` different weightings in parallel, each in its own `head_dim = d_model / H` dimensional slice of the model dimension, and lets each head hear something different: one may track who is speaking, one the previous word, one the punctuation.

```
x                       (B, T, d_model)
qkv = Linear(x)         (B, T, 3 * d_model)      one fused matmul for q, k and v
q, k, v = split         (B, T, d_model) each
split heads             (B, H, T, head_dim)      view(B, T, H, hd).transpose(1, 2)
scores = q @ k^T / sqrt(head_dim)                (B, H, T, T)
weights = softmax(-1)                            (B, H, T, T)   rows sum to 1
heads = weights @ v                              (B, H, T, head_dim)
merge heads             (B, T, d_model)          transpose(1, 2).contiguous().view(B, T, d_model)
out = proj(...)         (B, T, d_model)
```

Two things to notice. The attention in the middle is *exactly* the single-head function from Room 6.1 with `(B, H)` as batch axes, which is why you write that function to touch only the last two dims. And the head split is `view` **then** `transpose`: head `h` owns columns `h*hd : (h+1)*hd` of the model dimension. A direct `view(B, H, T, hd)` runs without error and scrambles the heads.

Parameter count: `qkv` has `3·d_model² + 3·d_model`, `proj` has `d_model² + d_model`, total **`4·d_model² + 4·d_model`**. That is exactly `nn.MultiheadAttention`'s count, and not by coincidence: torch stores the same fused `in_proj_weight` `(3d, d)` and `out_proj`, with the same contiguous head chunks. Copy your `qkv.weight` into `in_proj_weight`, your `proj` into `out_proj`, and the outputs match to float precision. Room 6.3 makes you prove it. The same layout, under the names `c_attn` and `c_proj`, is the attention inside the Chronicler you build on Floor 7.

Cost: time `O(T² · d_model)` per layer, memory `O(H · T²)` for the weights. The `T²` is why long contexts are expensive, and why the secret room exists.

### The rotary gallery: positions as rotations

Everything above is blind to order. Permute the tokens of `x` and multi-head attention permutes its outputs the same way and changes nothing else; Room 6.5 has you demonstrate it. "the cat sat" and "sat the cat" are the same bag of vectors. A transformer needs positions from somewhere. GPT-2 adds a learned vector per position to the token embedding before the first layer (Floor 7's `wpe`: one row per position, none for position `block_size`). Nearly every model since uses **rotary position embeddings** (RoPE, Su et al. 2021): no table, no parameters, and the position goes into `q` and `k` inside every attention layer, as a rotation.

Take one head's query at position `m`, `head_dim` numbers. Group them into `head_dim / 2` consecutive pairs `(x_0, x_1), (x_2, x_3), ...`; each pair is a point in the plane. Rotate pair `i` by the angle `m · θ_i`:

```
θ_i = base^(-2i / head_dim)        i = 0 .. head_dim/2 - 1,  base = 10000
x_{2i}   <- x_{2i} cos(m θ_i) - x_{2i+1} sin(m θ_i)
x_{2i+1} <- x_{2i} sin(m θ_i) + x_{2i+1} cos(m θ_i)
```

Do the same to every key with its own position. Leave `v` alone. Then compute the scores as usual.

The ladder of frequencies is the one Floor 5's sinusoidal embedding uses: pair 0 turns one radian per position, pair `i` turns `θ_i` radians, the last pair about `1/base`. For `head_dim = 64` and `base = 10000` the slowest pair needs about 47,000 positions for one full turn. Fast pairs resolve neighbours; slow pairs tell position 10 from position 1000.

**Why a rotation gives *relative* position.** Write `R(a)` for the 2x2 rotation by angle `a`. Two facts about rotations in the plane: they compose by adding angles, `R(a) R(b) = R(a + b)`, and the transpose undoes them, `R(a)^T = R(-a)`. So for one pair, with `q` at position `m` and `k` at position `n`:

```
(R(m θ) q) · (R(n θ) k) = q^T R(m θ)^T R(n θ) k = q^T R((n - m) θ) k
```

The score depends on `q`, on `k`, and on the *difference* `n - m`. Not on `m`, not on `n`. Sum over the pairs (each with its own `θ_i`) and the whole dot product has the property; the softmax and the weighted sum of the unrotated `v` inherit it. Three consequences, all checked by the trial:

1. Shift every position by the same offset and nothing changes, to float precision. This is what makes decoding against a KV cache (Floor 12) correct: cached keys keep the absolute positions they were written with, the new query gets its own, and only the differences reach the softmax.
2. A rotation is orthogonal, so `|q|` and `|k|` are unchanged and the `sqrt(d_k)` argument still holds. Position 0 is the identity.
3. Position-free attention is permutation-equivariant; RoPE attention is not. Shuffle the input tokens and the outputs no longer merely shuffle.

**Two layouts.** Rotating consecutive pairs `(x_{2i}, x_{2i+1})` is the *interleaved* layout of the paper and of this room. Llama, and most Hugging Face code, pairs `x_i` with `x_{i + head_dim/2}` instead and writes the rotation as `x · cos + rotate_half(x) · sin`, with `rotate_half(x) = cat(-x[hd/2:], x[:hd/2])`. It is the same operation after a fixed permutation of the head dimension, and since the `q` and `k` projections are learned, the two are equivalent models. But weights trained under one layout produce garbage under the other; the Hugging Face Llama conversion script permutes `q_proj` and `k_proj` for exactly this reason. Know which one you are holding.

**Extrapolation, and its limit.** A learned table has no row for position `block_size`. RoPE has an angle for every integer, and since only differences matter, a model trained at `T = 4096` *runs* at 8192 without error. It degrades, though: at distances it never saw, the slow pairs reach relative angles the model never learned to read, and attention to far tokens turns to noise. The long-context recipes (position interpolation, NTK-aware scaling, YaRN; Llama 3 simply raised `base` to 500,000) all rescale `θ_i` so that longer distances map onto angles the model has seen, then fine-tune briefly. No code for that here. The point to keep: `base` and `θ_i` are knobs. A larger base turns every pair except the first more slowly, so far positions stay distinguishable for longer. One practical note: `m · θ_i` with `m` in the tens of thousands wants the angle computed in float32 or better, even when the activations are bf16.

### Fewer keys than queries: grouped-query attention

Room 6.3's layer has `H` heads and each carves its own `k` and `v`. When a model generates, it caches every layer's `k` and `v` for every past token so it never recomputes them (Floor 12 builds that cache). Producing one new token then means, in every layer, reading the *entire* cache back from memory to score a single query against it. Arithmetic per byte read is tiny. Decoding is bound by **memory bandwidth**, and the bytes are:

```
kv_cache_bytes = 2 · n_layer · T · n_kv_heads · head_dim · bytes_per        (2: a key and a value)
```

For a 7B-class model with 32 layers, 32 heads of 128, 4096 tokens, bf16: `2 · 32 · 4096 · 32 · 128 · 2 = 2 GiB`, per sequence. Serve 40 sequences at once and the cache is 80 GB, before a single weight is loaded.

**Grouped-query attention** (Ainslie et al. 2023) keeps `n_heads` query heads and gives them only `n_kv_heads` key/value heads; each group of `n_heads / n_kv_heads` query heads shares one `k` and one `v`:

```
q = q_proj(x)   Linear(d_model, d_model)                 (B, H, T, hd)      after the head split
k = k_proj(x)   Linear(d_model, n_kv_heads · hd)         (B, n_kv, T, hd)
v = v_proj(x)   Linear(d_model, n_kv_heads · hd)         (B, n_kv, T, hd)
k, v = k.repeat_interleave(groups, dim=1), ...           (B, H, T, hd)      head h reads kv head h // groups
```

then the ordinary attention from Room 6.1 and the output projection. `n_kv_heads = n_heads` is plain multi-head attention; `n_kv_heads = 1` is **multi-query attention** (Shazeer 2019), the extreme. The cache shrinks by `n_heads / n_kv_heads`, and so does the memory traffic per decoded token. Llama 2 70B uses 64 heads and 8 kv heads (8x); Llama 3 8B and Mistral 7B use 32 and 8 (4x). The 1 GiB per 8k-token sequence that a Llama-3-8B-shaped model caches would be 4 GiB with full MHA. The quality cost is small: eight kv heads land within noise of full MHA on the usual benchmarks, one kv head costs measurably more, and eight is the compromise the field settled on. Parameters drop too (`2d² + 2d + 2 · n_kv · hd · (d + 1)` instead of `4d² + 4d`), but that is a side effect; the cache is the reason.

`repeat_interleave`, not `repeat`: query heads `0..groups-1` share kv head 0, the next `groups` share kv head 1. That is the convention of `torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)` and of every released checkpoint; `repeat` gives `0, 1, 0, 1`, a model that loads without error and speaks nonsense. When RoPE is in play, rotate `k` *before* the repeat: the rotation touches only the last axis, so it costs `n_kv_heads` heads instead of `n_heads`. In production the repeat never happens at all; the kernel indexes the shared head directly.

### The tiled gaze (secret room)

Attention can be computed **without ever building the `(Tq, Tk)` matrix**. Walk the keys in blocks and keep, per query, a running max `m`, a running sum `l` of `exp(score - m)`, and a running accumulator `acc` of `exp(score - m) · v`. When a new block raises the max, rescale the old state by `exp(m_old - m_new)`; since `exp(s - m_old) · exp(m_old - m_new) = exp(s - m_new)`, the result is exact. The largest live tensor is `(B, Tq, block)`. This is the forward pass of FlashAttention, minus the hardware.

---

## Rooms

Run `dungeon enter 6` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code and every `None` in a prophecy with an answer, then run that room's trial.

### 6.1 The Single Gaze — `rooms/room_1_the_single_gaze.py`

The first head turns to face you. Two functions: `attention_scores(q, k)` (scaled) and `scaled_dot_product_attention(q, k, v, mask=None) -> (out, weights)`, with the mask applied as `-inf` before the softmax. The trial checks rows sum to 1, the scaling is exactly `1/sqrt(d)`, identical keys give the mean of `v`, masked keys weigh exactly zero, and the whole thing matches `torch.nn.functional.scaled_dot_product_attention` to `1e-6`.

Then **the Prophecy of the Peaks**: for `d = 4, 64, 1024`, with and without scaling, predict whether the largest softmax weight is `"spread"`, `"sharp"` or `"one_hot"`. Commit before the trial measures it.

```
dungeon trial 6 room_1
```

### 6.2 The Veil of Causality — `rooms/room_2_veil_of_causality.py`

A gauze veil between each head and the future. `causal_mask(T)`, `causal_attention(q, k, v)`, and the function this floor is really about: `leaks_future(attn_fn, B, T, d, seed) -> bool`, which perturbs the future and watches the past. The trial hands your detector an honest attention, an unveiled one, one with the veil applied after the softmax, one hung a seat too far, and one whose veil is honest but whose values were mixed with the next position's before attention. It must acquit the first and convict the other four without reading a line of their code.

```
dungeon trial 6 room_2
```

### 6.3 The Thousand Heads — `rooms/room_3_the_thousand_heads.py`

Look up. `MultiHeadAttention(nn.Module)` with a fused `qkv = nn.Linear(d_model, 3*d_model)`, `proj = nn.Linear(d_model, d_model)`, `split_heads`, `merge_heads` and a `forward(x, mask=None, return_weights=False)`. The trial checks shapes and the `4d² + 4d` parameter count, that one head equals Room 6.1 between two linear layers, and then copies your weights into `nn.MultiheadAttention` and demands agreement to `1e-5`, with and without a causal mask. (Watch the trial pass torch `~mask`. Torch's `True` means *ignore*.)

```
dungeon trial 6 room_3
```

### 6.4 The Padding Veil — `rooms/room_4_the_padding_veil.py`

Scrolls of unequal length on one rack. `key_padding_mask(lengths, T)`, `combine_masks(causal, padding) -> (B, 1, T, T)`, `safe_softmax(scores, mask)` that returns zeros instead of NaN for fully masked rows, and `masked_mha(mha, x, lengths)` assembled from your module's pieces. The trial checks that padded keys weigh exactly zero, that a length-0 sequence produces no NaN, and the one that matters: every real position of a padded batch equals the same sequence run alone and unpadded, to `1e-5`.

```
dungeon trial 6 room_4
```

### 6.5 The Rotary Gallery — `rooms/room_5_the_rotary_gallery.py`

Heads on brass rings, each turned a little further than the last. `rope_frequencies(head_dim, base)`, `rope_angles(positions, head_dim, base)` and `apply_rope(x, positions, base)` on interleaved pairs; then `RoPEAttention`, Room 6.3's module with the rotation on `q` and `k`; then `GroupedQueryAttention(d_model, n_heads, n_kv_heads)` with `expand_kv` and `kv_cache_bytes`. The trial checks that the rotation preserves norms, does nothing at position 0, agrees with complex multiplication (`view_as_complex`, times `exp(iθ)`), and that scores depend only on the distance between positions (shift everything by 7: nothing moves). The module must leak nothing, match a reference to `1e-5`, ignore where the sequence starts, and, unlike position-free attention, fail to be permutation-equivariant. GQA with `n_kv_heads == n_heads` must be Room 6.3, with one kv head multi-query attention, and in between agree with torch's `enable_gqa`. Then **the Prophecy of the Gallery**: which pair turns fastest, what a shift does to the scores, what a larger base does to far positions, and the cache saving when 32 heads share 8.

```
dungeon trial 6 room_5
```

---

## Boss: The Oracle Who Peeks

```
                    .-~~~~-.
                   /  _  _  \           "Ask me the next word. I have never
                  |  (o)(o)  |            been wrong. Not once. Curious, no?"
                  |    __    |
                   \  (__)  /            The Oracle sits behind four veils and
                  .-`------'-.           swears that each is causal. Its loss
                 /  |  ||  |  \          curve is magnificent. Its generations
                /   |  ||  |   \         are gibberish. Somewhere, a head is
               |____|__||__|____|        hearing the future.
```

**Weakness:** causal-mask correctness, verified empirically, position by position, never by reading the code.

The boss file contains five oracles as fixtures: `oracle_honest`, `oracle_mask_after_softmax`, `oracle_off_by_one`, `oracle_padding_on_keys` and `oracle_padding_on_queries`. They are complete and some are cursed. Leave them exactly as they are; the trial checks that you did.

- **Phase 1:** `leak_pairs(attn_fn, T, d, seed)` returns every `(t, j)` with `j > t` such that perturbing position `j` *alone* changes the output at `t`; `detect_leaks` reduces it to the compromised positions. The trial knows the exact answers: the honest oracle leaks nowhere, the veil-after-softmax oracle leaks at every future pair, the off-by-one oracle at exactly `(t, t+1)`. Perturb one position at a time or you cannot tell the last two apart.
- **Phase 2:** `detect_padding_leak(attn_fn, lengths)`: rewrite only the padding; do the real positions move? Convicts the oracle that veils the query axis.
- **Phase 3:** `FixedOracle`: Room 6.3's heads behind Room 6.4's veils, passing both detectors and matching torch's attention with the combined mask to `1e-5`.
- **Phase 4:** `ORACLE_PROPHECY`: before running anything, write `"leaks"` or `"honest"` next to each oracle.

```
dungeon fight 6
```

## Secret room: The Tiled Gaze *(optional)*

A slit behind the dais too narrow for the whole hall. `blockwise_logsumexp(x, block_size)` as a warm-up, then `online_softmax_attention(q, k, v, block_size)`: exact attention that walks the keys in blocks with the running max / running sum recurrence and never materialises the `(Tq, Tk)` matrix. The trial compares against plain attention for block sizes that do and do not divide `Tk`, then watches every tensor operation you perform and closes the slit if a `(Tq, Tk)` tensor appears or torch's own attention is called.

```
dungeon trial 6 --secret
```

## Loot

Clear the five rooms and unmask the Oracle to unlock:

- **Codex of Attention Shapes** — `loot/attention_shapes_cheat_sheet.md`. Every shape through multi-head attention, the mask conventions (including torch's inverted one), the `sqrt(d_k)` argument with measured numbers, the exact weight mapping to `nn.MultiheadAttention`, the leakage-test recipe, RoPE and GQA shapes with the KV-cache arithmetic, the online-softmax recurrence and the eleven bugs everyone hits.

## Stuck?

- `dungeon hint 6 room_2` reveals one hint at a time (three per room).
- The trial failure messages say *which shape or concept* went wrong and show the observed value. Read them before the hints.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, and to compare, not to copy.

When `dungeon map` shows the Hall cleared, take the stairs. Below, someone is stacking heads into a tower.
