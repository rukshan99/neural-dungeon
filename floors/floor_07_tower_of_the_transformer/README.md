# Floor 7 — The Tower of the Transformer

> *Identical floors, stacked with care. A stairwell up the middle that nothing may replace, only add to. At the top, someone is speaking. Listen closer: repeating.*

```
                        ☁  the Sovereign's balcony
                   ┌────────┴────────┐
                   │ ☠ THE STUTTERING│   "...was vast, vast, and usually
                   │     SOVEREIGN   │    right. The chronicle records..."
          ┌────────┴────────┬────────┴────────┐
          │  7.5 THE VOICE  │  ◇ the reading  │
          │    (decoding)   │  room: the      │
          │                 │  Accumulated    │
          ├─────────────────┤  Verse          │
          │  7.4 THE        ├─────────────────┤
          │    TRAINING     │                 │
          ├─────────────────┤   the residual  │
          │  7.3 THE TOWER  │   stairwell     │
          ├─────────────────┤                 │
          │  7.2 THE BLOCK  │   x = x + f(x)  │
          ├─────────────────┤        ▲        │
          │  7.1 THE NORM   │        │        │
          │    AND THE      │        │        │
          │   NONLINEARITY  │        │        │
          └────────┬────────┴─────────────────┘
                   ▲ stairs up from Floor 6
```

On Floor 6 you built attention: a way for every position in a sequence to read from the positions before it. Attention on its own is a mechanism, not a model. This floor is where it becomes one. You will build a GPT from the bricks up, load the dungeon's own pretrained Chronicler into it to prove the architecture is exactly right, train a small one on the dungeon's chronicles, and then make it talk.

Every large language model you have used is this tower, taller. The layout you build here — token and position embeddings, a stack of identical pre-norm blocks, a final norm, a tied output head, cross-entropy on the next token — is GPT-2's, and it is also, with modest changes (RMSNorm for LayerNorm, rotary embeddings for `wpe`, SwiGLU for the MLP, grouped-query attention), the layout of the models that followed. Knowing it at the level of shapes is the difference between reading a model card and being able to say why a change to one number in it costs what it costs.

The boss at the top does not test your weights. The Chronicler's weights are good: validation loss 0.21 nats per character. Ask it to speak by always taking the most likely character and it says the same sentence forever. The boss is about the last mile — how a distribution over the next token becomes text — and about how to judge that fairly.

**You will learn:** LayerNorm and GELU · the position-wise MLP · causal self-attention with a fused qkv projection · pre-norm residual blocks · the full GPT with weight tying and GPT-2 initialisation · next-token cross-entropy on shifted targets · AdamW, gradient clipping, warmup + cosine · greedy, temperature, top-k and top-p decoding · repetition penalties and n-gram blocking · gradient accumulation (secret room).

**You need:** PyTorch (CPU is plenty: `pip install torch --index-url https://download.pytorch.org/whl/cpu`), Floor 6's attention, and Floor 5's tokenizer ideas. Everything runs in a few seconds on a laptop.

---

## The lore of the tower (read this before the rooms)

### The problem: predict the next token

A language model is a function from a sequence of tokens to a probability distribution over the next token. The Chronicler works on characters: 72 of them, so at every position it outputs 72 numbers. Training data is just text. Take a window of `T` tokens `x[0..T-1]`; the targets are the same window shifted one step: `y[t] = x[t+1]`. One window gives `T` training examples at once, because position `t` predicts `y[t]` from `x[0..t]` — and *only* from `x[0..t]`, which is what the causal mask enforces.

The stream of activations that flows through the model has shape `(B, T, D)`: `B` sequences in the batch, `T` positions, and a `D`-dimensional vector per position (`D = n_embd`, 128 for the Chronicler). Every module on this floor maps `(B, T, D)` to `(B, T, D)` except the very first (tokens in) and the very last (logits out).

### Brick 1: LayerNorm

For each position's vector `x` of length `D`:

```
mean = x.mean(-1, keepdim=True)                  (B, T, 1)
var  = x.var(-1, keepdim=True, unbiased=False)   (B, T, 1)   biased: divide by D
y    = (x - mean) / sqrt(var + eps) * weight + bias
```

`weight` and `bias` are `(D,)` learnable vectors, initialised to ones and zeros. Every position is normalised independently — not across the batch (that is BatchNorm, Floor 3's crucible), not across time. LayerNorm's job is to keep the scale of the vectors entering each sub-layer predictable no matter how deep the tower is or how large the residual stream has grown. `eps` (1e-5) lives inside the square root so a constant vector does not divide by zero.

### Brick 2: GELU

The nonlinearity between the MLP's two layers. ReLU is `max(0, x)`; GELU is `x · Φ(x)` where `Φ` is the standard normal CDF:

```
gelu(x) = 0.5 · x · (1 + erf(x / √2))
```

It is smooth, it is close to ReLU for large `|x|`, and it lets a small negative signal through (`gelu(-1) ≈ -0.159`). GPT-2 used a tanh approximation, `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 x³)))`, which is within 5e-4 of the exact form. Both appear in checkpoints in the wild; you will write both.

### Brick 3: the MLP

```
fc:    (B, T, D)  -> (B, T, 4D)
gelu:  elementwise
proj:  (B, T, 4D) -> (B, T, D)
```

It is applied to every position separately and identically ("position-wise"); no information moves between positions here. That is attention's job. The factor 4 is convention. Parameters: `fc` has `4D·D + 4D`, `proj` has `D·4D + D`, total `8D² + 5D` — two thirds of each block.

### Causal self-attention, shape by shape

Floor 6 taught the mechanism. Here is the exact form the tower uses, for `H` heads of size `hd = D / H`:

```
qkv = c_attn(x)                                (B, T, 3D)   one Linear, D -> 3D
q, k, v = qkv.split(D, dim=2)                  3 × (B, T, D)   in this order
q = q.view(B, T, H, hd).transpose(1, 2)        (B, H, T, hd)   same for k, v
att = q @ k.transpose(-2, -1) / sqrt(hd)       (B, H, T, T)    row t: query t against every key
att = att.masked_fill(~mask[:T, :T], -inf)     future keys become -inf ...
att = softmax(att, dim=-1)                     ... and therefore weight exactly 0
y = att @ v                                    (B, H, T, hd)   weighted sum of values
y = y.transpose(1, 2).contiguous().view(B, T, D)   heads side by side again
out = c_proj(y)                                (B, T, D)
```

Three things the trials check that are easy to get subtly wrong:

- **The fused projection.** One `Linear(D, 3D)` named `c_attn`, split into q, k, v in that order along the last dim. Three separate projections compute the same thing but have different parameter names, and the checkpoint will not load.
- **The scale is `1/√hd`**, the head size, not `1/√D`. Dot products of `hd` roughly unit-variance numbers have variance `hd`; dividing by `√hd` keeps the softmax inputs O(1) so it is not saturated at initialisation.
- **The mask is a buffer, not a parameter, and not saved.** `torch.tril(torch.ones(block_size, block_size, dtype=bool))` shaped `(1, 1, block_size, block_size)`, registered with `persistent=False`. Slice it to `[:T, :T]` for shorter inputs. It is derived from `block_size`; the checkpoint does not contain it, and a `strict=True` load will complain if yours does.

Parameters: `c_attn` has `3D² + 3D`, `c_proj` has `D² + D`: `4D² + 4D`.

### The block, and why pre-norm

```
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
```

Read it carefully. `x` is the **residual stream** — the stairwell. Each sub-layer takes a *normalised copy* of the stream, computes something, and **adds** it back. The stream itself is never normalised in place and never replaced. This is why a 96-block tower can be trained: the gradient of the loss with respect to the input of block 1 has a direct path — the sum of identities — all the way from the top, no matter what the sub-layers do. Zero out every `c_proj` and `mlp.proj` and the block is exactly the identity function; the trial checks this.

The 2017 Transformer put the norm *after* the add (`x = ln(x + attn(x))`, "post-norm"). GPT-2 moved it before ("pre-norm"), which trains stably at depth without careful warmup and is what nearly every decoder-only model uses. The trial distinguishes the two.

Per block: attention `4D² + 4D`, MLP `8D² + 5D`, two LayerNorms `4D`: **`12D² + 13D`**. For `D = 128` that is 198,272.

### The tower

```
GPT
├── wte      Embedding(vocab_size, D)     token id -> vector
├── wpe      Embedding(block_size, D)     position  -> vector
├── drop     Dropout
├── blocks   ModuleList of n_layer Blocks
├── ln_f     LayerNorm(D)
└── lm_head  Linear(D, vocab_size, bias=False)   weight IS wte.weight
```

Forward, with `idx` of shape `(B, T)` holding int64 token ids:

```
if T > block_size: raise ValueError           wpe has only block_size rows
pos = arange(T)                               (T,)
x = drop(wte(idx) + wpe(pos))                 (B, T, D) + (T, D) broadcasts over B
for block in blocks: x = block(x)             (B, T, D), n_layer times
x = ln_f(x)
logits = lm_head(x)                           (B, T, V)
```

**Why add position embeddings?** Attention is a set operation: permute the keys and values and the weighted sum does not change. Without `wpe`, six copies of the same token would produce six identical output vectors. Adding a learned vector per position gives every token a "where" as well as a "what". (Rotary embeddings, used by most newer models, put the "where" into q and k instead; the idea is the same.)

**Why tie the head to the embedding?** `wte` maps a token to a vector; `lm_head` scores a vector against every token. Both are `(V, D)` matrices that relate tokens to the same vector space. Using one tensor for both (`self.lm_head.weight = self.wte.weight`, an assignment of the `Parameter` object, not a copy) saves `V·D` parameters — for GPT-2 that was 38 million of 124 — and acts as a regulariser: the output vector for "the" must live where the input vector for "the" lives. `model.parameters()` yields the tied tensor once, which matters when you count.

**Initialisation** (GPT-2's recipe, which the trial checks statistically): every `Linear` and `Embedding` weight is `N(0, 0.02)`, every `Linear` bias is 0, and then the weights that *write into the residual stream* — every `attn.c_proj.weight` and `mlp.proj.weight` — are re-drawn with std `0.02 / √(2·n_layer)`. `2·n_layer` sub-layers each add into the stream; if each added variance `σ²`, the stream would leave the tower with variance `2·n_layer·σ²`. Shrinking the writers by `√(2·n_layer)` cancels that. A newborn tower therefore outputs near-zero logits and a loss of about `ln V` — 4.28 for 72 characters — which is exactly "I have no idea", which is the right place to start from.

**The Chronicler:** `vocab_size=72, block_size=128, n_layer=4, n_head=4, n_embd=128`, **802,560** non-embedding parameters (818,944 with `wpe`). Room 3's final trial loads its `state_dict` into your GPT with `strict=True` and demands the same loss to five decimal places. That is the exactness test of the whole floor: every name, every shape, every operation in the same order.

### The loss

`logits` is `(B, T, V)`; `targets` is `(B, T)`. Flatten both and hand them to cross-entropy:

```
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
```

That is `B·T` independent `V`-way classifications, averaged. For one position with target `y`, the loss is `-log softmax(logits)[y]`: the negative log-probability the model assigned to what actually came next. Its units are nats per token. `ln V` is "uniform guessing"; 0 is "certain and right". The Chronicler's 0.21 means it assigns the true next character about `e^-0.21 ≈ 81%` probability on average — the chronicles are formulaic, and it has read them a great many times. That will matter at the top of the tower.

### Training

**Batches.** Pick `batch_size` random start indices `i` into the token stream with `i + block_size < len(data)`, take `x = data[i : i+block_size]` and `y = data[i+1 : i+1+block_size]`. Draw `i` with a seeded `torch.Generator` so a run is reproducible.

**AdamW.** Adam with weight decay applied directly to the weights rather than folded into the gradient. `betas=(0.9, 0.95)` and `weight_decay=0.1` are the GPT defaults. (A serious setup excludes biases and norm weights from decay; the trial does not insist.)

**Gradient clipping.** After `backward()`, `clip_grad_norm_(params, 1.0)` rescales the whole gradient so its global L2 norm is at most 1.0. One unlucky batch — a rare character, an exploding attention score — produces a gradient a hundred times the usual size; without clipping it moves the weights a hundred times too far and the loss spikes. The trial trains with an absurd clip of 0.01 and checks the norm at every step really is 0.01.

**Warmup + cosine.** Adam's running estimates of gradient variance start at zero, so its first steps are badly scaled; a linear warmup from a small learning rate protects them. After warmup, a cosine from `lr_max` down to `lr_min` lets the model settle instead of bouncing. The exact contract:

```
step < warmup:   lr_max · (step + 1) / warmup          step warmup-1 reaches lr_max
step >= total:   lr_min
otherwise:       progress = (step - warmup) / (total - warmup)
                 lr_min + ½ (lr_max - lr_min) (1 + cos(π · progress))
```

The trial checks the boundary values exactly — step 0, the last warmup step, the first decay step, the midpoint, the end.

**The loop**, every step: set the learning rate on every `param_group`, get a batch, forward, `zero_grad`, `backward`, clip, `step`. Room 4 trains a 2-layer, 64-wide tower for 120 steps on the chronicles: from 4.3 to about 2.4 in a few seconds on a CPU.

### Decoding: from logits to text

At each step the model gives logits `z` `(V,)` for the next token; `p = softmax(z)`. Then:

| | |
|---|---|
| **Greedy** | `argmax(z)`. Deterministic. The single most likely token every time. |
| **Temperature** `T` | sample from `softmax(z / T)`. `T < 1` sharpens toward greedy; `T > 1` flattens toward uniform; `T → 0` *is* greedy, which is why `temperature <= 0` means greedy in this room. |
| **Top-k** | keep the `k` largest logits, set the rest to `-inf`, sample. `-inf` becomes probability exactly 0 after softmax. `k = 1` is greedy. |
| **Top-p (nucleus)** | sort by probability, keep the smallest prefix whose cumulative mass reaches `p`, set the rest to `-inf`, sample. Precisely: a token is dropped iff the mass *before* it in sorted order is already `>= p`. The best token always survives (nothing is before it). Adapts to the distribution: keeps one token when the model is sure, many when it is not. |

Filters apply to the temperature-scaled logits, then one `torch.multinomial` draw per row with a `Generator` for reproducibility.

**The loop.** `generate` appends tokens one at a time. Before each forward pass, **crop the context to the last `block_size` tokens** — `idx[:, -block_size:]` — because `wpe` has no row for position 128. Use the logits of the **last position only**; the others are predictions for tokens you already have. Every step re-runs the whole prefix (O(T²) total); Floor 12 caches keys and values to fix that.

### Why the Sovereign stutters

The Chronicler assigns 81% to the right character on average. Under greedy decoding a confident model is a deterministic function of its last 128 characters; once it emits a sentence it has seen many times, the most likely continuation is the start of that sentence again, and the context that follows is identical to the context that came before. It is a fixed point: *"The chronicle records that Bellamy was vast, vast, and usually right."* — forever. The measured distinct-4-gram ratio of 300 greedy characters is 0.23.

Fixes that change no weights: **sample** (temperature 0.7–1.0, top-k or top-p) so the loop can be broken by chance; **penalise repeats** (CTRL-style: for every token already generated, divide a positive logit by `r`, multiply a negative one by `r`); **block repeated n-grams** (if the last `n-1` tokens have occurred before, set the tokens that followed them to `-inf`, so no n-gram can ever appear twice — the distinct-n-gram ratio becomes exactly 1.0 by construction).

But diversity is cheap: uniform noise has a distinct-4-gram ratio of 1.0. So the court also measures **fluency**: the mean NLL of the produced text under the model itself, teacher-forced — feed the text in, score every character given the ones before it. Noise scores about 11; greedy scores 0.16; temperature 2.0 scores 1.2–2.0. The boss requires **distinct-4-gram ≥ 0.6 and mean NLL ≤ 1.0**, on three seeds. Greedy fails the first; noise fails the second; a sensible strategy passes both with room to spare.

---

## Rooms

Run `dungeon enter 7` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial. Rooms 2 and 3 import your own code from the rooms below them, so the tower is built from your bricks.

### 7.1 The Norm and the Nonlinearity — `rooms/room_1_norm_and_nonlinearity.py`

The mason's yard at the foot of the tower. `LayerNorm(ndim, eps, bias)` from scratch — its outputs *and gradients* must match `nn.LayerNorm` to 1e-6, so use ordinary tensor ops and let autograd through. `gelu` via `torch.erf`, `gelu_tanh` via the approximation. `MLP(n_embd)` with layers named `fc` and `proj`; the trial loads the reference MLP's weights into yours by name and compares outputs, and counts `8D² + 5D` parameters.

```
dungeon trial 7 room_1
```

### 7.2 The Block — `rooms/room_2_the_block.py`

One complete floor. `CausalSelfAttention(n_embd, n_head, block_size)` with fused `c_attn`, `c_proj`, a non-persistent mask buffer and `1/√hd` scaling. The trial rewrites the future and checks the past did not notice, then loads the reference attention's weights into yours and demands agreement to 1e-5. `Block` wires `ln_1, attn, ln_2, mlp` as pre-norm; the trial silences the sub-layers and checks the block becomes the identity, distinguishes pre- from post-norm, counts `12D² + 13D`, and finally loads a whole reference block into yours with `strict=True`.

```
dungeon trial 7 room_2
```

### 7.3 The Tower — `rooms/room_3_the_tower.py`

`GPT(cfg)`: embeddings, `drop`, `blocks`, `ln_f`, tied `lm_head`, GPT-2 init, `num_params`, `forward(idx, targets=None) -> (logits, loss)`, `ValueError` past `block_size`. Then the key trial of the floor: the shipped Chronicler checkpoint is loaded into your tower with `strict=True` and must produce the reference's loss on a fixed batch of chronicles to 1e-5. 802,560 non-embedding parameters, `model.lm_head.weight is model.wte.weight`.

```
dungeon trial 7 room_3
```

### 7.4 The Training — `rooms/room_4_the_training.py`

`encode_corpus`, `get_batch` (y is x shifted by one, never past the end), `lr_schedule` (exact at the boundaries), `train` (AdamW via `torch.optim.AdamW`, clip, schedule, one float per step) and `estimate_loss`. The trial swaps AdamW for a spy that reads the learning rate and the gradient norm at every step, then trains your GPT from room 3 — 2 layers, 64 wide — for 120 steps on the chronicles and expects the last ten losses to average below 2.8 (the reference reaches about 2.4).

```
dungeon trial 7 room_4
```

### 7.5 The Voice — `rooms/room_5_the_voice.py`

`sample_next(logits, temperature, top_k, top_p, generator)` on hand-made logits where the right answer is known: temperature 0 is the argmax, `top_k=1` is greedy, top-p keeps exactly the nucleus of a known distribution, the generator makes it reproducible, `-inf` is never drawn. `generate(model, idx, max_new_tokens, **sampling)` must crop to `block_size`, use the last position's logits and pass the options through. Finally, greedy generation from `"\n"` with the pretrained Chronicler must reproduce its fixed line token for token.

```
dungeon trial 7 room_5
```

---

## Boss: The Stuttering Sovereign

```
                    _.-=-._
                  .'  ___  '.          "The chronicle records that Bellamy
                 /  .'   '.  \          was vast, vast, and usually right.
                |  |  o o  |  |         The chronicle records that Bellamy
                |  |   ^   |  |         was vast, vast, and usually right.
                 \  '.___.'  /          The chronicle records that Bellamy
                  '._______.'           was vast, vast, and usually r-"
                 .-'|     |'-.
                /   |_____|   \
               |   / _____ \   |
               |  | |     | |  |
               |__|_|     |_|__|
```

**Weakness:** the decoding strategy — judged by *both* diversity and fluency, so neither a loop nor noise can win.

- **Phase 1:** `distinct_ngram_ratio(ids, n)`, `repetition_penalty(logits, generated_ids, penalty)`, `block_repeated_ngrams(logits, generated_ids, n)`. Checked on sequences you can count by hand, then on the Sovereign's own greedy output.
- **Phase 2:** `mean_nll(model, ids)`: teacher-forced NLL per token, walking long texts in non-overlapping windows of `block_size`.
- **Phase 3:** `speak_without_stuttering(model, tokenizer, prompt, n_tokens=300, seed)`: prompt plus 300 characters with distinct-4-gram ≥ 0.6 and mean NLL ≤ 1.0, on seeds 0, 1 and 2. Combine your room 5 sampler with a penalty or a block and a temperature.
- **Phase 4:** `SOVEREIGN_PROPHECY`: for greedy, temperature 0.7 + top-k 40, temperature 2.0, and greedy + no-repeat-4-gram, predict *stutters*, *babbles* or *speaks* before the court measures 200 characters of each.

```
dungeon fight 7
```

## Secret room: The Accumulated Verse *(optional)*

Behind the throne, a reading room whose lectern holds one page at a time. `train_step_accumulated(model, batches, optimizer, accum_steps)`: the gradient after `k` micro-batches, each with loss scaled by `1/k`, must equal the gradient of one `k`-times-larger batch to 1e-5, and the optimizer must step exactly once per window. Plus `count_tokens_seen`. This is how every large model is trained on hardware that cannot hold its batch.

```
dungeon trial 7 --secret
```

## Loot

Clear the five rooms and defeat the Sovereign to unlock:

- **The GPT Skeleton** — `loot/gpt_skeleton.py`. The whole tower, a training loop and a sampler in one heavily annotated file that runs on any text file. Start your next project from it.
- **Decoding Cheat Sheet** — `loot/decoding_cheat_sheet.md`. Greedy, temperature, top-k, top-p, repetition penalty, no-repeat n-grams, beam search: what each does, when to use which, the diversity-vs-fluency trade-off, and the prompt-cropping rule.

## Stuck?

- `dungeon hint 7 room_2` reveals one hint at a time (three per room).
- Failure messages name the shape, the formula or the wiring that is wrong, with the observed value. The "wears the reference weights" trials are the most informative: same weights, different answer means one specific operation differs.
- `solutions/` exists. After an honest attempt, compare; do not copy.

When `dungeon map` shows the Tower cleared, take the stairs down. Someone below wants to teach the Chronicler a goblin's accent without retraining it.
