# Codex of Attention Shapes

*Loot from Floor 6. Every tensor through multi-head attention, every mask convention, and the two tests that catch the bugs the loss curve hides.*

## The equations

```
scores  = q @ k^T / sqrt(d_k)             one number per (query, key)
weights = softmax(scores, dim=-1)          over the KEYS: each query's row sums to 1
out     = weights @ v                      a weighted average of the values
```

Masks are applied to `scores` *before* the softmax as `-inf`. `exp(-inf) == 0.0` exactly, so a masked key vanishes from both numerator and denominator.

## Shapes through multi-head attention

| Step | Tensor | Shape |
|---|---|---|
| input | `x` | `(B, T, d_model)` |
| fused projection | `qkv = Linear(d_model, 3*d_model)(x)` | `(B, T, 3*d_model)` |
| split | `q, k, v = qkv.split(d_model, -1)` | `(B, T, d_model)` each |
| heads | `q.view(B, T, H, hd).transpose(1, 2)` | `(B, H, T, hd)`, `hd = d_model / H` |
| scores | `q @ k.transpose(-2, -1) / sqrt(hd)` | `(B, H, T, T)` |
| mask | bool, broadcast to scores | `(T, T)`, `(B, 1, T, T)` or `(B, H, T, T)` |
| weights | `softmax(dim=-1)` | `(B, H, T, T)` |
| heads out | `weights @ v` | `(B, H, T, hd)` |
| merge | `.transpose(1, 2).contiguous().view(B, T, d_model)` | `(B, T, d_model)` |
| output | `proj = Linear(d_model, d_model)` | `(B, T, d_model)` |

Cross-attention: `q` comes from one sequence `(B, Tq, d)`, `k, v` from another `(B, Tk, d)`; scores are `(B, H, Tq, Tk)` and the output has `Tq` rows. `v` may have a different last dim `dv`; the output inherits it.

Parameters: `qkv` has `3d² + 3d`, `proj` has `d² + d`. Total **`4d² + 4d`**, identical to `nn.MultiheadAttention`.

Cost: time `O(B · H · T² · hd) = O(B · T² · d_model)`; memory for the weights `O(B · H · T²)`. The `T²` is why context length is expensive and why the Tiled Gaze exists.

## Why `sqrt(d_k)`

If the entries of `q` and `k` are independent with mean 0 and variance 1, then `q · k = Σ_i q_i k_i` has mean 0 and variance `d_k` (each product has variance 1, and the `d_k` terms are uncorrelated). Standard deviation `sqrt(d_k)`. At `d_k = 1024` the scores are spread over tens of units and the softmax saturates: one weight near 1, the rest near 0, and the gradient `p(1 - p)` of the softmax is nearly zero everywhere. Dividing by `sqrt(d_k)` makes the variance 1 for any `d_k`. Measured (16 keys, unit-variance entries): the mean largest weight goes 0.43, 0.84, 0.96 at d = 4, 64, 1024 unscaled, and stays near 0.25 scaled.

## Mask conventions (read twice)

| Where | Type | Meaning of `True` | Notes |
|---|---|---|---|
| this floor | bool | **may attend** | `scores.masked_fill(~mask, -inf)` |
| `F.scaled_dot_product_attention(attn_mask=)` | bool | **may attend** | same as this floor; float masks are *added* to the scores |
| `F.scaled_dot_product_attention(is_causal=True)` | flag | causal | shorthand for a lower-triangular mask |
| `nn.MultiheadAttention(attn_mask=)` | bool | **may NOT attend** | inverted; pass `~mask`. Shape `(T, T)` or `(B*H, T, T)` |
| `nn.MultiheadAttention(key_padding_mask=)` | bool | **is padding, ignore** | shape `(B, T)`; inverted relative to this floor's `key_padding_mask` |
| additive masks (many codebases) | float | `0` = attend, `-inf` (or `-1e9`) = ignore | `scores + mask`; equivalent |

Rule: before trusting any mask, look up which way its `True` points.

## Causal, padding, combined

```python
causal   = torch.tril(torch.ones(T, T, dtype=torch.bool))             # (T, T)     causal[i, j] = j <= i
padding  = torch.arange(T)[None, :] < lengths[:, None]                # (B, T)     True at real tokens
combined = causal[None, None, :, :] & padding[:, None, None, :]       # (B, 1, T, T)
```

- Causal hides the **future**: query `i` may not hear key `j > i`.
- Padding hides **blank keys** and differs per batch element, so it lives on the *last* (key) axis with a batch axis in front.
- The size-1 axis in `(B, 1, T, T)` is the head axis, left as 1 so one mask serves every head.
- Padding on the **query** axis is a bug: it silences rows nobody reads and leaves real queries hearing the padding.
- Right-padded causal batches never produce a fully masked row (query `i < L` always hears key 0) unless `L == 0`.

## Fully masked rows

A row of all `-inf` goes through `softmax` as `NaN` (`(-inf) - (-inf)`), and NaN poisons the whole batch's gradient. Guard it:

```python
weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
alive   = mask.any(dim=-1, keepdim=True)
weights = weights.masked_fill(~alive, 0.0)          # silence, not poison
```

`F.scaled_dot_product_attention` has changed its behaviour on such rows across torch versions (NaN in older releases, zeros in recent ones). Handle it yourself.

## Weight mapping to `nn.MultiheadAttention`

Your layout is torch's layout. With `twin = nn.MultiheadAttention(d_model, H, batch_first=True, bias=True)`:

```python
twin.in_proj_weight.copy_(mha.qkv.weight)      # (3d, d): rows [0:d] -> q, [d:2d] -> k, [2d:3d] -> v
twin.in_proj_bias.copy_(mha.qkv.bias)          # (3d,)
twin.out_proj.weight.copy_(mha.proj.weight)    # (d, d)
twin.out_proj.bias.copy_(mha.proj.bias)        # (d,)
out, avg_weights = twin(x, x, x, attn_mask=~causal, need_weights=True)   # note the ~
```

Heads are contiguous chunks of the model dim on both sides (`view(B, T, H, hd)`), so nothing needs permuting. `need_weights=True` returns weights **averaged over heads**, `(B, T, T)`; compare with your `(B, H, T, T).mean(1)`. The Chronicler on Floor 7 uses the same layout under the names `c_attn` and `c_proj`.

## The leakage test (never read the code, perturb it)

```
draw q, k, v from a seeded generator; base = attn(q, k, v)
for each future position j (one at a time):
    rewrite row j of q, k, v with fresh random values
    out = attn(q2, k2, v2)
    for each t < j: if out[:, t] moved (atol 1e-6) -> (t, j) leaks
```

- Honest causal attention passes **bitwise**: masked weights are exactly `0.0`.
- Mask after softmax: every `(t, j)` with `j > t` leaks (the denominator still holds the future).
- `tril(diagonal=1)`: exactly `(t, t+1)` leaks.
- Padding test: rewrite only the padded positions; real positions must not move.
- A leak is invisible in the training loss (the loss gets *better*). This test is the only way to see it.

## The online-softmax recurrence (tiled attention)

Per query, walk the keys in blocks and keep `m` (running max), `l` (running sum of `exp`), `acc` (running `exp`-weighted sum of `v`):

```
m = -inf, l = 0, acc = 0
for each block:
    s      = q @ k_blk^T / sqrt(d)                # (B, Tq, block)
    m_new  = max(m, s.max(-1))
    alpha  = exp(m - m_new)                       # rescale the old state
    p      = exp(s - m_new)
    l      = alpha * l   + p.sum(-1)
    acc    = alpha * acc + p @ v_blk
    m      = m_new
out = acc / l
```

Exact because `exp(s - m_old) * exp(m_old - m_new) == exp(s - m_new)`. The largest live tensor is `(B, Tq, block)`; the `(Tq, Tk)` matrix never exists. `logsumexp` is the same recurrence without `acc`: `m + log(l)`.

## The bugs everyone hits

1. Softmax over the wrong axis (`dim=-2`): columns sum to 1 instead of rows. Rows sum to 1, always.
2. Scaling by `sqrt(d_model)` inside multi-head attention. It is `sqrt(head_dim)`.
3. `x.view(B, H, T, hd)` instead of `x.view(B, T, H, hd).transpose(1, 2)`: no error, scrambled heads, torch disagrees.
4. Mask multiplied in after the softmax: rows do not sum to 1 and the future leaks through the denominator.
5. `tril(..., diagonal=1)` or `triu`: one seat too far, or the wrong half of the matrix.
6. Passing a True-means-attend mask to `nn.MultiheadAttention` without `~`.
7. Padding masked along the queries instead of the keys.
8. A fully masked row: NaN in one sequence, NaN gradients for the whole batch.
