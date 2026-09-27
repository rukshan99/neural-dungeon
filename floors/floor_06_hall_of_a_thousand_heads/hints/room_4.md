`key_padding_mask`: `torch.arange(T)[None, :] < lengths[:, None]` broadcasts (1, T) against (B, 1) into (B, T), True at real positions. `combine_masks`: `causal[None, None, :, :] & padding[:, None, None, :]`. Count the axes: batch, head (size 1), query, key. The padding sits on the *last* axis because padded tokens are hidden as keys, not as queries.
---
`safe_softmax`: fill with `-inf`, softmax over `dim=-1`, then `alive = mask.any(dim=-1, keepdim=True)` marks rows that hear at least one key. `weights.masked_fill(~alive, 0.0)` turns the NaN rows into zeros (`masked_fill` overwrites NaN like any other value). `torch.where(alive, weights, 0.0)` works too.
---
`masked_mha` assembles the pieces by hand so you can use your own softmax:
```python
B, T, _ = x.shape
q, k, v = mha.qkv(x).split(mha.d_model, dim=-1)
q, k, v = mha.split_heads(q), mha.split_heads(k), mha.split_heads(v)
scores = q @ k.transpose(-2, -1) / math.sqrt(mha.head_dim)          # (B, H, T, T)
mask = combine_masks(causal_mask(T), key_padding_mask(lengths, T))  # (B, 1, T, T)
weights = safe_softmax(scores, mask)
return mha.proj(mha.merge_heads(weights @ v))
```
