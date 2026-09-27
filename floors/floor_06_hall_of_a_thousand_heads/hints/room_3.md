`__init__`: `super().__init__()` first, raise `ValueError` when `d_model % n_heads`, store `d_model`, `n_heads`, `head_dim = d_model // n_heads`, then `self.qkv = nn.Linear(d_model, 3 * d_model)` and `self.proj = nn.Linear(d_model, d_model)`. `split_heads` is `x.view(B, T, H, head_dim).transpose(1, 2)`; `merge_heads` is the reverse: `x.transpose(1, 2).contiguous().view(B, T, H * head_dim)`. A `view` straight from (B, T, d) to (B, H, T, hd) without the transpose gives the wrong layout and no error.
---
`forward`: `q, k, v = self.qkv(x).split(self.d_model, dim=-1)` (that order), split heads on each to get (B, H, T, hd), then call Room 6.1's `scaled_dot_product_attention(q, k, v, mask=mask)`; its `sqrt(q.shape[-1])` is now `sqrt(head_dim)`, which is exactly right. `merge_heads` the (B, H, T, hd) result and pass it through `self.proj`. Return `(out, weights)` only when `return_weights` is True.
---
If the shapes are right but the `nn.MultiheadAttention` comparison fails: (1) heads must be contiguous chunks of the model dimension (view-then-transpose, never a reshape to (B, H, T, hd)); (2) the scale is `sqrt(head_dim)`, not `sqrt(d_model)`; (3) merge with `transpose(1, 2)` *before* the reshape. The trial hands torch `~mask` on purpose: torch's boolean `attn_mask` means True = *may not* attend. Full forward:
```python
q, k, v = self.qkv(x).split(self.d_model, dim=-1)
q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
heads, weights = scaled_dot_product_attention(q, k, v, mask=mask)
out = self.proj(self.merge_heads(heads))
return (out, weights) if return_weights else out
```
