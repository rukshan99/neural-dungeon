RoPE is one 2-D rotation per pair of coordinates, with an angle that grows linearly with the position. `rope_frequencies`: `torch.arange(0, head_dim, 2, dtype=torch.float32)` is `2i`; the ladder is `base ** (-2i / head_dim)`, so `theta_0 == 1` and the rest shrink. `rope_angles` is the outer product `positions.float()[:, None] * theta[None, :]`. Why it gives *relative* position: `R(a)^T R(b) = R(b - a)`, so a rotated `q` at `m` dotted with a rotated `k` at `n` sees only `n - m`. Rotate `q` and `k`, never `v`. GQA is smaller than it sounds: `k_proj` and `v_proj` produce `n_kv_heads` heads instead of `n_heads`, and each kv head is repeated across its group of query heads before the ordinary attention. For the Prophecy: which `theta_i` is largest? What happens to `R(m t)^T R(n t)` when both `m` and `n` grow by 7? Does a larger base make `base^(-2i/hd)` bigger or smaller? How many query heads share one kv head when 32 share 8?
---
`apply_rope` shapes: `angles = rope_angles(positions, x.shape[-1], base)` is `(T, hd/2)`; `cos, sin = angles.cos(), angles.sin()` broadcast against `x[..., 0::2]` and `x[..., 1::2]`, both `(B, H, T, hd/2)`. Rotate `even' = even*cos - odd*sin`, `odd' = even*sin + odd*cos`, then `torch.stack((even', odd'), dim=-1).flatten(-2)` puts each pair back side by side. Check yourself: `apply_rope(x, zeros)` is `x`, and the norm along the last axis never changes. `RoPEAttention.forward`: split heads first, then rotate `q` and `k` (the pairs live inside each head's `hd`), then `scaled_dot_product_attention(q, k, v, mask=causal_mask(T) if self.causal else None)`. `GroupedQueryAttention.expand_kv` is `kv.repeat_interleave(self.groups, dim=1)`: head `h` reads kv head `h // groups`, which is what torch's `enable_gqa` does too. `kv_cache_bytes` is `2 * n_layer * T * n_kv_heads * head_dim * bytes_per`; the `2` is keys and values.
---
```python
def rope_frequencies(head_dim, base=10000.0):
    two_i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return base ** (-two_i / head_dim)

def rope_angles(positions, head_dim, base=10000.0):
    return torch.as_tensor(positions).float()[:, None] * rope_frequencies(head_dim, base)[None, :]

def apply_rope(x, positions, base=10000.0):
    angles = rope_angles(positions, x.shape[-1], base)
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
```
`RoPEAttention.forward`: `q, k, v = self.qkv(x).split(self.d_model, -1)`; split heads; `q, k = apply_rope(q, positions, self.base), apply_rope(k, positions, self.base)`; `heads, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(T) if self.causal else None)`; `return self.proj(self.merge_heads(heads))`. Default `positions = torch.arange(T)`.
`GroupedQueryAttention.forward`: `q = self.split_heads(self.q_proj(x), self.n_heads)`; `k = self.expand_kv(self.split_heads(self.k_proj(x), self.n_kv_heads))`; same for `v`; attention with the mask; merge; `self.proj`. Prophecy: `"first"`, `"unchanged"`, `"less"`, `4`.
