The veil is `torch.tril(torch.ones(T, T, dtype=torch.bool))`: True on and below the diagonal, so query i may hear keys j <= i. `causal_attention` is Room 6.1's function with `mask=causal_mask(T)`; the (T, T) mask broadcasts over the batch. For `leaks_future`, do not think about masks at all: think "if I change the future, does the past notice?"
---
`leaks_future` recipe: `g = torch.Generator().manual_seed(seed)`, draw `q, k, v = torch.randn(B, T, d, generator=g)` three times, `base = attn_fn(q, k, v)`. For each `t in range(T - 1)`: clone all three, overwrite `[:, t + 1:]` of each clone with fresh `randn(B, T - t - 1, d, generator=g)`, run `attn_fn` again and compare `out[:, :t + 1]` to `base[:, :t + 1]` with `torch.allclose(..., atol=1e-6, rtol=0)`. Any mismatch: return True. Compare only the past (`:t + 1`); the perturbed positions themselves are allowed to change.
---
```python
def leaks_future(attn_fn, B, T, d, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, T, d, generator=g) for _ in range(3))
    base = attn_fn(q, k, v)
    for t in range(T - 1):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for x in (q2, k2, v2):
            x[:, t + 1:] = torch.randn(B, T - t - 1, d, generator=g)
        out = attn_fn(q2, k2, v2)
        if not torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-6, rtol=0):
            return True
    return False
```
