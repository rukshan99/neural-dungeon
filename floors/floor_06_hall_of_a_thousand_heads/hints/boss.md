Phase 1 is Room 6.2's detector with a finer brush: instead of rewriting *all* of the future at once, rewrite *one* future row `j` (in q, k and v) and check every `t < j`. Record `(t, j)` when `out[:, t]` moved. A batch of 1 is plenty. `detect_leaks` is then `sorted({t for t, _ in leak_pairs(...)})`. The two cursed causal oracles compromise the same positions but through different pairs; only the pair report tells them apart.
---
Phase 2: `T = int(lengths.max())`, `real = torch.arange(T)[None, :] < lengths[:, None]`, `padded = ~real`. Boolean-index assignment rewrites all padding at once: `x2[padded] = torch.randn(int(padded.sum()), d, generator=g)`. Compare `out[real]` with `base[real]`. Phase 4 by reasoning: a veil after the softmax leaves the future in the denominator (leaks); `tril(diagonal=1)` hears one seat ahead (leaks); padding on the *query* axis silences rows nobody reads and lets real queries hear the blanks (leaks); the other two are honest.
---
Phase 3 is the assembly you already own. `FixedOracle` inherits `qkv`, `proj`, `split_heads`, `merge_heads` from Room 6.3, so:
```python
def forward(self, x, lengths=None):
    B, T, _ = x.shape
    if lengths is None:
        lengths = torch.full((B,), T, dtype=torch.long)
    return masked_mha(self, x, torch.as_tensor(lengths))
```
Phase 1 core: `for j in range(1, T)`: clone, `clone[:, j] = torch.randn(1, d, generator=g)` for each of q, k, v, run, then `for t in range(j)`: append `(t, j)` if `not torch.allclose(out[:, t], base[:, t], atol=1e-6, rtol=0)`.
