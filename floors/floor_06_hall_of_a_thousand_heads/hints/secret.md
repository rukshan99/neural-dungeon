The whole trick is one identity: `exp(s - m_old) * exp(m_old - m_new) == exp(s - m_new)`. Everything you accumulated under the old max can be moved under the new max with a single multiply by `alpha = exp(m_old - m_new)`. Keep three running tensors with `keepdim=True` so they broadcast: `m` (B, Tq, 1) starting at `-inf`, `l` (B, Tq, 1) starting at 0, `acc` (B, Tq, dv) starting at 0. `blockwise_logsumexp` is the same recurrence with only `m` and `l`; its answer is `m + log(l)`.
---
Loop `for start in range(0, Tk, block_size)` and slice `k[:, start:start + block_size]`, `v[:, start:start + block_size]`; a short final block needs no special case. Per block: `s = q @ k_blk.transpose(-2, -1) / math.sqrt(d)` is (B, Tq, blk), the only score tile that ever exists. `m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))`, `alpha = torch.exp(m - m_new)`, `p = torch.exp(s - m_new)`. On the first block `m` is `-inf`, so `alpha` is 0 and the zero initial state is harmlessly wiped.
---
```python
for start in range(0, Tk, block_size):
    k_blk, v_blk = k[:, start:start + block_size], v[:, start:start + block_size]
    s = q @ k_blk.transpose(-2, -1) / math.sqrt(d)
    m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
    alpha = torch.exp(m - m_new)
    p = torch.exp(s - m_new)
    l = alpha * l + p.sum(dim=-1, keepdim=True)
    acc = alpha * acc + p @ v_blk
    m = m_new
return acc / l
```
The trial's watcher flags any tensor whose last two dims are (Tq, Tk); `(B, Tq, blk)` tiles and the `(B, d, blk)` transposed key blocks are fine.
