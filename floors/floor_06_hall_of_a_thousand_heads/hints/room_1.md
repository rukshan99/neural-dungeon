Attention is two matrix products with a softmax between them. `attention_scores` is `q @ k.transpose(-2, -1)` divided by `math.sqrt(q.shape[-1])`. The softmax runs over the last axis (the keys): one row of weights per query, summing to 1. The mask goes *before* the softmax: `scores.masked_fill(~mask, float("-inf"))`, so `exp(-inf) == 0` removes the masked keys from both numerator and denominator.
---
Shapes: `(B, Tq, d) @ (B, d, Tk) -> (B, Tq, Tk)` for the scores, then `(B, Tq, Tk) @ (B, Tk, dv) -> (B, Tq, dv)` for the output. Use `transpose(-2, -1)` and `softmax(dim=-1)` so extra leading axes (heads, later) pass through. For the Prophecy: with unit-variance entries, `q . k` has variance `d`, so its standard deviation is `sqrt(d)`: 2 at d=4, 8 at d=64, 32 at d=1024. Sixteen keys with scores spread over 32 units is a one-hot. Scaled scores always have standard deviation 1, whatever `d`.
---
```python
def attention_scores(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = attention_scores(q, k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```
Prophecy: unscaled goes `"spread"`, `"sharp"`, `"one_hot"` as d grows; scaled is `"spread"` three times.
