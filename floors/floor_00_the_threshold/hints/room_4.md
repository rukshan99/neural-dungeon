`one_hot`: make a zero matrix, then `out[np.arange(n), labels] = 1.0`. Two index arrays of equal length pick out one element per row.
---
`moving_average`: `c = np.cumsum(np.concatenate([[0.0], x]))`, then `(c[k:] - c[:-k]) / k`. `softmax_rows`: subtract `x.max(axis=1, keepdims=True)`, exponentiate, divide by the row sum with `keepdims=True`.
---
`count_neighbors_within`: `sq = (points ** 2).sum(1)`; `d2 = sq[:, None] + sq[None, :] - 2 * points @ points.T`; clamp with `np.maximum(d2, 0)`; `(d2 <= radius ** 2).sum(1) - 1` (the minus one removes the point itself, whose distance is 0).
