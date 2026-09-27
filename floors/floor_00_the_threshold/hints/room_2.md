Reshape when the *data order* is already right and you only want to regroup it: `x.reshape(x.shape[0], -1)`. Transpose when axes need to change places: `np.transpose(x, (0, 3, 1, 2))` sends old axis 3 to position 1.
---
`np.stack([a, b], axis=1)` creates a brand-new axis of length 2 at position 1. `x[:, -1]` drops a dimension; `x[:, -1:]` keeps it. `x[::2]` is every other row. `x[None, ...]` adds a leading axis of size 1.
---
For the prophecy: a reduction with `axis=k` removes axis k (unless `keepdims=True`, which leaves a 1 there). `@` on (4,3) and (3,2) gives (4,2). Broadcasting (3,1,4) with (5,4) aligns from the right: (3,1,4) vs (1,5,4) -> (3,5,4). `transpose(2, 0, 1)` puts old axis 2 first.
