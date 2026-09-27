Cosine similarity is the dot product of unit vectors. Normalise the rows of both matrices once (`np.linalg.norm(x, axis=1, keepdims=True)`, divided with `np.maximum(norm, eps)` so a zero row stays zero) and the whole (N, M) matrix is `a_n @ b_n.T`. For `top_k`, think in two steps: *select* k candidates per row cheaply, then *sort* only those k.
---
`np.argpartition(-scores, k - 1, axis=1)[:, :k]` gives (N, k) candidate indices, unordered. `np.take_along_axis(scores, candidates, axis=1)` gathers their values. To order each row by descending value and then ascending index, use `np.lexsort((candidates, -values), axis=1)`: lexsort treats the LAST key as primary. When `k >= M`, skip the partition and take every index.
---
```python
def top_k(scores, k):
    n, m = scores.shape; k = min(k, m)
    cand = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k < m else np.broadcast_to(np.arange(m), (n, m)).copy()
    vals = np.take_along_axis(scores, cand, axis=1)
    order = np.lexsort((cand, -vals), axis=1)
    return np.take_along_axis(cand, order, axis=1), np.take_along_axis(vals, order, axis=1)
```
`nearest_neighbors` is `top_k(cosine_similarity(query_vecs, doc_vecs), k)`.
