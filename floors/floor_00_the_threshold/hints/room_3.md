Every function here is one or two lines. The trick is always the same: make the smaller array's shape line up with the bigger one *from the right*, inserting size-1 axes where they differ. `keepdims=True` on `mean`/`std`/`norm` does that insertion for you.
---
`center_rows`: `x - x.mean(axis=1, keepdims=True)`. `normalize_rows`: `x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)`. `standardize_columns` reduces over axis 0, and a (D,) result already broadcasts against (N, D), so no keepdims needed.
---
`outer_sum`: `a[:, None] + b[None, :]`. `pairwise_distances`: `diff = a[:, None, :] - b[None, :, :]` has shape (N, M, D); then `np.sqrt((diff ** 2).sum(-1))`. `scale_channels`: `images * scale`. `clamp_to_range`: `np.minimum(np.maximum(x, low), high)`.
