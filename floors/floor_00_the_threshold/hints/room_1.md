`np.arange(1, 101)` is the integers 1 to 100. Arrays have a `.sum()` method. The result is a numpy integer, so wrap it: `int(...)`.
---
`np.ones(n, dtype=np.float32)`. Without `dtype`, numpy gives you float64, and the trial will say so.
---
`np.asarray(x)` accepts lists and arrays alike. `.size` is the total element count. Return `int(np.asarray(x).size)`.
