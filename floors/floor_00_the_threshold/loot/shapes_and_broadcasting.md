# Grimoire of Shapes and Broadcasting

*Loot from Floor 0. One page. Pin it next to your terminal.*

## The four rules of broadcasting

Given two shapes:

1. **Align at the right edge.** Pad the shorter shape with 1s on the *left*.
   `(N, D)` and `(D,)` become `(N, D)` and `(1, D)`.
2. **Compare pairwise.** Each pair of dims must be equal, or one of them must be 1.
3. **Stretch the 1s.** The result dim is the larger of the pair.
4. **Anything else is an error.** `(3,)` vs `(4,)` fails. `(2, 1)` vs `(8, 4, 3)` fails at 2 vs 4.

```
(8, 1, 6, 1)
(   7, 1, 5)  ->  (8, 7, 6, 5)
```

## Inserting axes on purpose

| You have | You want | Write |
|---|---|---|
| `(N,)` per-row values to scale `(N, D)` | `(N, 1)` | `v[:, None]` |
| `(D,)` per-column values to scale `(N, D)` | `(1, D)` or `(D,)` | `v` already works (rule 1) |
| `(C,)` per-channel gain for `(B, C, H, W)` | `(1, C, 1, 1)` | `v[None, :, None, None]` or `v.reshape(1, -1, 1, 1)` |
| `(B, T)` mask for `(B, T, D)` | `(B, T, 1)` | `mask[..., None]` |
| a reduction that still broadcasts back | same ndim | `x.mean(axis=1, keepdims=True)` |

`None` and `np.newaxis` are the same thing.

## Reshape vs transpose

- `reshape` reinterprets the **same memory** in a new shape. The element order is unchanged. `x.reshape(B, -1)` computes one dim for you. `-1` at most once.
- `transpose` / `np.moveaxis` / `np.swapaxes` **reorders axes**. The elements you see change position.
- Reshaping when you meant to transpose *runs without error* and gives garbage. `(B, H, W, C) -> (B, C, H, W)` is `x.transpose(0, 3, 1, 2)`, never a reshape.
- Both return views when they can. Transposed arrays are not contiguous; `.copy()` or `np.ascontiguousarray` if a library complains.

## Indexing that changes rank

| Expression | Result shape from `(N, D)` |
|---|---|
| `x[0]` | `(D,)` |
| `x[0:1]` | `(1, D)` |
| `x[:, -1]` | `(N,)` |
| `x[:, -1:]` | `(N, 1)` |
| `x[None]` | `(1, N, D)` |
| `x[..., None]` | `(N, D, 1)` |
| `x[[0, 2]]` | `(2, D)` (fancy indexing copies) |
| `x[np.arange(N), idx]` | `(N,)` one element per row |

## Reductions

`sum`, `mean`, `max`, `std`, `argmax`, `np.linalg.norm` all take `axis=`. The named axis *disappears* unless `keepdims=True`. `axis=(0, 2)` reduces two axes at once. Negative axes count from the end, so `axis=-1` is "the last one" regardless of rank.

## Stack vs concatenate

- `np.stack([a, b], axis=k)` creates a **new** axis of length 2 at position k. Inputs must have identical shapes.
- `np.concatenate([a, b], axis=k)` **extends** an existing axis. Inputs must match on every other axis.

## Vectorization idioms

| Loop | Vectorized |
|---|---|
| set `out[i, labels[i]] = 1` | `out[np.arange(n), labels] = 1` |
| window sums | `c = cumsum([0, *x]); c[k:] - c[:-k]` |
| pairwise squared distance | `\|a\|² + \|b\|² - 2 a·b` via `sq[:, None] + sq[None, :] - 2 a @ b.T` |
| pairwise anything | `a[:, None, :] op b[None, :, :]` then reduce axis -1 |
| softmax | `e = exp(x - x.max(-1, keepdims=True)); e / e.sum(-1, keepdims=True)` |
| conditional assignment | `np.where(cond, a, b)` |

## Einsum idioms

```
"ij,jk->ik"       matmul
"bij,bjk->bik"    batched matmul
"ii->"            trace          "ii->i"  diagonal      "bii->bi"  batched diagonal
"i,j->ij"         outer          "nd,nd->n"  row-wise dot
"bqd,bkd->bqk"    attention scores (every query x every key)
"bhqd,bhkd->bhqk" ...with heads
"bi,ij,bj->b"     bilinear form x W y
```

Letters missing from the output are summed. A letter repeated within one operand walks its diagonal.

## dtypes

- numpy defaults to `float64`; deep learning lives in `float32`, `bfloat16`, `float16`. Say `dtype=` explicitly.
- Integer arrays do not silently become floats: `np.zeros(3, dtype=int) / 2` is float, but `x[i] = 0.5` into an int array truncates to 0.
- `x.astype(np.float32)` copies. Mixed-dtype operations promote to the wider type.

## The shape bugs everyone hits

1. `(N,)` minus `(N, 1)` silently broadcasts to `(N, N)`. If your loss is suddenly a matrix, look here.
2. Forgetting `keepdims=True` and then dividing `(N, D)` by `(N,)`: error if `D != N`, *silent wrong answer* if `D == N`.
3. Reshape instead of transpose: no error, scrambled pixels.
4. `x[:, -1]` when you needed `x[:, -1:]`. Dropped a dim, broke a later matmul.
5. `axis=0` when you meant `axis=1`. Print the shape after every reduction until it is second nature.
