# Grimoire of Derivatives

*Loot from Floor 2. Everything a backward pass needs, on one page.*

## The chain rule, as an engine uses it

For `out = f(a, b)` with upstream gradient `out.grad = dL/d out`:

```
a.grad += (d out / d a) * out.grad
b.grad += (d out / d b) * out.grad
```

Local derivative times upstream gradient, **added** into the child. If a value feeds several consumers, the total derivative is the sum over paths, which is exactly what the `+=` computes once every consumer has run.

## Scalar local derivatives

| forward `out` | `d out / d a` | `d out / d b` | note |
|---|---|---|---|
| `a + b` | 1 | 1 | |
| `a * b` | `b` | `a` | the *other* operand |
| `a ** n` (n a number) | `n * a**(n-1)` | | keep the `n`; `a / b` is `a * b**-1` |
| `-a` | -1 | | `a * -1` |
| `exp(a)` | `exp(a)` = `out.data` | | reuse the output |
| `log(a)` | `1 / a` | | |
| `tanh(a)` | `1 - tanh(a)**2` | | in (0, 1]; equals 1 only at 0 |
| `sigmoid(a)` | `s * (1 - s)` | | at most 0.25, at a = 0 |
| `relu(a)` | `1 if a > 0 else 0` | | 0 at exactly 0 by convention |

## Tensor rules

Same table elementwise, plus these.

| forward | backward |
|---|---|
| `a + b`, `a * b` with broadcasting | compute the elementwise local gradient at the **output** shape, then `unbroadcast(g, child.shape)` |
| `C = A @ B` | `dA = dC @ B.T`, `dB = A.T @ dC` (use `swapaxes(-1, -2)` for batched operands, then unbroadcast) |
| `sum(axis, keepdims)` | `g = out.grad`; if the axis was dropped, `g = expand_dims(g, axis)`; then `broadcast_to(g, input.shape)` |
| `mean(axis)` | `sum` times `1 / count`, where `count = input.size / output.size` |
| `reshape(shape)` | `out.grad.reshape(input.shape)` |
| `transpose(perm)` | `out.grad.transpose(argsort(perm))` (the inverse permutation) |
| `softmax_cross_entropy(z, y)` (mean over N) | `(softmax(z) - onehot(y)) / N` |

### The unbroadcast rule

Broadcasting copies data forward; copies collect gradient backward. To send a gradient `g` (output shape) back to a child of shape `s`:

1. While `g.ndim > len(s)`: `g = g.sum(axis=0)` (axes broadcasting inserted on the left).
2. For each axis where `s[i] == 1` but `g.shape[i] != 1`: `g = g.sum(axis=i, keepdims=True)`.

Invariant to assert everywhere: `grad.shape == data.shape`.

### Why the matmul rule is what it is

`C_ij = sum_k A_ik B_kj`. Then `dL/dA_ik = sum_j dL/dC_ij * B_kj = (dC @ B.T)_ik` and `dL/dB_kj = sum_i A_ik dL/dC_ij = (A.T @ dC)_kj`. If you forget, the shapes only fit one way.

## The topological-sort recipe

A node's `_backward` reads `out.grad`, which is complete only after **every consumer** of `out` has run. Reverse post-order DFS guarantees that.

```python
def backward(root):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)          # after all children
    build(root)
    root.grad = 1.0                 # or ones_like(root.data)
    for v in reversed(topo):        # root first, leaves last
        v._backward()
```

Iterative version for very deep graphs (Python's recursion limit is ~1000): keep an explicit stack of `(node, children_iterator)` pairs; append a node when its iterator is exhausted.

## Accumulation gotchas

1. **`=` instead of `+=`** drops every path but the last. `a*a` gives `a` instead of `2a`. Always `+=`.
2. **Forgetting `zero_grad`**. Parameters are leaves reused by every step's fresh graph; their grads add up across steps. Zero them before each `backward()`.
3. **Calling `backward()` twice on the same graph**. In a micrograd-style engine *intermediate* nodes accumulate too, so the second pass compounds (more than doubles). Build a fresh graph per pass, or zero every node.
4. **Detached constants**. Anything you compute with plain numpy inside a forward (a max for stability, a mask) has no gradient path. Usually intended; know when it is not.
5. **In-place edits of `.data`** after the forward pass silently make saved values stale. Backward closures capture the data they need at forward time.

## Gradient checking

Central difference: `(f(x + h) - f(x - h)) / (2h)` with `h ~ 1e-6` in float64 has error `O(h^2)`. Compare with `rtol ~ 1e-5`. Check one element at a time; for tensors, pass the upstream gradient through a *random* signal (`sum(out * R)`) so that transposes and reshapes cannot hide behind all-ones. Every curse on this floor is found this way in the wild.

## Vanishing and exploding: the product of Jacobians

With `x_{l+1} = act(W_l x_l)` and `delta_l = dL/dx_l`:

```
delta_l = J_l^T delta_{l+1},     J_l = diag(act'(z_l)) W_l
delta_0 = J_0^T J_1^T ... J_{L-1}^T delta_L
```

A product of L matrices. If each shrinks its input by a factor r, the gradient at layer 0 is `r^L` of the gradient at layer L. For iid `W ~ N(0, s^2)` of width n, in expectation and ignoring correlations, one layer multiplies the *squared* norm by about

```
s^2 * n * mean(act'(z)^2)
```

With `s = c / sqrt(n)` this is `c^2 * mean(act'^2)`:

| activation | init | `c^2` | `mean(act'^2)` | per-layer factor (squared norm) | 50 layers, width 64 (measured) |
|---|---|---|---|---|---|
| sigmoid | small (c = 0.5) | 0.25 | <= 1/16 | <= 1/64 | ~1e-46: **vanishes** |
| tanh | xavier (c = 1) | 1 | a bit under 1 | a bit under 1 | ~0.06-0.3: **survives**, slowly decaying |
| relu | he (c = sqrt 2) | 2 | 1/2 | 1 | ~0.3-7: **survives** |
| relu | large (c = 3) | 9 | 1/2 | 4.5 | ~1e16: **explodes** |

Sigmoid is the Wraith: its derivative is at most 0.25 *everywhere*, so no init rescues a deep plain sigmoid chain. He's `sqrt(2/n)` is exactly the constant that cancels ReLU's `1/2`. Xavier's `1/sqrt(n)` is the tanh analogue (derivative near 1 around 0).

**Residual connections** `x_{l+1} = x_l + f(x_l)` add an identity to the Jacobian: `delta_l = (I + J_l^T) delta_{l+1}`. The gradient can no longer vanish (the identity path has factor 1), but the squared norm now grows by about `1 + c^2 mean(act'^2)` per layer: residual + relu/he is `2^L`, which is why real residual networks scale the branch down (normalization layers, or an init that shrinks the branch by `1/sqrt(L)`).

## When you see it in practice

- Loss flat from step 0, early-layer grads ~1e-8: vanishing. Fix: ReLU-family activations, He/Xavier init, residuals, normalization.
- Loss goes NaN after a few steps: exploding. Fix: smaller init, gradient clipping, lower learning rate, normalization.
- Print `||grad||` per layer once. It costs one line and settles the argument.
