Do the whole forward in numpy on `logits.data`, exactly like every op in Room 2.3: `shifted = z - z.max(axis=1, keepdims=True)`, `log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))`, `loss = -log_probs[np.arange(N), targets].mean()`. Then `out = Tensor(loss, (logits,), "softmax_xent")`. Because you never built intermediate Tensors, `out._prev` is exactly `{logits}`.
---
The backward is the closed form. `probs = np.exp(log_probs)` (already stable), `probs[np.arange(N), targets] -= 1.0` turns it into softmax minus one-hot, and `logits.grad += probs / N * out.grad`. Do not forget `out.grad`: the loss might be multiplied by something downstream, and the trial checks.
---
```python
z = logits.data
n = z.shape[0]
rows = np.arange(n)
shifted = z - z.max(axis=1, keepdims=True)
log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
out = Tensor(-log_probs[rows, targets].mean(), (logits,), "softmax_xent")

def _backward():
    probs = np.exp(log_probs)
    probs[rows, targets] -= 1.0
    logits.grad += probs / n * out.grad

out._backward = _backward
return out
```
