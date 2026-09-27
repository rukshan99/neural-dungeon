A Neuron owns `self.w` (a list of `Value(rng.uniform(-1.0, 1.0))`, one per input, drawn in order) and `self.b = Value(0.0)`. Its call is a weighted sum that stays inside the graph: `act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)`, then `act.tanh()` if `self.nonlin`. Plain `sum()` without a Value `start` would begin from the integer 0, which works (via `__radd__`) but the `start=` form makes the intent clear. A Layer is a list of Neurons; an MLP is a list of Layers; every `parameters()` just concatenates the lists below it.
---
Sizes for the MLP: `sizes = [n_in] + list(n_outs)`, then `Layer(sizes[i], sizes[i + 1], rng, nonlin=(i != len(n_outs) - 1))` for `i in range(len(n_outs))`, so only the last layer is linear. `zero_grad` is `for p in self.parameters(): p.grad = 0.0`. `mse_loss` is `sum(((p - t) ** 2 for p, t in zip(preds, targets)), start=Value(0.0)) * (1.0 / len(preds))`.
---
The training loop, in order:

```python
rng = np.random.default_rng(seed)
model = MLP(2, [8, 1], rng)
losses = []
for _ in range(steps):
    preds = [model(x)[0] for x in XOR_INPUTS]
    loss = mse_loss(preds, XOR_TARGETS)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= lr * p.grad
    losses.append(loss.data)
return model, losses
```

If the loss will not fall: check the sign of the update (minus), and that `zero_grad()` runs *before* `backward()` every step.
