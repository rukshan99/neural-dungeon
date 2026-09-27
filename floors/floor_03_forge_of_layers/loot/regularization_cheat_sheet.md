# Regularization Cheat Sheet

*Loot from Floor 3. Everything you need to recognize a model that is memorizing, and to stop it.*

## The symptoms

| You see | It means |
|---|---|
| train loss keeps falling, val loss bottoms out and climbs | **overfitting**: capacity is being spent on noise |
| train accuracy far above val accuracy (gap > ~10 points) | overfitting |
| val loss ~ train loss, both high, both flat | **underfitting**: too small, too slow, too few epochs, or broken inputs |
| train loss is `nan` or jumps around | learning rate too high, or an unstable softmax/log; not a regularization problem |
| val loss noisy epoch to epoch | val set too small; average over more examples before you trust a diagnosis |

The rule Floor 3 uses, stated so a script can apply it: **overfitting** if `val[-1] > 1.10 * min(val)` and `train[-1] < train[argmin(val)]`; **underfitting** if `train[-1] > 0.5 * train[0]`; else **healthy**. The constants are honest defaults, not laws.

## Bias and variance, in one paragraph

Test error decomposes (for squared loss, exactly; for others, in spirit) into **bias** (the model is too simple to represent the pattern: underfitting), **variance** (the model changes a lot when the training sample changes: it is fitting the sample, not the population; overfitting) and irreducible **noise**. More capacity lowers bias and raises variance. Regularization is anything that lowers variance at a (hopefully smaller) cost in bias. More data lowers variance for free.

## The toolbox

### L2 penalty = weight decay

Add `0.5 * λ * Σ ||W||²` to the loss. Its gradient is `λ W`, so the SGD update becomes

```
W <- W - lr * (dW + λ W) = (1 - lr λ) W - lr dW
```

Every step shrinks every weight by a factor `(1 - lr λ)`, hence *weight decay*. Large weights are what let a network carve sharp boundaries around single points; taxing them smooths the function. Do **not** decay biases (or norm gains/shifts): they set offsets, not slopes, and decaying them only adds bias. Typical `λ`: 1e-4 to 1e-2 with plain SGD. (With Adam, "L2 in the loss" and "decay in the update" stop being equivalent; AdamW is the decay version, and it is the one you want.)

### Dropout

Train time: keep each unit with probability `1 - p`, and scale the kept ones by `1 / (1 - p)` (inverted dropout):

```
mask = (rng.random(x.shape) >= p) / (1 - p)
out  = x * mask            # E[out] = x
dx   = dout * mask         # same mask; dropped units get no gradient
```

Test time: identity, no scaling, because the expectation was already fixed at train time. It works by preventing co-adaptation: no unit can rely on a specific other unit being present, so features have to be individually useful. Typical `p`: 0.1 to 0.5 on wide fully connected layers. It hurts on small layers (a 64-wide layer losing half its units per step is starved, not regularized) and on convolutional or normalized layers (use it sparingly there, or not at all). **Always** put the model in eval mode before measuring anything.

### Early stopping

Track validation loss after every epoch. Keep a copy of the parameters at the best epoch; stop after `patience` epochs without a new best; restore the best copy. Networks learn broad patterns first and memorize noise later, so stopping early is a real regularizer, not a shortcut. It costs nothing, works on every model, and is the first thing to turn on. Choose `patience` ~ 5 to 10% of your epoch budget.

### Data augmentation

Show the model transformed copies of each example that keep the label (flips, crops, noise, small rotations for images; synonym swaps, dropout on tokens for text). More data lowers variance; augmented data is more data that you already own. Apply it only to the training set.

### Smaller model / fewer features

The bluntest lever and sometimes the right one. If a 3-layer MLP overfits 300 points, a 2-layer one might not.

### Normalization (BatchNorm, LayerNorm)

Primarily an optimization aid (faster, more stable training, tolerates larger learning rates), with a mild regularizing side effect in the batch version because each example's normalization depends on its batch-mates. Do not rely on it as your regularizer.

## When to use which

| Situation | Reach for |
|---|---|
| any model, any time | validation set + early stopping |
| MLP or transformer overfitting | weight decay (1e-4..1e-2), then dropout on the wide layers |
| tiny dataset, big model | all of the above, plus a smaller model or more data |
| labels are noisy | early stopping and weight decay; the noise is learned last |
| images | augmentation first, then weight decay |
| val loss still climbing after all that | you have a distribution shift or a leak, not a regularization problem |

Add one weapon at a time and watch the two curves. If train loss stops falling too, you have overshot into underfitting.

## The initialization table

Variance propagates as `Var(z) = fan_in * Var(w) * Var(x)` through each linear layer; the activation then keeps it (tanh near 0), halves it (ReLU) or clips it (saturated tanh).

| Init | `Var(w)` | Made for | 20 layers deep, width 256 |
|---|---|---|---|
| constant std 0.01 | 1e-4 | nothing | collapses (`0.16**20`) |
| LeCun normal | `1 / fan_in` | tanh, sigmoid-ish | healthy-ish, slow drift down |
| Xavier / Glorot uniform `U(-a, a)`, `a = sqrt(6 / (fan_in + fan_out))` | `2 / (fan_in + fan_out)` | tanh | healthy; collapses with ReLU (`sqrt(1/2)**20`) |
| He / Kaiming normal `N(0, 2 / fan_in)` | `2 / fan_in` | ReLU and friends | healthy |
| constant std 1.0 | 1 | nothing | tanh saturates (87% pinned); ReLU explodes (~11x per layer) |

Biases start at zero. Two failure modes to remember: a **std that shrinks** below 1e-2 means no signal and no gradient; a tanh whose **std looks fine (~1) but whose units are all at ±1** is equally dead, because `1 - tanh²` is ~0 everywhere. Measure both.

## The five-line loop, for reference

```python
model.zero_grad()                      # forget the last batch
logits = model.forward(X[idx])         # forward
L = loss.forward(logits, y[idx])       # measure
model.backward(loss.backward())        # backward
opt.step(model.grads())                # update
```

Shuffle every epoch. Keep the short last batch. Validate in eval mode after each epoch. Snapshot the best. Stop when patient.
