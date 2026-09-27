# Floor 3 — The Forge of Layers

> *Autograd is a luxury. Down here you hammer every layer's backward by hand, and you learn why a network that fits everything has learned nothing.*

```
            ┆  stairs down from Floor 2
            ▼
   ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │  3.1 THE ANVIL  │─────│ 3.2 THE TEMPERING│─────│   3.3 THE BELLOWS    │
   │  forward/back   │     │   init & stats   │     │  the training loop   │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐                │
   │ ☠ THE HYDRA'S   │─────│ 3.4 THE MIRROR OF│────────────────┘
   │     LAIR        │     │    VALIDATION    │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a white glow behind the lair: the Batch Norm Crucible
            ▼  stairs down to Floor 4
```

The stairs from Floor 2 open onto heat. A forge fills the cavern: an anvil the size of a cart, a quenching trough, bellows taller than you, and a mirror of polished steel bolted to the far wall. The smith does not look up. "Floor 2 gave you an engine that differentiates anything," she says. "Good. Now put it down. A smith who cannot forge a blade without the machine does not understand blades."

On Floor 2 a graph remembered every operation and differentiated it for you. On this floor the unit of work is the **layer**: a function with a forward pass that caches what it needs, and a backward pass that turns the gradient at its output into the gradient at its input, by hand. This is how every deep learning framework is built underneath, and it is the level at which you debug them. A shape error in a custom `backward`, a loss that will not decrease, a model that scores 100% on training data and 60% on the rest: every one of those is diagnosed with what this floor teaches. The layers, the training loop, the validation split and the regularizers you write here are, line for line, what `torch.nn` and `torch.optim` do on Floor 4. You will recognize them.

Then the boss. It is not a bug. It is the most expensive mistake in applied machine learning: a model that memorizes its training set. You will build one on purpose, watch it grow, and learn the three weapons that kill it.

**You will learn:** layers with `forward`/`backward` and explicit caches · gradient checking · Xavier and He initialization and why depth punishes the alternatives · mini-batch SGD, shuffling, epochs · train/validation split and early stopping · diagnosing overfitting and underfitting · L2 weight decay and inverted dropout · (secret) batch normalization with its full backward.

**You need:** numpy and Floor 0's shape fluency. Floor 1's gradient-descent intuition helps. Floor 2's autograd engine is *not* used here, on purpose. No PyTorch until Floor 4.

---

## The lore of layers (read this before the rooms)

### A layer is a function with a memory

Every layer on this floor obeys one contract:

```python
out = layer.forward(x)        # compute; stash what backward needs in layer.cache
dx  = layer.backward(dout)    # dout = dL/dout (shape of out) -> dx = dL/dx (shape of x)
                              # and ADD dL/dparam into layer.grads for each parameter
```

`L` is the scalar loss at the top of the network. `dout` is how much `L` would change per unit change of each output element. The chain rule says `dL/dx = dL/dout · dout/dx`, but you never build the Jacobian `dout/dx` (for a batch of 32 through a 64-wide layer it would have 4 million entries). You compute its product with `dout` directly, and for every layer here that product is one or two lines of numpy. Stacking layers is then automatic: the `dx` of one layer is the `dout` of the layer below it.

Gradients **accumulate**: `backward` does `self.grads["W"] += ...`, not `=`. A parameter used twice in one graph must collect both contributions, which is why every framework works this way and why every training loop starts with `zero_grad()`.

### Linear, and where the transposes go

`out = x @ W + b` with `x (N, in)`, `W (in, out)`, `b (out,)`, `out (N, out)`. Element-wise, `out[n, j] = Σ_i x[n, i] W[i, j] + b[j]`. Differentiate `L` through it:

```
dL/dW[i, j] = Σ_n x[n, i] · dout[n, j]      ->  dW = x.T @ dout      (in, N) @ (N, out) = (in, out)
dL/db[j]    = Σ_n dout[n, j]                ->  db = dout.sum(axis=0)                     (out,)
dL/dx[n, i] = Σ_j dout[n, j] · W[i, j]      ->  dx = dout @ W.T      (N, out) @ (out, in) = (N, in)
```

Two facts worth tattooing: `dW` needs `x` **transposed** (the sum runs over the batch), and `db` is a **sum over the batch axis** (every example pushes on the same bias). The shapes force both, unless everything is square, which is why one trial makes everything square.

### ReLU and tanh

`ReLU(x) = max(x, 0)`; its derivative is 1 where `x > 0` and 0 elsewhere, so `dx = dout * (x > 0)`. Cache the mask, not `x`.

`tanh` has the convenient derivative `1 - tanh(x)²  = 1 - out²`. Cache the **output** and backward never needs the input.

### Softmax + cross-entropy, fused

For logits `z (N, C)` and integer labels `y (N,)`, softmax gives `p[n, k] = exp(z[n, k]) / Σ_j exp(z[n, j])` and the mean cross-entropy loss is

```
L = -(1/N) Σ_n log p[n, y[n]]
```

Write `log p[n, y] = z[n, y] - log Σ_j exp(z[n, j])` and differentiate with respect to `z[n, k]`: the first term gives `1[k = y]`, the second gives `exp(z[n, k]) / Σ_j exp(z[n, j]) = p[n, k]`. So

```
dL/dz = (p - onehot(y)) / N
```

One subtraction. That is why the two are always fused into one layer: separately, softmax's Jacobian is a dense `C×C` matrix per example that cancels almost entirely against the log's derivative. The `1/N` is not decoration: the loss is a *mean*, so its gradient is too. Forgetting it multiplies your effective learning rate by the batch size.

**Stability.** `exp(1000.0)` is `inf`; `log(0.0)` is `-inf`. Compute `log_softmax` directly, shifting by the row maximum first (the shift cancels mathematically and saves you numerically):

```python
shifted   = z - z.max(axis=1, keepdims=True)
log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))   # log-sum-exp
loss      = -log_probs[np.arange(N), y].mean()
```

Never take `log(probs)` after `probs` might have rounded to zero.

### Gradient checking

For any scalar function `f` of an array `x`, the central difference

```
df/dx[i] ≈ (f(x + ε e_i) - f(x - ε e_i)) / (2ε)          ε ~ 1e-6 in float64
```

is accurate to `O(ε²)`. To check a layer's `dx`, pick a random `dout` and check `backward(dout)` against the numerical gradient of `f(x) = Σ forward(x) * dout`, which is exactly the scalar whose gradient `backward` computes. Compare with a *relative* error, `|a - b| / (|a| + |b|)`, and expect below `1e-5`. This works in float64 only (float32 rounding is `~1e-7`, the same order as `ε`), which is why this whole floor runs in float64: `toydata` returns float32, so the trials cast with `X.astype(np.float64)` first. Do the same in your own experiments. It is also faster: numpy's matmul with mixed dtypes cannot use BLAS directly.

### Initialization: variance through depth

For `z = x @ W` with independent zero-mean entries,

```
Var(z_j) = Σ_i Var(x_i) Var(w_ij) = fan_in · Var(w) · Var(x)
```

So each linear layer multiplies the signal's variance by `fan_in · Var(w)`, and a 20-layer stack raises that factor to the 20th power. `0.5**20 ≈ 1e-6`: the signal is gone, and since gradients flow back through the same weights, so is every gradient. `2**20 ≈ 1e6` is no better. The activation then modifies the factor: tanh near zero is roughly linear (factor unchanged), ReLU discards the negative half (second moment halved), and a *saturated* tanh clips everything to `±1`, which keeps the std near 1 while making the derivative `1 - tanh²` vanish everywhere: a healthy-looking std hiding a dead network. Measure both the std and the fraction of units with `|a| > 0.99`.

| Init | `Var(w)` | Designed for |
|---|---|---|
| Xavier / Glorot | `2 / (fan_in + fan_out)` (uniform: `U(-a, a)`, `a = sqrt(6 / (fan_in + fan_out))`) | tanh; balances forward (`fan_in`) and backward (`fan_out`) |
| He / Kaiming | `2 / fan_in` (normal: `N(0, 2 / fan_in)`) | ReLU; the 2 pays for the discarded half |
| constant std (0.01, or 1.0) | ignores `fan_in` | nothing deeper than one layer |

Biases start at zero. Room 1's `Linear` uses `Var(w) = 1 / fan_in` (LeCun), a reasonable default that Room 2 will teach you to override.

### Mini-batch SGD: the loop

```python
for epoch in range(epochs):
    model.train()
    for idx in iterate_minibatches(n, batch_size, rng):    # a fresh permutation, every index once
        model.zero_grad()                                  # 1. forget the previous batch
        logits = model.forward(X[idx])                     # 2. forward
        L = loss.forward(logits, y[idx])                   # 3. measure
        model.backward(loss.backward())                    # 4. backward
        opt.step(model.grads())                            # 5. update
```

An **epoch** is one pass over the data. A **minibatch** gradient is an unbiased but noisy estimate of the full gradient at a fraction of the cost; the noise also helps escape sharp minima. **Shuffle every epoch**: fixed batches teach the model spurious co-occurrences, and a sorted dataset makes each batch a lie about the whole. Keep the short last batch. The optimizer owns the update rule; plain SGD with weight decay is

```
W <- W - lr · (dW + λ W)       for weight matrices (ndim >= 2)
b <- b - lr · db               biases are never decayed
```

`params()` hands the optimizer the model's *own* arrays, and `step` updates them in place (`p -= ...`). Rebinding (`p = p - ...`) would silently detach the optimizer from the model.

### The mirror: validation, early stopping, diagnosis

The training loss measures fit to the examples the model was shown. From inside the training set, learning the pattern and memorizing the examples look identical. Only data the model never trained on can tell them apart, so you **split** before training and evaluate the held-out **validation set** (in eval mode, no gradient) after every epoch.

- **Healthy:** both curves fall, the gap stays small.
- **Overfitting:** train keeps falling; val bottoms out and climbs. Every epoch past the minimum makes the model worse at the task.
- **Underfitting:** neither moves much. Too little capacity, too small a learning rate, too few epochs, or broken inputs.

**Early stopping** is the cheapest regularizer: remember the parameters at the best validation loss, stop when `patience` epochs pass without a new best, restore the best. Networks learn broad patterns first and noise later, so stopping early really does regularize.

This floor's `diagnose` rule, stated so a trial can check it: *overfitting* if `val[-1] > 1.10 · min(val)` and `train[-1] < train[argmin(val)]`; else *underfitting* if `train[-1] > 0.5 · train[0]`; else *healthy*. The constants are defaults, not laws.

### Regularization: L2 and dropout

**L2 penalty** adds `0.5 λ Σ ||W||²` to the loss. Its gradient is `λ W`, so the SGD update becomes `W <- (1 - lr λ) W - lr dW`: every step shrinks every weight a little, hence **weight decay**. Same regularizer, two views. Sharp decision boundaries around individual points need large weights; taxing them smooths the function. Biases are exempt.

**Inverted dropout** zeroes each unit with probability `p` at train time and scales the survivors by `1 / (1 - p)`:

```python
mask = (rng.random(x.shape) >= p) / (1 - p)     # E[mask] = (1 - p) / (1 - p) = 1
out  = x * mask                                  # E[out] = x
dx   = dout * mask                               # the SAME mask; dropped units get no gradient
```

Because the expectation is fixed at train time, **eval mode is the identity** and no test-time rescaling is needed. Dropout stops units from co-adapting (no unit can count on another being present) and works best on wide layers with redundant features. On small layers it starves rather than regularizes; you will see that at the boss. A model with dropout has two behaviours, so `Sequential.train()` / `.eval()` exist to switch every such layer, and every measurement (`accuracy`, `evaluate_loss`) must happen in eval mode.

### Batch normalization (the secret room)

`BatchNorm1d` normalizes each feature over the batch, then lets two learnable vectors restore any scale and shift the network wants:

```
mu = mean(x, axis=0)    var = var(x, axis=0)     xhat = (x - mu) / sqrt(var + eps)     out = γ xhat + β
```

It keeps running averages of `mu` and `var` (`running <- m · running + (1 - m) · batch`) for eval mode, where a single example has no batch to be normalized against. The backward is the one derivation on this floor that is not a one-liner, because `mu` and `var` depend on every row of `x`. With `dxhat = dout · γ` and sums over the batch:

```
dγ = Σ dout · xhat        dβ = Σ dout
dx = (1 / (N · sqrt(var + eps))) · (N · dxhat - Σ dxhat - xhat · Σ (dxhat · xhat))
```

The three terms are the direct path, the path through the mean and the path through the variance. The tempting shortcut `dx = dxhat / sqrt(var + eps)` keeps only the first and fails the gradient check.

---

## Rooms

Run `dungeon enter 3` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial. Rooms build on each other: Room 3 imports Room 1's layers, Room 4 imports Room 3's loop, the boss imports all of them.

### 3.1 The Anvil — `rooms/room_1_the_anvil.py`

The smith hands you a bar of raw matrix multiplication. Four layers to strike into shape: `Linear` (forward, and a backward that fills `dW`, `db` and returns `dx`), `ReLU`, `Tanh`, and the fused `SoftmaxCrossEntropy` whose `backward()` takes no argument because it sits at the top of the graph.

The trial gradient-checks every one of them against central finite differences, and the two classic bugs have their own tests: `dW` without the transpose and `db` without the batch sum. One test makes every shape square so that a wrong transpose gives the right shape and the wrong number. Finish with a two-layer network that must gradient-check end to end.

```
dungeon trial 3 room_1
```

### 3.2 The Tempering — `rooms/room_2_the_tempering.py`

Four initializers (`xavier_uniform`, `he_normal`, `small_normal`, `large_normal`) and an instrument: `activation_statistics` pushes a standard-normal batch through 20 square layers and records the std and the saturated fraction after each. Then **the Prophecy**: for six (init, activation) pairs, predict *before running anything* whether the signal collapses, saturates, or stays healthy at depth 20. Reason from `Var(z) = fan_in · Var(w) · Var(x)`. One of the six surprises most people.

```
dungeon trial 3 room_2
```

### 3.3 The Bellows — `rooms/room_3_the_bellows.py`

The training loop. `Sequential` (forward in order, backward in reverse, flat `params()`/`grads()` dicts keyed `"0.W"`, `"0.b"`, ...), `SGD` with weight decay that never touches biases, `iterate_minibatches` that yields every index exactly once per epoch in a fresh random order, `train` (the five lines, one mean loss per epoch) and `accuracy` (in eval mode, mode restored afterwards).

The fire: a `[2, 64, 64, 3]` ReLU MLP with He init must reach 90% training accuracy on three interleaved spirals in 800 epochs, about a second of CPU. If it does not, the loop is not learning, and the trial tells you which of the five lines to suspect.

```
dungeon trial 3 room_3
```

### 3.4 The Mirror of Validation — `rooms/room_4_mirror_of_validation.py`

`evaluate_loss`, `train_with_validation` (Room 3's `train` one epoch at a time, then the mirror; with `patience`, early stopping that restores the best parameters), `early_stopping` (argmin and a stop flag), and `diagnose` (the three-step rule). Then a second **Prophecy**: three pairs of curves scratched into the mirror's frame, an eager apprentice, a blunt hammer and a tempered blade. Diagnose them by eye before the trial applies the rule.

The trial's centrepiece trains a big network on 45 noisy points and asserts that the validation loss climbs to at least 1.5x its minimum while the training loss keeps falling: overfitting, made visible on purpose so you recognize it forever.

```
dungeon trial 3 room_4
```

---

## Boss: The Overfit Hydra

```
                     __/\__      __/\__
                    (  oo  )    (  oo  )        "Show me your training set.
                     \ \/ /      \ \/ /          I will learn every example.
                 __/\_\  /__/\__  \  /_/\__     Every. Single. One."
                (  oo  )( oo  )(  oo  )(  oo )
                 \ \/ /  \ \/ /  \ \/ /  \ \/ /   Each head is one training
                  \  /    \  /    \  /    \  /    example the model has
                   ||      ||      ||      ||     MEMORIZED. One in five of
                 __||______||______||______||__   the labels in its lair is
                /                              \  a lie, and the Hydra grows
               (   t r a i n i n g   s e t     )  a head for every lie it
                \______________________________/  learns by heart.
```

**Weakness:** regularization and validation. Weight decay makes memorizing expensive, dropout makes it unreliable, early stopping cuts it short, and a validation set is the only way to see the heads at all.

**The arena.** 300 training points on three spirals (`make_spirals(n_per_class=100, noise=0.05, seed=1)`), with 60 of the labels (20%) flipped to a wrong class. A clean validation set of 600 points. A **head** is a training example whose label was flipped *and* whose predicted class equals the flipped label: the model learned the lie. `summon_the_hydra()` in the boss file builds the identical arena so you can experiment.

- **Phase 1:** `l2_penalty` (exactly `0.5 λ Σ ||W||²`, biases exempt; the trial checks its gradient is the decay term SGD adds) and `Dropout` with inverted scaling and train/eval modes (eval is the identity; train zeroes a fraction `p`, scales the rest by `1/(1-p)`, and backward reuses the same mask).
- **Phase 2:** `count_heads(model, X_train, y_noisy, y_clean)`.
- **Phase 3:** the trial first trains a `[2, 128, 128, 3]` net with *your* Room 3 loop and no regularization for 1500 epochs, and shows it swallowing at least 20 of the 60 lies. Then `slay_the_hydra(X_train, y_noisy, X_val, y_val, rng)` must return a model with validation accuracy of at least 0.80, at most 15 heads, and a train-val accuracy gap of at most 0.20.

Try the weapons one at a time and watch the head count. Weight decay around `1e-3` plus early stopping on validation loss is enough; dropout is implemented and worth trying, and on a 64-wide network it will teach you something about where dropout belongs.

```
dungeon fight 3
```

## Secret room: The Batch Norm Crucible *(optional)*

Behind the lair, a crucible that brings every feature to mean zero and unit variance. Implement `BatchNorm1d` with train/eval modes, running statistics (`momentum` convention: `running <- m · running + (1 - m) · batch`, biased variance throughout), learnable `gamma`/`beta`, and the full three-term backward. The trial gradient-checks `dx`, `dgamma` and `dbeta`, verifies that eval mode uses the running statistics and changes nothing, and checks two consequences of the exact formula: `dx` sums to zero over the batch and is orthogonal to `xhat`.

```
dungeon trial 3 --secret
```

## Loot

Clear the four rooms and slay the Hydra to unlock:

- **Training Loop Template** — `loot/training_loop_template.py`. A clean, commented numpy training loop with minibatches, validation, early stopping, L2 and dropout. Runs on its own; copy it into any project and swap the layers.
- **Regularization Cheat Sheet** — `loot/regularization_cheat_sheet.md`. Symptoms, the bias/variance framing, L2 vs weight decay, dropout math, early stopping, augmentation, a when-to-use-which table, and the initialization table.

## Stuck?

- `dungeon hint 3 room_1` reveals one hint at a time (three per room). Hint 3 is nearly the code.
- The trial failure messages name the concept or the shape that is wrong and show the observed value. Read them before you read anything else.
- Every gradient trial compares against finite differences. If a check fails, print your analytic gradient and the numerical one side by side; the pattern of the difference (a factor of `N`, a transpose, a missing term) usually names the bug.
- `solutions/` exists. Use it as a walkthrough after an honest attempt, to compare, not to copy.

When `dungeon map` shows the Forge cleared, take the stairs. Below, the autograd engines are humming, and they speak PyTorch.
