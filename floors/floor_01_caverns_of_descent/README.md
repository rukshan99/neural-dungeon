# Floor 1 — The Caverns of Descent

> *The gauge on your wrist reads loss. Every corridor slopes down. The trick is knowing how big a step to take.*

```
            ▼ stairs from the Threshold
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  1.1 THE ALTAR  │─────│  1.2 THE NUMER-  │─────│  1.3 THE DESCENT │
   │     OF LOSS     │     │    ICAL ORACLE   │     │   (a long slope) │
   └─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
   ┌─────────────────┐     ┌──────────────────┐     ┌────────┴─────────┐
   │ ☠ THE LICH'S    │─────│  1.5 THE MOMENTUM│─────│  1.4 THE PROPHECY│
   │    SANCTUM      │     │      CHAMBER     │     │    OF CURVES     │
   └────────┬────────┘     └──────────────────┘     └──────────────────┘
            ┆  ◇ a whisper from a side ledge: the Whispering Saddle
            ▼  stairs down to Floor 2
```

The stairs from the Threshold end in a chapel carved into wet rock. On the altar sits a brass gauge with one needle, and the needle does not measure distance or time. It measures how wrong you are. Every passage that leads out of the chapel slopes downward, water runs along each floor in the same direction, and the deeper you go the narrower the valleys become: steep walls a hand's width apart, with a floor so flat you cannot tell which way is down.

This is the floor where training happens. A model is a function with knobs; a loss turns its mistakes into one number; the gradient says which way to turn each knob; the learning rate says how far. Everything that follows in the dungeon, from a linear regression to a language model with billions of knobs, is this loop. A large share of the training runs that fail in practice fail here: a loss that overflows to NaN, a learning rate a hair too large, a valley so narrow that a fixed step size can either survive the walls or reach the floor but not both. The Caverns exist so that you know *why* each of those happens, from the algebra, before a framework hides the loop from you.

**You will learn:** loss functions and their gradients (MSE, MAE, binary and softmax cross-entropy, stable logit forms) · numerical gradient checking · gradient descent · what a learning rate does, exactly · momentum, Nesterov and Adam · warmup and cosine schedules · the learning-rate range test · backtracking line search.

**You need:** Python 3.11+ and numpy. Floor 0's habits (shapes, `keepdims`, the row-max trick) are assumed.

---

## The lore of descent (read this before the rooms)

### Loss is a depth gauge

A model produces predictions from parameters `θ`. A loss compares predictions to targets and returns one non-negative number, averaged over the data:

```
L(θ) = (1/N) · Σᵢ ℓ(prediction_i(θ), target_i)
```

Training is finding θ that makes `L` small. Every loss on this floor is a **mean**. That matters more than it sounds: the gradient of a mean carries a factor `1/N`, and forgetting it is the most common gradient bug there is.

### The five losses and their gradients

Write `p` for a prediction, `t` or `y` for a target, `N` for the number of elements. The derivative of the per-element loss `ℓ` with respect to `p`, divided by `N`, is the gradient of the mean.

**Mean squared error** (regression):

```
ℓ = (p - t)²                dℓ/dp = 2 (p - t)
```

**Mean absolute error** (regression, robust to outliers):

```
ℓ = |p - t|                 dℓ/dp = sign(p - t)        (any value in [-1, 1] at p = t)
```

The gradient's size does not shrink as you approach the minimum, so MAE needs a decaying learning rate to settle.

**Binary cross-entropy from probabilities** (`p ∈ (0, 1)`, `y ∈ {0, 1}`):

```
ℓ = -[ y·ln p + (1 - y)·ln(1 - p) ]        dℓ/dp = (p - y) / (p (1 - p))
```

`ln 0` is `-inf`. A model that outputs exactly 0 or 1 (which float32 sigmoids do) produces an infinite loss and a NaN gradient. The standard defence is clipping `p` into `[ε, 1 - ε]` with `ε ≈ 1e-7`, which caps the per-element loss at `-ln ε ≈ 16.1`.

**Binary cross-entropy from logits.** A logit `z` is the raw score before the sigmoid, `p = σ(z) = 1 / (1 + e⁻ᶻ)`. Substituting and simplifying with `ln σ(z) = -softplus(-z)` and `ln(1 - σ(z)) = -softplus(z)`, where `softplus(z) = ln(1 + eᶻ)`:

```
ℓ = softplus(z) - y·z                       dℓ/dz = σ(z) - y
```

(using `softplus(-z) = softplus(z) - z`). This is not just tidier; it is the only form that survives large logits. `softplus(1000)` computed as `ln(1 + e¹⁰⁰⁰)` overflows to `inf`. Computed as

```
softplus(z) = max(z, 0) + ln(1 + e^(-|z|))
```

every exponent is `≤ 0` and the answer is exactly `1000.0`. numpy ships this as `np.logaddexp(0, z)`. The sigmoid itself needs the same care: `σ(z) = exp(-softplus(-z))` or `½(1 + tanh(z/2))` never overflow.

**Softmax cross-entropy** (multi-class, logits `z ∈ ℝᶜ`, integer label `y`):

```
softmax(z)_c = e^{z_c} / Σⱼ e^{z_j}
ℓ = -ln softmax(z)_y = logsumexp(z) - z_y
logsumexp(z) = m + ln Σⱼ e^{z_j - m}      with m = max(z)
dℓ/dz_j = softmax(z)_j - 1[j = y]
```

Subtracting the max makes every exponent `≤ 0`, and the `m` cancels algebraically, so the result is exact for any logits. Adding a constant to every logit of a row changes nothing (shift invariance), which is a good property to test. The gradient, "probabilities minus one-hot", has rows that sum to zero, and pushes the correct logit up and every other logit down in proportion to how much probability it stole.

Notice the pattern: for every loss paired with its natural output function, the gradient with respect to the pre-activation is `prediction - target`. That is not a coincidence; these losses are the negative log-likelihoods of the distributions the output functions parametrise.

### The oracle: checking a gradient numerically

You will derive gradients by hand on this floor and by backpropagation on the next. Both go wrong quietly. The defence is the definition of the derivative, made computable. From the Taylor expansion `f(x + h) = f(x) + h f'(x) + (h²/2) f''(x) + (h³/6) f'''(x) + …`:

```
forward difference    (f(x + h) - f(x)) / h          = f'(x) + (h/2) f''(x) + …       error O(h)
central difference    (f(x + h) - f(x - h)) / (2h)   = f'(x) + (h²/6) f'''(x) + …     error O(h²)
```

In the central formula the even powers cancel, so halving `h` quarters the error instead of halving it. For a function of an array, do this once per element: bump element `i` up, bump it down, divide, store; `np.ndindex(x.shape)` walks every index of any shape.

Why not make `h` tiny? `f(x + h)` and `f(x - h)` are two nearly equal float64 numbers. Their difference keeps only the digits in which they differ, about `16 + log₁₀ h` of them, and dividing by `2h` magnifies the rubble that remains. Below `h ≈ 1e-7` the estimate gets *worse* as `h` shrinks. The sweet spot for central differences in float64 is `h ≈ 1e-5` to `1e-6` (theory says `ε_machine^(1/3) ≈ 6e-6`). At `h = 1e-12` you are off by a factor of a million relative to `h = 1e-5`.

Compare *relatively*, so the verdict does not depend on the gradient's scale:

```
rel = ‖g_num - g_ana‖ / (‖g_num‖ + ‖g_ana‖ + tiny)
```

Below `1e-7`: correct. Around `1e-4`: a kink (relu, abs, max) near the point, or a loose `h`. Above `1e-2`: a bug. Around `1`: a sign error or a shape error. A missing `1/N` scores about `(N - 1)/(N + 1)`.

### Gradient descent

The gradient `g = ∇L(θ)` is the direction of steepest increase. To first order, `L(θ - lr·g) ≈ L(θ) - lr·‖g‖²`, so a small step against the gradient lowers the loss. The algorithm is the update, repeated:

```
θ ← θ - lr · ∇L(θ)
```

**What the learning rate does, exactly.** Take the simplest possible valley, `f(x) = ½·a·x²`, with gradient `a·x`. One step is

```
x_new = x - lr·a·x = (1 - lr·a) · x
```

so every step multiplies `x` by the same number `r = 1 - lr·a`, and after `k` steps `x_k = rᵏ·x₀`. Everything about convergence is in `r`:

| learning rate | `r` | what happens |
|---|---|---|
| `0 < lr < 1/a` | `0 < r < 1` | shrinks every step, same sign: converges smoothly |
| `lr = 1/a` | `0` | lands on the minimum in one step |
| `1/a < lr < 2/a` | `-1 < r < 0` | shrinks while flipping sign: converges, oscillating |
| `lr = 2/a` | `-1` | bounces between two walls forever |
| `lr > 2/a` | `r < -1` | grows without bound: diverges |

**Several directions at once.** For a quadratic bowl `f(x) = ½·xᵀAx` with `A` symmetric and eigenvalues `0 < μ = λ_min ≤ … ≤ λ_max = L`, change to the eigenvector basis and the problem falls apart into independent one-dimensional valleys, each with its own curvature `λᵢ` and its own factor `rᵢ = 1 - lr·λᵢ`. The *same* `lr` must keep every `|rᵢ| < 1`. Two consequences:

- **Stability is decided by the sharpest direction.** GD converges iff `lr < 2/L`. Nothing else matters for stability.
- **Speed is decided by the flattest direction.** With `lr` capped near `1/L`, the flat direction shrinks by only `1 - μ/L` per step. Reaching precision `ε` takes about `κ·ln(1/ε)` steps, where `κ = L/μ` is the **condition number**. The best constant `lr`, `2/(L + μ)`, improves this to about `(κ/2)·ln(1/ε)`; the dependence on `κ` remains.

Near any smooth minimum, the loss is approximately a quadratic bowl with `A` the Hessian, so this analysis is the local truth for every loss you will ever minimise. The steep-wall-flat-floor valleys of this floor's lore are literally large condition numbers.

**Conditioning of linear regression.** For `y_hat = Xw + b` and the MSE, with `Xₐ = [X | 1]` and `θ = [w; b]`:

```
loss = mean((Xₐθ - y)²)         ∇ = (2/N) Xₐᵀ (Xₐθ - y)         Hessian H = (2/N) XₐᵀXₐ
```

A feature with ten times the scale of another contributes a hundred times the curvature. **Standardizing** each column (subtract its mean, divide by its standard deviation) makes `H ≈ 2·I` when the features are uncorrelated, so `κ ≈ 1` and GD converges in a handful of steps where the raw problem needed hundreds. Room 1.3 makes you watch this. Correlated features stay ill-conditioned after standardizing; that is one job of the optimizers below.

### Momentum

Plain GD forgets everything between steps. **Heavy-ball momentum** keeps a velocity:

```
v ← β·v + g
θ ← θ - lr·v
```

(This is PyTorch's convention; some texts fold `lr` into `v`.) Along a direction where the gradient keeps pointing the same way, `v` accumulates toward `g/(1 - β)`: with `β = 0.9` the effective step is ten times `lr`. Along a direction where the gradient flips sign every step (the steep wall), successive gradients cancel in `v`. So momentum accelerates the flat floor and damps the steep wall, which is exactly what an ill-conditioned valley needs. With the optimal parameters on a quadratic,

```
lr = 4 / (√L + √μ)²        β = ((√L - √μ) / (√L + √μ))²
```

the contraction per step is `(√κ - 1)/(√κ + 1)` and the step count scales with `√κ` instead of `κ`. For `κ = 100` that is roughly ten times fewer steps; Room 1.5 has you race it. Momentum's own stability region on a quadratic is `lr < 2(1 + β)/L`.

**Nesterov momentum** evaluates the gradient at the look-ahead point `θ + β·v` rather than at `θ`. Written in terms of the look-ahead point itself (a change of variables), it becomes the form PyTorch uses for `SGD(nesterov=True)`:

```
v ← β·v + g
θ ← θ - lr·(g + β·v)
```

which is what you implement: the gradient you are handed is already the look-ahead gradient. On quadratics the two momentum methods behave similarly; Nesterov is a little better damped.

### Adam

Adam keeps two moving averages per parameter, of the gradient and of its square, and takes a step of about `lr` in every coordinate:

```
t ← t + 1
m ← β₁·m + (1 - β₁)·g               β₁ = 0.9
v ← β₂·v + (1 - β₂)·g²              β₂ = 0.999
m̂ = m / (1 - β₁ᵗ)                    bias correction
v̂ = v / (1 - β₂ᵗ)
θ ← θ - lr · m̂ / (√v̂ + ε)           ε = 1e-8
```

**Bias correction.** `m` and `v` start at zero, so early averages are biased toward zero: after one step `m = 0.1·g` and `v = 0.001·g²`, and `m/√v = 3.16·g/|g|`, a step three times larger than intended. Dividing by `1 - βᵗ` (exactly `0.1` and `0.001` at `t = 1`) fixes this; with it, Adam's first step is `lr·g/(|g| + ε)`, i.e. `±lr` in every coordinate, whatever the gradient's scale. Multiply every gradient by 1000 and Adam's trajectory is unchanged; that scale invariance is why one `lr` works across wildly different layers, and why Adam does not care that one coordinate of the Lich's valley is a thousand times steeper than the other.

Two honest caveats. Adam's normalisation is per coordinate, so it helps when the ill-conditioning is aligned with the axes; a rotated narrow valley is still hard. And a constant `lr` leaves Adam jittering around the minimum in a ball of radius proportional to `lr`, which is why it is almost always paired with a schedule.

**AdamW** is Adam plus *decoupled* weight decay, `θ ← θ - lr·λ·θ` applied directly to the parameters rather than added to `g` (where `√v̂` would normalise it away). It is the default optimizer for training transformers.

### Schedules

A learning rate does not have to be one number for all time. The modern default is **linear warmup followed by cosine decay**:

```
step < W :   lr = lr_max · step / W
otherwise:   progress = (step - W) / (T - W)
             lr = lr_min + ½ (lr_max - lr_min) (1 + cos(π · progress))
```

Warmup (typically 1 to 5 percent of `T`) protects the first steps, when Adam's `v̂` is estimated from a handful of gradients and a freshly initialised network produces unusually large, inconsistent ones. Decay shrinks the jitter ball at the end so the iterate can settle; decaying to `lr_min = 0` lets it settle completely. Other shapes you will meet: step decay (`× 0.1` every so many epochs), linear decay to zero (fine-tuning), one-cycle.

**The learning-rate range test.** How large can `lr_max` be? Ask the problem: from the same starting point, run a few dozen plain GD steps at each of a grid of candidates; a candidate *diverges* if any loss along the way is NaN or above the starting loss. The largest survivor is your upper bound; use a fraction of it. On a quadratic the survivor is exactly the largest candidate below `2/L`, so the test is a measurement of the sharpest direction's curvature, and the Lich checks that you get it right.

### Line search (the secret room)

Instead of a schedule chosen in advance, a **backtracking line search** asks the function at every step. Given a descent direction `d` at `x` (steepest descent uses `d = -g`), try a step `α`, and accept it only if the loss dropped by at least a fraction `c` of what the local slope predicted:

```
f(x + α d) ≤ f(x) + c · α · (g · d)          the Armijo condition, 0 < c < 1, typically 1e-4
```

Since `g · d < 0`, the right-hand side is below `f(x)`. If the test fails, shrink `α ← ρ·α` (typically `ρ = 0.5`) and try again. The accepted step always decreases `f`, so the loss history is monotone, and nobody chooses a learning rate. The cost is extra loss evaluations per step, which is a full forward pass over a batch in deep learning; that is why schedules and adaptive optimizers took over. But on a small smooth problem, steepest descent with backtracking walks Rosenbrock's banana valley on its own.

---

## Rooms

Run `dungeon enter 1` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial.

### 1.1 The Altar of Loss — `rooms/room_1_altar_of_loss.py`

The brass gauge must be taught what wrong means. Twelve short functions: MSE, MAE, binary cross-entropy from probabilities, a stable sigmoid, binary cross-entropy from logits, a stable log-softmax, softmax cross-entropy, and the gradient of each loss with respect to its prediction.

The trial checks known values (a coin flip costs `ln 2`; uniform logits over `C` classes cost `ln C`), symmetry, that logits of `±1000` produce finite and *exact* losses rather than `inf` or NaN, and that every gradient agrees with a finite-difference oracle. The 1/N is the usual casualty.

```
dungeon trial 1 room_1
```

### 1.2 The Numerical Oracle — `rooms/room_2_numerical_oracle.py`

A blind oracle in an alcove: it cannot read your formula, only nudge each element and feel the tilt. Implement forward and central differences for an input of any shape, and `gradient_check`, which returns the relative error and refuses gradients of the wrong shape.

The trial feeds the oracle a forgery (a gradient missing its factor of 2) and expects it caught, runs every gradient from the Altar through it, and then reads **the Prophecy**: you commit, in advance, to the error order of each scheme and to whether `eps = 1e-12` beats `eps = 1e-5`. The machine then measures.

```
dungeon trial 1 room_2
```

### 1.3 The Descent — `rooms/room_3_the_descent.py`

The first long slope. A linear model, its MSE and gradients, a generic `gradient_descent(grad_fn, params, lr, steps, callback)` loop that returns the whole trajectory, feature standardization, and a fit that recovers the true line from noisy data.

Then the canyon: the same data with one feature ten times the scale of the other. Both fits use the largest safe learning rate, `1/λ_max`. The raw canyon takes about 700 steps to reach the optimum; the standardized bowl takes 3. The trial demands at least 10x.

```
dungeon trial 1 room_3
```

### 1.4 The Prophecy of Curves — `rooms/room_4_prophecy_of_curves.py`

No code. A marble, a bowl with curvature `a = 4`, and four learning rates etched around the rim: `0.1, 0.25, 0.4, 0.6`. Say what each does (smooth, oscillating, or off into the dark), which one lands on the minimum in a single step, and where the cliff edge is. Then a two-dimensional bowl with curvatures `(1, 25)`: which coordinate caps the learning rate, at what value, and which coordinate crawls once it is capped.

Compute `r = 1 - lr·a` before you guess. The chamber runs the real thing and quotes your `r` back at you if you are wrong.

```
dungeon trial 1 room_4
```

### 1.5 The Momentum Chamber — `rooms/room_5_momentum_chamber.py`

A boulder at the top of a narrow valley. Four optimizer steps as pure functions, `(params, grads, state, hparams) → (new_params, new_state)`: SGD, heavy-ball momentum, Nesterov (PyTorch form), and Adam with bias correction; plus `run_optimizer`, the loop that drives any of them.

The trial checks each against its reference formula step by step, checks that nothing you were handed is modified, checks that Adam's first step is `±lr` in every coordinate regardless of the gradient's scale (`0.316·lr` means you skipped the bias correction), and then races momentum against SGD down a valley with condition number 100. SGD needs about 700 steps; heavy ball, about 100.

```
dungeon trial 1 room_5
```

---

## Boss: The Learning-Rate Lich

```
                    .-''''-.
                   /  _  _  \          "Choose your step, little optimizer.
                  |  (o)(o)  |          Too large, and my east wall throws you
                  |    /\    |          into the ceiling. Too small, and you
                   \  \__/  /           will still be crawling west when the
                    '-.__.-'            torches burn out. There is no number
                   /|  ||  |\           that pleases me."
                  / |  ||  | \
                 /  |  ||  |  \         Its valley: curvature 1 along one axis,
                    |  ||  |            1000 along the other. Plain GD must keep
                   /|  ||  |\           lr < 2/1000 to survive the wall, and at
                  '-'  ''  '-'          that lr the floor shrinks by 0.998 per
                                        step: ~3500 steps to reach 1e-3.
                                        You have 2000.
```

**Weakness:** it believes a learning rate is one number for all directions and all time. Per-coordinate steps (Adam) or momentum, with a schedule that decays as you settle, cross its valley with room to spare. And it cannot hide that stability is decided by the sharpest direction alone: the boundary is `2/λ_max`, and a range test measures it.

- **Phase 1:** `cosine_with_warmup(step, total_steps, warmup_steps, lr_max, lr_min)`. Linear warmup from 0, cosine to `lr_min`, clamped after the end. Checked to the decimal.
- **Phase 2:** `lr_range_test(grad_fn, loss_fn, x0, candidate_lrs, steps)`. The largest candidate that does not diverge, tested on an axis-aligned valley and on a tilted one against the analytic `2/λ_max`.
- **Phase 3:** `slay_the_lich(grad_fn, loss_fn, x0, max_steps)`. One implementation must bring the `κ = 1000` valley below `‖x‖ = 1e-3` within 2000 gradient evaluations *and* bring the Rosenbrock function from `(-1.5, 2.0)` below `1e-2` within 5000. The trial counts your calls.

```
dungeon fight 1
```

## Secret room: The Whispering Saddle *(optional)*

A ledge to the side of the Lich's sanctum, where the slope whispers how far to step. Implement `backtracking_line_search` (the Armijo condition, shrinking by `ρ`) and `gd_with_line_search`, steepest descent that asks the slope every step. The trial checks that the returned step satisfies Armijo and is the *first* in the backtracking sequence to do so, refuses uphill directions, and expects Rosenbrock to yield to plain steepest descent within 10,000 steps with a monotone loss history. Not required to clear the floor.

```
dungeon trial 1 --secret
```

## Loot

Clear the five rooms and defeat the Lich to unlock:

- **The Optimizer Grimoire** — `loot/optimizers_grimoire.md`. SGD, momentum, Nesterov, Adam, AdamW and bias correction on one page; the `2/L` rule and the `κ` versus `√κ` step counts; every schedule you will meet and why warmup exists; the gradient-checking recipe with thresholds; a table for reading a sick loss curve.
- **The Gradient Oracle (portable)** — `loot/gradient_check.py`. A clean, dependency-free numerical gradient checker with a readable verdict, to drop into any project's tests. Run it to see it catch a planted bug.

## Stuck?

- `dungeon hint 1 room_3` reveals one hint at a time (three per room).
- Failure messages quote the numbers they saw and the formula they expected. When a trial says "0.316 instead of 0.1", that is the bias correction; when it says "relative error 0.33", that is a missing factor of 2.
- `solutions/` exists. Use it after an honest attempt, to compare rather than copy; the reference solutions are commented for exactly that.

When `dungeon map` shows the Caverns cleared, take the stairs. Below, something is whispering your gradients back along a very long chain.
