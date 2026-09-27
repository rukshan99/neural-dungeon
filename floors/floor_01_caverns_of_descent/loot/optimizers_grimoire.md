# The Optimizer Grimoire

*Loot from Floor 1. One page of formulas you will eventually type from memory. Until then, pin it.*

Notation: `θ` parameters, `g = ∇L(θ)` the gradient, `lr` the learning rate, `t` the step count (starting at 1), `κ` the condition number.

## Update rules

| Optimizer | State | Update | Notes |
|---|---|---|---|
| **SGD** | none | `θ ← θ - lr·g` | The baseline. Stable on a quadratic iff `lr < 2/L`. |
| **Momentum** (heavy ball) | `v` | `v ← β·v + g`  <br> `θ ← θ - lr·v` | PyTorch convention. Along a constant gradient `v → g/(1-β)`, so the effective step is `lr/(1-β)`: with β = 0.9, ten times lr. Typical β = 0.9. |
| **Nesterov** | `v` | `v ← β·v + g` <br> `θ ← θ - lr·(g + β·v)` | The form `torch.optim.SGD(nesterov=True)` uses. Its iterate is the look-ahead point of the classical formulation (a change of variables), so `g` is already the look-ahead gradient. |
| **Adam** | `m, v, t` | `t ← t+1` <br> `m ← β₁·m + (1-β₁)·g` <br> `v ← β₂·v + (1-β₂)·g²` <br> `m̂ = m/(1-β₁ᵗ)`, `v̂ = v/(1-β₂ᵗ)` <br> `θ ← θ - lr·m̂/(√v̂ + ε)` | Defaults β₁ = 0.9, β₂ = 0.999, ε = 1e-8. Per-coordinate step ≈ lr early on, whatever the gradient scale. lr 1e-3 is the classic default for networks, 3e-4 the classic "it just works". |
| **AdamW** | `m, v, t` | Adam's step, then `θ ← θ - lr·λ·θ` | *Decoupled* weight decay: the decay acts on θ directly, not through `g` (L2 through `g` gets normalised away by `√v̂`). Typical λ = 0.01 to 0.1. |

### Why bias correction

`m` and `v` start at zero, so after one step `m = 0.1·g` and `v = 0.001·g²`. Without correction the ratio `m/√v` is `0.1/0.0316 = 3.16` times `g/|g|`: the first step is 3.16·lr, not lr. Dividing by `1 - βᵗ` (which is exactly `0.1` and `0.001` at t = 1) undoes the shrinkage. After a few hundred steps `1 - βᵗ ≈ 1` and the correction vanishes on its own.

### Adam in one sentence

Sign-descent with momentum and a per-coordinate step around `lr`, which is why it does not care that one direction of the loss is a thousand times steeper than another, *as long as those directions are the coordinate axes.* A rotated narrow valley is still hard for Adam.

## Stability: the 2/L rule

For `f(x) = ½·a·x²` one step of GD is `x ← (1 - lr·a)·x`. Call `r = 1 - lr·a`:

| `lr` | `r` | behaviour |
|---|---|---|
| `0 < lr < 1/a` | `0 < r < 1` | shrinks every step, same sign: smooth |
| `lr = 1/a` | `0` | exact minimum in one step |
| `1/a < lr < 2/a` | `-1 < r < 0` | shrinks while flipping sign: oscillates |
| `lr = 2/a` | `-1` | bounces forever between two walls |
| `lr > 2/a` | `r < -1` | grows: diverges |

For a quadratic `½·xᵀAx` with Hessian eigenvalues `0 < μ ≤ … ≤ L`, GD acts on each eigen-direction independently with its own `rᵢ = 1 - lr·λᵢ`, and one lr must satisfy all of them:

- **Stable iff `lr < 2/L`.** The sharpest direction alone sets the ceiling.
- **Best constant lr:** `2/(L + μ)`, contraction `(κ-1)/(κ+1)` per step, `κ = L/μ`. Steps to shrink the error by `1/ε`: about `(κ/2)·ln(1/ε)`. With the safe `lr = 1/L` the rate is `1 - 1/κ` and it takes about `κ·ln(1/ε)` steps.
- **Heavy ball, optimal:** `lr = 4/(√L + √μ)²`, `β = ((√L - √μ)/(√L + √μ))²`, contraction `(√κ-1)/(√κ+1)`: about `(√κ/2)·ln(1/ε)` steps. κ = 100 → 10x fewer steps than GD; κ = 10 000 → 100x. Heavy ball's own stability region on a quadratic is `lr < 2(1+β)/L`.
- For a general smooth `f`, `L` is the largest Hessian eigenvalue you will meet along the path (the Lipschitz constant of the gradient). It *changes* as you move: Rosenbrock's Hessian eigenvalues are about (10, 2100) at (-1.5, 2) and about (0.4, 1000) at the minimum (1, 1): the safe step and the condition number both change along the way. That is one reason schedules exist.

### Conditioning of linear regression

For `L = mean((Xw + b - y)²)` the Hessian is `H = (2/N)·XₐᵀXₐ` with `Xₐ = [X | 1]`. A feature with 10x the scale of another contributes 100x the curvature. Standardizing the columns (subtract mean, divide by std) makes `H ≈ 2·I` when the features are uncorrelated: `κ ≈ 1`, and GD converges in a handful of steps. Strongly correlated features stay ill-conditioned after standardizing; that is what second-order methods and Adam-style preconditioning are for.

## Schedules

| Schedule | `lr(t)` | Used for |
|---|---|---|
| constant | `lr₀` | debugging, small problems |
| step decay | `lr₀·γ^⌊t/k⌋` | classic vision recipes (γ = 0.1 every 30 epochs) |
| linear decay | `lr₀·(1 - t/T)` | fine-tuning language models |
| cosine | `lr_min + ½(lr_max - lr_min)(1 + cos(π·t/T))` | most pretraining runs |
| **warmup + cosine** | `t < W`: `lr_max·t/W`; then cosine on `(t-W)/(T-W)` | the modern default |
| one-cycle | linear up to `lr_max` then down below `lr₀` | fast training on a fixed budget |

**Why warmup.** Early on, Adam's `v̂` is estimated from a handful of gradients and can be badly off; the first gradients of a randomly initialised network are also unusually large and inconsistent. Ramping `lr` up over the first 1 to 5 percent of steps keeps those early steps from wrecking the initialisation.

**Why decay.** Noise (mini-batches) and oscillation (lr near `2/L`) both leave the iterate jittering in a ball whose radius is proportional to `lr`. Shrinking `lr` shrinks the ball, and decaying to (near) zero lets the iterate settle. Decay trades early speed for late precision.

**LR range test.** Try a geometric grid of candidates (`1e-5, 3e-5, 1e-4, …`) for a few dozen steps each from the same start; a candidate *diverges* if the loss goes NaN or above its starting value. Keep the largest survivor as an upper bound; use a fraction of it (a half to a tenth) as `lr_max`. On a quadratic the survivor is the largest candidate below `2/L`, exactly.

## Gradient checking, the recipe

1. **Work in float64.** In float32 even a correct gradient often scores 1e-4 to 1e-2, which cannot be told apart from a subtle bug.
2. **Central differences**, `(f(x + h·eᵢ) - f(x - h·eᵢ)) / 2h`, with `h ≈ 1e-5` (1e-6 to 1e-4 all fine). Error `~ h²·f'''/6`. Forward differences have error `~ h·f''/2`; do not use them for checking.
3. **Do not shrink `h` further.** `f(x+h) - f(x-h)` is a difference of nearly equal numbers; below `h ~ 1e-7` cancellation dominates and the estimate gets *worse*. The float64 sweet spot for central differences is about `(machine epsilon)^(1/3) ≈ 6e-6`.
4. **Compare relatively:** `‖g_num - g_ana‖ / (‖g_num‖ + ‖g_ana‖ + tiny)`, or element-wise `|a - b| / max(|a|, |b|, tiny)`.
5. **Read the number:** `< 1e-7` correct · `1e-7 to 1e-4` suspicious (a kink, or loose `h`) · `> 1e-2` wrong · `≈ 1` sign error or shape error.
6. **Kinks** (relu, abs, max, sort) are not differentiable at ties. A perturbation that crosses a kink gives a large error at that element only; nudge the input away from the kink or exclude that element. Do not "fix" the formula.
7. **Random points, not special ones.** At zeros or symmetric inputs bugs cancel. Use small random inputs, and check the gradient with respect to *every* input the function has (weights, biases, and the data if a layer's backward pass returns it).
8. **Sum vs mean.** The single most common failure is a missing `1/N`. The relative error will be close to `(N-1)/(N+1)`.

`gradient_check.py` next to this file does all of this and prints a verdict.

## Reading a loss curve

| Symptom | Likely cause | Fix |
|---|---|---|
| NaN within a few steps | lr past `2/L`, or `exp` of a large logit in the loss | stable loss formulas (`logaddexp`, max-subtracted log-softmax); halve lr; clip gradients |
| flat at `ln C` (classification) or `var(y)` (regression) | predictions are constant: no gradient flowing, lr far too small, or labels misaligned with inputs | gradient-check, raise lr, shuffle labels and confirm the loss gets *worse* |
| zig-zag, no progress | lr near `2/L` in the steep direction | halve lr, or add momentum and lower lr |
| fast drop, then slow crawl | ill-conditioning: steep directions done, flat ones crawling | standardize inputs; momentum; Adam |
| falls, then climbs late | curvature increased along the path, lr now too big | decay schedule |
| Adam's first steps are wild | missing bias correction (steps 3x too big), or `ε` too small with tiny gradients | check `1 - βᵗ`; raise `ε` |
