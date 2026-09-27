# Floor 2 — The Chain of Whispers

> *Every gradient you will ever compute is a message passed back along a chain. Learn to hear it before it fades.*

```
            ┆ stairs from Floor 1
            ▼
   ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │ 2.1 THE WHISPER-│─────│ 2.2 THE NEURON'S │─────│  2.3 THE TENSOR      │
   │    ING VALUE    │     │     WHISPER      │     │      WHISPER         │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐                │
   │ ☠ THE WRAITH'S  │─────│  2.4 THE CURSED  │────────────────┘
   │     CAVERN      │     │     BACKWARD     │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a short corridor behind the lair: the Fused Whisper
            ▼  stairs down to Floor 3
```

The stairs open onto a cavern so long you cannot see its end. Say something, and nothing comes back. Then, after a while, it does: your own words, whispered from the dark by a chain of voices, each one having heard it from the one behind. Some of them speak clearly. Some of them mumble. Fifty voices later, what reaches you is not always what was said.

That chain is backpropagation. A neural network is a long composition of functions, and training it needs the derivative of one number, the loss, with respect to every parameter in the composition. The chain rule says that derivative is a product of local derivatives along the path from the loss back to the parameter. Reverse-mode automatic differentiation (autograd) is the bookkeeping that computes all of those products in one pass, sharing work, at roughly the cost of the forward pass itself. Every framework you will use below this floor, `loss.backward()` included, is this floor with a faster runtime.

Building it yourself, twice, is not an academic exercise. Engineers who have written an autograd engine read `grad` shapes without fear, know exactly what `zero_grad()` clears and why, can tell a vanishing gradient from a bug in five minutes, and can gradient-check a custom op when a framework's numbers look wrong. The Wraith at the end of this floor is the reason deep networks did not work for twenty years. You will measure it, predict it, and starve it.

**You will learn:** the chain rule as an algorithm · reverse-mode autodiff · a scalar autograd engine · gradient accumulation and topological order · a tensor engine with broadcasting-aware gradients · vanishing and exploding gradients, and the activations and initializations that prevent them.

**You need:** Python 3.11+ and numpy. Floor 0 (shapes and broadcasting) is assumed; the `unbroadcast` rule in Room 2.3 is broadcasting run backwards. No PyTorch.

---

## The lore of whispers (read this before the rooms)

### The chain rule, one variable at a time

If `y = f(u)` and `u = g(x)`, then

```
dy/dx = dy/du · du/dx
```

Compose more functions and you multiply more factors. For `L = f_3(f_2(f_1(x)))`:

```
dL/dx = f_3'(·) · f_2'(·) · f_1'(x)
```

Each factor is a **local derivative**: the derivative of one operation with respect to its own input, evaluated at the value that flowed through it. Each whisperer in the chain knows only their own factor. The product is assembled by passing the message along.

When a variable reaches the output by **several paths**, the derivatives along the paths **add**. If `L` depends on `x` through `u_1, ..., u_k`:

```
∂L/∂x = Σ_i  ∂L/∂u_i · ∂u_i/∂x
```

This one line is the reason `grad` is accumulated with `+=` and never assigned with `=`.

### Forward mode vs reverse mode

You could compute `dL/dx` by pushing a derivative *forwards* from `x` through every operation. That costs one pass per input variable. A network has millions of inputs (its parameters) and one output (the loss). Reverse mode instead starts at the output with `dL/dL = 1` and pushes derivatives *backwards*, visiting each operation once. One pass yields `dL/d(everything)`. That asymmetry is why deep learning is possible on a budget, and it dictates the design below.

### The engine: a Value that remembers

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0              # dL/d(this), filled in by backward()
        self._backward = lambda: None
        self._prev = set(_children)  # the Values this one was computed from
        self._op = _op
```

Every operator does the same four things: compute the forward value, create a new `Value` that remembers its children, define a closure that will pass the gradient back, and attach it.

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad  += other.data * out.grad     # d(a·b)/da = b
        other.grad += self.data  * out.grad     # d(a·b)/db = a

    out._backward = _backward
    return out
```

Read the two lines inside `_backward` as: *local derivative × upstream gradient, added to the child*. `out.grad` is `dL/d out`, which the node's own consumers have already filled in by the time this closure runs. That "already" is the next idea.

### Topological order: who speaks when

A node's `_backward` reads `out.grad`. That value is only complete after **every consumer** of `out` has added its contribution. So nodes must run their backward closures in an order where every node comes *before* all of its children. A depth-first post-order walk of `_prev` appends children before parents; reversing it gives root first, leaves last.

```python
def backward(self):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)              # after all its children
    build(self)
    self.grad = 1.0                     # dL/dL: the whisper starts at full volume
    for v in reversed(topo):
        v._backward()
```

Two facts about this design that Rooms 2.1 and 2.4 test:

- **Grads accumulate across calls.** The engine never zeroes anything. Parameters are leaves reused by every training step's freshly built graph, so without `zero_grad()` each step's gradient is added to the last one's. Calling `backward()` twice on the *same* graph is worse than doubling: intermediate nodes accumulate too, and the second pass compounds. Build a fresh graph per step and zero the parameters first.
- **Order is not optional.** Walk the list forwards and the leaves are asked to speak before they have heard anything; every gradient one step from the root comes out zero.

### The local derivatives

| op | derivative | remark |
|---|---|---|
| `a + b` | 1, 1 | |
| `a * b` | `b`, `a` | |
| `a ** n` | `n · a^(n-1)` | `n` a plain number; `a / b = a · b^-1` |
| `exp(a)` | `exp(a)` | equals the output: reuse `out.data` |
| `log(a)` | `1 / a` | |
| `tanh(a)` | `1 − tanh(a)²` | in `(0, 1]` |
| `sigmoid(a)` | `s (1 − s)` | at most `0.25` |
| `relu(a)` | `1 if a > 0 else 0` | |

Two of them, `tanh` and `sigmoid`, can never exceed 1. Keep that in mind; the Wraith is built from it.

### Gradient checking: the habit that finds every bug

For a scalar function, the central difference

```
f'(x) ≈ (f(x + h) − f(x − h)) / 2h        h ≈ 1e-6 in float64
```

has error `O(h²)` and agrees with a correct analytic gradient to about six digits. Every trial on this floor checks your engine this way, and Room 2.4 asks you to do it yourself to find planted bugs. In real work: when a custom op's gradient looks wrong, do not stare at the formula; finite-difference it.

### Neurons, layers, training

A neuron is `tanh(w · x + b)`. Built from Values, it is differentiable with no extra code. A layer is `n_out` neurons reading the same input; an MLP is layers in sequence, with the last one linear so the output is not squashed. Training is the loop that every framework hides:

```python
preds = [model(x)[0] for x in inputs]        # forward: builds a fresh graph
loss = mse_loss(preds, targets)               # one Value at the root
model.zero_grad()                             # clear last step's whispers
loss.backward()                               # every parameter hears dL/dp
for p in model.parameters():
    p.data -= lr * p.grad                     # one step downhill
```

Mean squared error, `mean((pred − target)²)`, is the loss here because it is the simplest one whose gradient `2 (pred − target) / n` you can check in your head.

### Tensors: the same engine, one level up

Replace `float` with `np.ndarray` and every rule above holds elementwise. One thing is new. Broadcasting **copies** data on the way forward: `(3, 4) + (4,)` uses the vector three times. On the way back each copy has its own gradient, so the gradient of the original is the **sum over the copies**:

```
y[i, j] = x[i, j] + b[j]      ⇒      dL/db[j] = Σ_i dL/dy[i, j]
```

`unbroadcast(grad, shape)` performs exactly that reduction: sum away leading axes that broadcasting inserted, then sum (with `keepdims`) over axes that were 1 in `shape` but stretched in `grad`. Every binary op calls it before `+=`-ing into a child, so the invariant `grad.shape == data.shape` always holds.

The other tensor rules follow from writing out the sums once:

- **matmul.** `C = A @ B`, `C_ij = Σ_k A_ik B_kj`. Then `dL/dA_ik = Σ_j dL/dC_ij · B_kj`, i.e. `dA = dC @ Bᵀ`, and `dB = Aᵀ @ dC`. The shapes only fit one way; if you forget, let them tell you.
- **sum.** Every input element contributed with weight 1, so each receives the gradient of the sum it landed in: put the reduced axis back (`expand_dims`) and `broadcast_to` the input shape. **mean** is `sum × 1/count`.
- **reshape** sends the gradient back through the inverse reshape. **transpose** applies the inverse permutation, `np.argsort(perm)`.

### The Wraith: products of small numbers

Stack `L` layers, `x_{l+1} = act(W_l x_l)`, and write `δ_l = dL/dx_l`. One layer of backprop is

```
δ_l = J_lᵀ δ_{l+1},        J_l = diag(act'(z_l)) · W_l
```

so the gradient at the input is a **product of L Jacobians**: `δ_0 = J_0ᵀ J_1ᵀ ⋯ J_{L−1}ᵀ δ_L`. If each Jacobian shrinks its input by a factor `r`, the gradient at layer 0 is `r^L` of the gradient at layer L. For `r = 0.25` and `L = 50` that is `10^-30`. For `r = 2` it is `10^15`. Neither trains.

How big is `r`? For weights drawn iid from `N(0, s²)` at width `n`, in expectation and ignoring correlations, one layer multiplies the *squared* norm of `δ` by roughly

```
s² · n · mean(act'(z)²)
```

Writing `s = c / √n` (every named init on this floor has that form), the factor is `c² · mean(act'²)`. That formula is the whole boss fight:

| activation | init | `c²` | `mean(act'²)` | squared-norm factor per layer | measured, 50 layers |
|---|---|---|---|---|---|
| sigmoid | small (`c = 0.5`) | 0.25 | ≤ 1/16 | ≤ 1/64 | ~1e-46, vanishes |
| tanh | xavier (`c = 1`) | 1 | a little under 1 | a little under 1 | 0.06–0.3, survives |
| relu | he (`c = √2`) | 2 | 1/2 | **1** | 0.3–7, survives |
| relu | large (`c = 3`) | 9 | 1/2 | 4.5 | ~1e16, explodes |

Sigmoid's derivative is at most 0.25 *everywhere*, so no initialization saves a deep plain sigmoid chain: this is the Wraith. **He initialization**, `√(2/n)`, is precisely the constant that cancels ReLU's `1/2`. **Xavier/Glorot**, `1/√n`, plays the same role for tanh, whose derivative is near 1 around 0; at 50 layers it still decays by a factor of a few, which is why real deep networks add normalization layers as well.

**Residual connections** `x_{l+1} = x_l + f(x_l)` change the Jacobian to `I + J_l`: the identity path carries the gradient with factor exactly 1, so it cannot vanish. But the squared norm now *grows* by roughly `1 + c²·mean(act'²)` per layer. Residual + relu/he is about `2^L`. That is a real effect, the trial will show it to you, and it is why residual networks scale their branches down (normalization, or a `1/√L` factor at init).

---

## Rooms

Run `dungeon enter 2` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code (Room 2.4 excepted: there you fix code), then run that room's trial.

### 2.1 The Whispering Value — `rooms/room_1_whispering_value.py`

You meet the first whisperer and learn the pattern every other one follows. Implement the `Value` class: `+ - * / **` (float exponents), the reflected operators so `2 * v` works, `exp`, `log`, `tanh`, `relu`, `sigmoid`, and `backward()` with a topological sort.

The trial finite-differences every operator, then asks the hard questions: the **diamond** (`a*a + a` must give `2a + 1`, which only `+=` gets right), a node used at two depths, a **second `backward()`** without zeroing (it doubles, by design), and a chain of a hundred `tanh` whisperers that must still deliver a finite gradient.

```
dungeon trial 2 room_1
```

### 2.2 The Neuron's Whisper — `rooms/room_2_neurons_whisper.py`

The whisperers form rows. `Neuron`, `Layer`, `MLP` built from Values, each with `parameters()`; a `zero_grad()`; `mse_loss`; and `train_xor`, a hand-written SGD loop that teaches `MLP(2, [8, 1])` the four points of XOR in 200 steps.

The trial counts parameters (33 for `[2, 8, 1]`), checks the last layer is linear, checks the gradient against finite differences, insists `zero_grad` silences everything, and demands the loss fall below 0.05 with reproducible results for a given seed.

```
dungeon trial 2 room_2
```

### 2.3 The Tensor Whisper — `rooms/room_3_tensor_whisper.py`

The same engine over numpy arrays: `unbroadcast`, then `add`, `sub`, `mul`, `neg`, `pow`, `exp`, `log`, `relu`, `matmul`, `sum(axis, keepdims)`, `mean`, `reshape`, `transpose`, `backward`.

Every op is checked forward against numpy and backward against finite differences, through a *random* upstream signal so a wrong transpose cannot hide behind ones. The broadcast shapes are the ones that bite: `(3,4)+(4,)`, `(3,1)*(1,4)`, `(2,3,4)+(3,1)`, `sum` with and without `keepdims`, `axis=(0, 2)`. Then a two-layer MLP's gradients are compared with a backward pass derived by hand.

```
dungeon trial 2 room_3
```

### 2.4 The Cursed Backward — `rooms/room_4_cursed_backward.py`

A complete autograd engine someone left running. It has **exactly six bugs**. The trial has one targeted test per curse whose message describes the symptom (a whisper that never arrives, a derivative louder than it can be, a node that hears one message instead of two) but never the fix. Four curses are in the operators' backward closures; two are in `backward()` itself. Gradient-check your way through it. When all six are lifted, an integration test compares the whole engine with finite differences.

```
dungeon trial 2 room_4
```

---

## Boss: The Vanishing Wraith

```
                    .------.
                  .'  _  _  '.        "Say what you like at the far end.
                 /   (o)(o)   \        By the time it reaches the door
                |      /\      |       it will be a rounding error."
                |   \______/   |
                 \            /       The Wraith is fifty layers deep. Every
                .-'\   ..   /'-.      layer it passes through multiplies the
               /    \ .''. /    \     gradient by a derivative smaller than
              /      '    '      \    one. It does not need to attack you.
             ( ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ )  It only needs to wait.
              '~~~~~~~~~~~~~~~~~~~'
```

**Weakness:** activations and initializations whose per-layer factor `c² · mean(act'²)` is close to 1, and skip connections that give the gradient a path with derivative exactly 1.

- **Phase 1:** `gradient_norms_per_layer(depth, width, activation, init_scale, seed, use_residual)`. Build the chain in numpy (or with your Room 2.3 `Tensor`), backpropagate `loss = sum(x_depth)`, return `‖dL/dx_l‖` for every layer. The trial rebuilds the same network from the seed and compares against an independent backprop; then it summons the Wraith: sigmoid with small init at depth 50, where the norm at layer 0 must be below `1e-6` of the norm at layer 50.
- **Phase 2:** `WRAITH_PROPHECY`. For sigmoid/small, tanh/xavier, relu/he and relu/large, predict *vanishes* (ratio below `1e-3`), *explodes* (above `1e3`) or *survives*. Predict from the table above before running anything.
- **Phase 3:** `banish_the_wraith()`. Return a configuration (activation, init, residual or not) whose ratio stays within `[0.05, 20]` at depth 50, width 64, for seeds 0, 1 and 2. There is a textbook answer and a surprising one.

```
dungeon fight 2
```

## Secret room: The Fused Whisper *(optional)*

Behind the Wraith's cavern, six whisperers have been replaced by one. Add `softmax_cross_entropy(logits, targets)` to your tensor engine as a **single graph node**: a numerically stable log-softmax forward (subtract the row max), and the closed-form backward `(softmax − onehot) / N`. The trial checks it against the six-node composed version, feeds it logits of size 1000 and 3000 without tolerating a NaN, and gradient-checks it through a linear layer. This is how every real framework implements the classification loss, and why they ask for logits rather than probabilities.

```
dungeon trial 2 --secret
```

## Loot

Clear the four rooms and banish the Wraith to unlock:

- **Grimoire of Derivatives** — `loot/derivatives_grimoire.md`. Every local derivative on this floor, the unbroadcast rule, the topological-sort recipe, the accumulation gotchas, gradient checking, and the product-of-Jacobians argument with the table of who survives.
- **Micro Autograd** — `loot/micro_autograd.py`. A clean, commented scalar + tensor engine (numpy only, iterative topological sort so deep graphs do not hit the recursion limit), with the fused loss and a `gradient_check` helper. Run it to see its self-checks pass; copy it wherever you next need a gradient without a framework.

## Stuck?

- `dungeon hint 2 room_3` reveals one hint at a time (three per room).
- Trial failure messages describe the *concept* that went wrong and show the observed values. On this floor especially, read the numbers: a gradient that is exactly `x²` where `3x²` was due tells you which factor went missing.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, to compare, not to copy.

When `dungeon map` shows the Chain of Whispers cleared, take the stairs. Below, someone is hammering layers into shape, and the forge is running hot.
