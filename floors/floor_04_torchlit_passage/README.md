# Floor 4 — The Torchlit Passage

> *The torches are already lit. Someone else built the fire. Learn to trust it without being fooled by it.*

```
            ▲ stairs up from Floor 3, where you forged layers by hand
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  4.1 THE FIRST  │─────│  4.2 THE MODULE  │─────│  4.3 THE LOADER'S│
   │      TORCH      │     │      FORGE       │     │     LANTERN      │
   └─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
   ┌─────────────────┐     ┌──────────────────┐     ┌────────┴─────────┐
   │  4.6 THE HALF-  │─────│  4.5 THE DEVICE  │─────│  4.4 THE CURSED  │
   │      LIGHT      │     │      FERRY       │     │  TRAINING LOOP   │
   └────────┬────────┘     └──────────────────┘     └──────────────────┘
            │
   ┌────────┴────────┐
   │ ☠ THE REVENANT'S│  ◇ a whisper behind the crypt wall: the Custom Whisper
   │      CRYPT      │
   └────────┬────────┘
            ┆
            ▼  stairs down to Floor 5
```

You come down the stairs from the Forge of Layers with soot on your hands. You have written backpropagation. You have initialised weights, batched data, watched a validation curve turn on you. Every one of those things you did in numpy, by hand, in the dark.

This passage is lit. Along its walls, in iron brackets, hang torches that somebody else lit long before you arrived: a library called PyTorch that records every operation you perform on a tensor, differentiates it for you, keeps your parameters in order, batches your data, and moves all of it to whatever hardware you have. Everything you built on Floors 2 and 3 exists here as a function call. That is a gift, and it is also the trap of this floor: a library that does the right thing 99% of the time trains you to stop looking. The 1% is where the weeks go. A softmax applied twice does not crash. A missing `zero_grad()` does not crash. A model evaluated with dropout still switched on does not crash. They just quietly learn less, and nobody tells you.

So the passage teaches two things at once. The API, precisely enough that you can write any training loop from memory. And the habit of knowing what the light is doing: which tensors are being tracked, which mode the model is in, where the randomness comes from, what a checkpoint must contain. The boss at the end is a creature that comes back different every time you train it. It feeds on people who skipped that second lesson.

**You will learn:** tensors and autograd · `nn.Module` and parameters · `state_dict` save/load · `Dataset` and `DataLoader` · the canonical training loop and its classic bugs · device-agnostic code · mixed precision: float16, bfloat16, autocast and loss scaling · determinism and checkpoint/resume.

**You need:** Python 3.11+, numpy, and PyTorch. The CPU build is small and enough for every floor:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`dungeon doctor` confirms it. No GPU is used or assumed anywhere in the dungeon.

---

## The lore of torch (read this before the rooms)

### A tensor is a numpy array that remembers three more things

`torch.Tensor` has everything an `ndarray` has (`shape`, `dtype`, indexing, broadcasting with the same four rules from Floor 0) plus a **device** it lives on, a flag saying whether autograd **requires_grad** for it, and, after a backward pass, a **`.grad`**.

```python
x = torch.tensor([1.0, 2.0, 3.0])            # float32: Python floats become float32
n = torch.tensor([1, 2, 3])                  # int64:  Python ints become int64
z = torch.zeros(2, 3, dtype=torch.float32)   # be explicit anyway
x.shape, x.dtype, x.device                   # torch.Size([3]), torch.float32, device(type='cpu')
x.item()                                     # only for a single element -> Python number
x.tolist()                                   # nested Python lists
```

Crossing the numpy border:

| Call | Copies? | dtype |
|---|---|---|
| `torch.tensor(arr)` | yes | numpy's dtype (float64 stays float64) |
| `torch.from_numpy(arr)`, `torch.as_tensor(arr)` | no, shares memory | numpy's dtype |
| `torch.as_tensor(arr, dtype=torch.float32)` | only if needed | what you asked for |
| `t.numpy()` | no, shares memory | refuses if `t` requires grad: `t.detach().numpy()` |

numpy defaults to float64 and torch models default to float32. `torch.from_numpy(np.random.rand(4, 64))` fed to an `nn.Linear` fails with *mat1 and mat2 must have the same dtype*. Fix the dtype at the border, once, and never think about it again.

### Autograd: the graph is built as you compute

Set `requires_grad=True` on a tensor you create (a **leaf**), and every operation on it records a node. The result carries a `grad_fn` pointing back at the operation. Call `.backward()` on a scalar and autograd walks the graph in reverse, depositing d(output)/d(leaf) into each leaf's `.grad`. That is the same reverse-mode algorithm you wrote on Floor 2, run by C++.

```python
w = torch.tensor(2.0, requires_grad=True)   # leaf
y = 3 * w**3 - 2 * w + 1                    # y.grad_fn = <AddBackward0>; y.requires_grad = True
y.backward()                                # walks the graph
w.grad                                      # tensor(34.) = 9*2^2 - 2
```

Five rules, each with a room that checks it:

1. **`.grad` accumulates.** A second `backward()` adds to whatever is there. This is deliberate (it is how gradient accumulation over micro-batches works) and it is why every training loop clears the gradients before each backward: `optimizer.zero_grad()` (which sets them to `None` by default) or `p.grad = None`.
2. **A leaf that requires grad refuses in-place operations.** `w.add_(1)` or `w += 1` raises *a leaf Variable that requires grad is being used in an in-place operation*. Inside `with torch.no_grad():` the same line is allowed: autograd is not watching, so there is no graph to corrupt. That is exactly what `optimizer.step()` does.
3. **`detach()` cuts; `clone()` copies.** `t.detach()` returns a tensor that shares storage with `t` but has no `grad_fn` and `requires_grad=False`. Writing into it writes into `t`. `t.clone()` allocates new memory but *stays in the graph*: a clone of a tracked tensor is tracked. For an independent, untracked copy: `t.detach().clone()`.
4. **Under `torch.no_grad()` nothing is recorded.** Results have `grad_fn is None`. Use it for evaluation, for metrics, for anything you will never backward through: a graph costs memory. The context manager restores the previous mode on exit; prefer it to `torch.set_grad_enabled(False)`.
5. **The graph is freed by `backward()`.** Intermediate values saved for the backward pass are released, so a second `backward()` through the same graph raises *Trying to backward through the graph a second time*. Rebuild the graph (recompute `y`) or pass `retain_graph=True` if you genuinely need two passes.

Two more things you will meet in the prophecy: `.numpy()` on a tracked tensor is a RuntimeError (numpy cannot carry a graph; detach first), and a tensor that has been detached acts as a *constant* in later computations: in `z = y.detach() * x`, `dz/dx` is the value of `y`, and no gradient flows into whatever produced `y`.

### nn.Module: a container with bookkeeping

`nn.Module` holds parameters and submodules and knows how to find them all. The rules:

- Call `super().__init__()` first. Then any `nn.Module` or `nn.Parameter` you assign to `self.name` is **registered**: it appears in `parameters()`, `named_parameters()` (as `"name.weight"`, nested with dots), `state_dict()`, and it follows the module through `.to(device)`, `.train()`, `.eval()`.
- A plain Python `list` of layers is **not** registered. Use `nn.ModuleList` (indexable, you write the forward) or `nn.Sequential` (calls them in order).
- You call the module, not `forward`: `model(x)` runs hooks and then `forward(x)`.
- `nn.Linear(in_f, out_f)` stores `weight` of shape `(out_f, in_f)` and `bias` of shape `(out_f,)`, and computes `x @ weight.T + bias`. Default init is a scaled uniform; for ReLU networks `nn.init.kaiming_normal_(w, nonlinearity="relu")` gives std `sqrt(2 / fan_in)`, the He initialisation you derived on Floor 3. `model.apply(fn)` calls `fn` on every submodule, which is how you re-initialise a whole network in four lines.
- Counting parameters: `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Freezing: `p.requires_grad_(False)`; frozen parameters get no `.grad` and the optimizer leaves them alone.

**state_dict.** `model.state_dict()` is an ordered dict of `name -> tensor` (parameters and buffers such as BatchNorm running statistics). `torch.save(obj, path_or_buffer)` pickles any object; `torch.load` reads it back, and since 2.6 defaults to `weights_only=True`, which accepts tensors, numbers, strings and plain containers and rejects arbitrary pickled classes. Save state_dicts, not modules: a pickled module stores the import path of your class and breaks the moment you rename it. `model.load_state_dict(sd)` is strict by default: every key must match in name and shape, or it raises. That strictness is a feature.

**Modes.** `model.train()` and `model.eval()` set a boolean `training` on every submodule. Only a few layers care, but they care a lot:

| Module | train mode | eval mode |
|---|---|---|
| `nn.Dropout(p)` | zeroes each activation with probability `p`, scales the survivors by `1/(1-p)` | identity |
| `nn.BatchNorm*` | normalises with batch statistics, updates running averages | normalises with the running averages |
| everything else | unaffected | unaffected |

`eval()` does **not** switch autograd off, and `no_grad()` does **not** change modes. Evaluation wants both. A fresh module starts in train mode.

### Losses: shapes and dtypes

The one that matters most on this floor: `nn.CrossEntropyLoss` (functional form `F.cross_entropy`) takes **raw logits** of shape `(N, C)` and **int64 class indices** of shape `(N,)`. It applies `log_softmax` itself. Feed it probabilities and the loss can never fall below about `log(1 + (C-1)/e)`, 1.46 for ten classes, no matter how right the model is. Feed it float labels and it raises *expected target dtype to be Long*.

| Loss | Input | Target |
|---|---|---|
| `CrossEntropyLoss` | `(N, C)` logits | `(N,)` int64 indices (or `(N, C)` float probabilities) |
| `NLLLoss` | `(N, C)` **log**-probabilities | `(N,)` int64 |
| `BCEWithLogitsLoss` | `(N, *)` logits | same shape, float in `[0, 1]` |
| `MSELoss`, `L1Loss` | `(N, *)` | **same shape**; a mismatch broadcasts with a warning and trains on nonsense |

The loot checklist has the full table.

### Dataset and DataLoader

A `Dataset` answers `__len__()` and `__getitem__(i)`. A `DataLoader` wraps it: it chooses indices (in order, or shuffled), fetches the examples and **collates** them, so a list of `(image, label)` pairs becomes `(stacked images, stacked labels)`. Python ints become 0-d tensors and then an int64 batch; tensors are stacked along a new leading axis.

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
for xb, yb in loader:          # xb: (32, 8, 8) float32   yb: (32,) int64
    ...
```

Three details the trials check:

- **The last batch is smaller** when `len(dataset)` is not a multiple of `batch_size` (`drop_last=False`). Count examples, not batches, when you average.
- **The shuffle draws from a `torch.Generator`.** Hand the loader its own, seeded, and the order is reproducible without depending on anything else the program does with the global RNG. Without one, the loader seeds itself from the global RNG at the start of each epoch, which is reproducible too, but only if nothing else consumed a random number in between. Either way the generator's state advances: epoch 2 walks a different order from epoch 1, and two loaders seeded identically walk identical orders.
- **`num_workers > 0`** forks worker processes. torch reseeds each worker's torch RNG; it does not reseed numpy, so numpy-based augmentation in `__getitem__` repeats across workers unless you add a `worker_init_fn`. This floor keeps `num_workers=0`.

### The canonical training loop

```python
model.train()
for xb, yb in loader:
    optimizer.zero_grad()              # 1. rule 1: .grad accumulates, so clear it
    logits = model(xb)                 # 2. forward, graph recorded
    loss = F.cross_entropy(logits, yb) # 3. raw logits, int64 labels
    loss.backward()                    # 4. gradients into every parameter's .grad
    optimizer.step()                   # 5. in-place update under no_grad (rule 2)
    running += loss.item()             # detach for logging; never sum the tensors
```

And its mirror:

```python
model.eval()
with torch.no_grad():
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        total += yb.shape[0]
model.train()                          # if training continues
```

Everything that can go wrong with these fourteen lines has a name and a symptom. Room 4.4 is seven of them; the loot checklist is seventeen. The ones that crash are cheap. The ones that do not (missing `zero_grad`, softmax before the loss, dropout left on during evaluation, `argmax` on the wrong axis when `B == C`, summing loss tensors) are the ones this floor exists for.

### Devices

Every tensor lives on a `torch.device`: `cpu`, `cuda:0`, `mps` (Apple silicon), or `meta` (shapes only, no data, exists on every build). Two tensors can only meet in an operation if they share a device. The rules of the ferry:

- `torch.device("cuda")` is a label; making one needs no GPU. `torch.cuda.is_available()` and `torch.backends.mps.is_available()` tell you what you actually have.
- `tensor.to(device)` returns a **new** tensor (or the same object if it is already there). `module.to(device)` moves the module **in place** and returns it. Build the model, move it, *then* create the optimizer and start training: optimizer state is created lazily on the parameters' device at the first step.
- A module has no `.device`. Ask its parameters: `next(model.parameters()).device`.
- Batches are nested structures (tuples, dicts of tensors). Write one recursive `to_device` and use it everywhere; the trial uses `meta` as the far bank so it can check the recursion on a CPU-only machine.

### Mixed precision: three floating-point formats

A float32 spends its 32 bits as 1 sign, 8 exponent and 23 mantissa bits. Two 16-bit formats exist, and the whole subject of mixed precision follows from how each one spends the bits it has left:

| | sign | exponent | mantissa | largest finite | `eps` (gap above 1.0) | smallest normal | significant digits |
|---|---|---|---|---|---|---|---|
| `torch.float32` | 1 | 8 | 23 | 3.40e38 | 2^-23 ≈ 1.19e-7 | 2^-126 ≈ 1.18e-38 | about 7 |
| `torch.float16` | 1 | 5 | 10 | **65504** | 2^-10 ≈ 9.77e-4 | 2^-14 ≈ 6.10e-5 | about 3 |
| `torch.bfloat16` | 1 | 8 | 7 | 3.39e38 | 2^-7 ≈ 7.81e-3 | 2^-126 ≈ 1.18e-38 | 2 to 3 |

Every number in the table is `torch.finfo(dtype)`: `.bits`, `.max`, `.eps`, `.tiny`. The two widths finfo does not report tie the columns together: `eps = 2 ** -mantissa_bits` and `tiny = 2 ** (2 - 2 ** (exponent_bits - 1))`. Below `tiny` a format still has *subnormal* numbers (float16 reaches down to 2^-24 ≈ 6e-8), and below those, zero.

**float16 keeps precision and gives up range.** Eleven significant bits, about three decimal digits (2049 rounds to 2048), but anything above 65504 is `inf` and anything below about 6e-8 is `0`. **bfloat16 keeps range and gives up precision.** The same 8 exponent bits as float32, so 70000 and 1e-8 are both fine, but only eight significant bits: 257 rounds to 256, and `1.0 + 0.001 == 1.0`. Both take 2 bytes per element instead of 4, which is the point of the exercise: half the memory traffic, and on tensor-core hardware several times the matmul throughput.

**Why float16 training needs loss scaling.** Gradients are small numbers. Activation gradients of 1e-6 to 1e-8 are ordinary in a network that is nearly trained, and every one of them below 6e-8 becomes exactly zero in float16; between 6e-8 and 6.1e-5 they are subnormal and lose digits. Meanwhile attention logits, sums of squares and variances are large numbers, and anything past 65504 becomes `inf`, which becomes `nan` one op later. Loss scaling attacks the underflow: multiply the loss by a scale S before `backward()`. The chain rule is linear, so every gradient in the network is multiplied by S too, and 1e-8 × 2^16 = 6.6e-4 is a perfectly ordinary float16 number. Divide the gradients by S before the optimizer step; they land in float32 `.grad`s, which hold 1e-8 without complaint. The overflow half is handled by making S *dynamic*: start large (2^16), and whenever any gradient comes back `inf` or `nan`, skip the step and halve S; after 2000 consecutive clean steps, double it. S settles just below the largest value the network can bear, which is exactly where you want it: as large as possible so nothing underflows, no larger so nothing overflows. That is `torch.amp.GradScaler`, and Room 4.6 has you write it by hand.

**Why bfloat16 mostly does not.** It has float32's exponent, so 1e-8 does not underflow and 70000 does not overflow; the values that break float16 are all in range. bfloat16 training normally runs without a scaler, which is one reason it has become the default for large models on hardware that supports it. Its cost is the missing precision, and that surfaces in two places: weight updates and reductions.

**Master weights stay in float32.** One SGD or Adam step changes a weight by `lr * g`, typically 1e-4 to 1e-6 of the weight's own size. The gap between neighbouring bfloat16 numbers is 2^-7, about 0.8% of the value; in float16 it is 0.1%. Add 1e-4 to a bfloat16 weight of 1.0 and you get 1.0 back: the update is rounded away, and the model stops learning while reporting no error. So the *stored* weights, and the optimizer's moments, stay float32; the 16-bit copies exist only for the arithmetic. "Mixed" precision, not half. Casting the weights themselves (`model.to(torch.bfloat16)`) is for inference, where nothing is updated and the halved memory is the whole prize.

**What autocast does.** `torch.autocast(device_type=..., dtype=...)` is a context manager that intercepts each op and applies a policy from three allow-lists:

- *Low precision:* the matmul family (`mm`, `addmm`, `bmm`, `matmul`, `linear`, `einsum`, the convolutions, `scaled_dot_product_attention`, the RNN cells). Inputs are cast to the autocast dtype on the way in; outputs come out in it. This is where the time goes, so this is where the speed comes from.
- *float32:* ops that are fragile in 16 bits. On CUDA that list holds the loss functions (`cross_entropy`, `nll_loss`, `mse_loss`, `binary_cross_entropy_with_logits`, ...), `softmax`, `log_softmax`, `layer_norm`, `group_norm`, `sum`, `prod`, `cumsum`, `exp`, `log`, `pow` and the norms. On CPU the float32 list is shorter: the losses, linear algebra and a few pooling ops, but *not* `layer_norm`, `softmax` or `sum`, which run in their inputs' dtype. Room 4.6 asserts only what the CPU build does (Linear out in bfloat16, `cross_entropy` out in float32); on CUDA expect more float32 than you see here.
- *Promote:* a few many-input ops (`addcmul`, `dot`, `tensordot`, and on CPU `cat` and `stack`) run in the widest dtype among their inputs.

Everything else runs in whatever dtype its inputs already have. Parameters are never modified: an `nn.Linear` keeps float32 weights, autocast casts a bfloat16 copy on entry (and caches it for the duration of the block), and the parameter's `.grad` is float32. The pattern:

```python
scaler = torch.amp.GradScaler("cuda")           # float16 only; bfloat16 needs no scaler
for xb, yb in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(xb)                      # matmuls in float16
        loss = F.cross_entropy(logits, yb)      # the loss in float32 (allow-list)
    scaler.scale(loss).backward()               # backward OUTSIDE the block; autograd replays the recorded dtypes
    scaler.step(optimizer)                      # unscale, look for inf, then step or skip
    scaler.update()                             # grow or back off
```

With bfloat16, drop the scaler: `loss.backward(); optimizer.step()`. On CPU write `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`, which is what this floor's trial does.

**Memory arithmetic.** Count bytes per parameter. Float32 weights are 4; their gradients another 4; Adam keeps two float32 moments, 8 more: **16 bytes per parameter** for float32 training with Adam, before a single activation. Mixed precision does not shrink that ledger: the float32 master weights, gradients and moments are still 16, and a framework that stores the 16-bit weights and gradients permanently counts 2 + 2 + 4 + 4 + 4, the same 16. What halves in training is the activations, and what you gain is speed. Inference is a different sum: only the weights, 2 bytes each in bfloat16, so a 7-billion-parameter model is 14 GB rather than 28. `parameter_bytes` in Room 4.6 is that sum, on a toy.

**A note on TF32.** Ampere and later NVIDIA GPUs can run float32 matmuls as *TensorFloat-32*: the inputs are rounded to 10 mantissa bits (float16's precision, float32's exponent) and the products accumulated in float32. It is not a dtype you can store a tensor in; it is a mode of the matmul kernel. `torch.set_float32_matmul_precision("high")` turns it on for matmuls (`"highest"` is the default, and what this CPU build reports), and cuDNN convolutions use it by default (`torch.backends.cudnn.allow_tf32`). Results then differ from true float32 at roughly the 1e-3 level, so if a "float32" GPU run refuses to match a CPU reference, this is the first suspect.

### Determinism and checkpoints

The sources of randomness in a training run, in the order you meet them: weight initialisation (torch global RNG), the shuffle (the loader's generator, or the global RNG if you gave it none), dropout masks (torch global RNG), anything your own code does with `random` or `numpy`, and, on GPUs, kernels that are non-deterministic by design.

- `torch.manual_seed(s)` seeds the CPU generator and every CUDA device. `random.seed(s)` and `np.random.seed(s)` cover the other two libraries. Do all three, once, **before building the model**.
- `torch.use_deterministic_algorithms(True)` makes torch raise instead of silently using a non-deterministic kernel. Fine on CPU; on CUDA some ops need `CUBLAS_WORKSPACE_CONFIG=:4096:8` and some have no deterministic version at all.
- On CPU, results are bitwise reproducible for a **fixed thread count**. Change `torch.get_num_threads()` and the order in which partial sums are combined changes, and so do the last bits. The trials on this floor run single-threaded (tiny models are faster that way anyway).

A checkpoint that can **resume exactly** holds five things: `model.state_dict()`, `optimizer.state_dict()` (momentum buffers, Adam moments, step counts), the epoch, `torch.get_rng_state()` (so dropout masks continue where they stopped), and each loader generator's `get_state()` (so the shuffle continues where it stopped). Miss any one and the resumed run drifts from the uninterrupted one: weights obviously, momentum slightly, the RNG states from the next batch onwards. All of those are tensors or numbers, so the file loads under `weights_only=True`; numpy's RNG state (a tuple containing a string and an array) does not, which is one more reason to draw from torch generators. The boss verifies exact resume on CPU, and it does hold exactly: the reference solution matches an uninterrupted three-epoch run with `torch.equal` on every tensor.

### Writing your own backward (the secret room)

When autograd does not know an operation, or you can do its gradient faster, subclass `torch.autograd.Function` with two static methods. `forward(ctx, *inputs)` computes the output and stashes whatever backward needs via `ctx.save_for_backward(...)`. `backward(ctx, grad_output)` returns one gradient per input of `forward`, in order, `None` for inputs that need none. You call it with `MyFunction.apply(...)`, never directly. `torch.autograd.gradcheck` compares your backward against finite differences in float64; every custom Function should pass it before it is trusted.

---

## Rooms

Run `dungeon enter 4` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial. Room 4.4 is the exception: it ships complete, cursed code, and you fix it.

### 4.1 The First Torch — `rooms/room_1_first_torch.py`

The first bracket on the wall. Six small functions, one per rule: differentiate a polynomial with `backward()` (the trial reads your source and insists you call it), backward twice with and without clearing, produce a detached shadow and an independent copy, evaluate a function with autograd off, and write an SGD step in place on a leaf.

Then **the Prophecy**: six snippets given as strings. For each, predict the value of `result` or write `"RuntimeError"`. Commit before running. An in-place op on a leaf, a second backward, `.numpy()` on a tracked tensor, an update under `no_grad`, accumulation, and a `detach()` in the middle of a chain.

```
dungeon trial 4 room_1
```

### 4.2 The Module Forge — `rooms/room_2_module_forge.py`

Pour the Floor 3 MLP into the `nn.Module` mould: `MLP(sizes, activation)` with an `nn.ModuleList` called `layers` and an activation called `act`, flattening `(B, 8, 8)` to `(B, 64)` and returning logits. Then the tools around it: `count_parameters` (trainable only), `init_weights` (Kaiming weights, zero biases, via `apply`), `manual_forward` (the same computation from raw matmuls; during the trial `nn.Linear.forward` and `F.linear` are cold iron and raise), `serialize`/`deserialize_into` (a state_dict round trip through `io.BytesIO`, strict on load), and `freeze(model, prefix)`.

```
dungeon trial 4 room_2
```

### 4.3 The Loader's Lantern — `rooms/room_3_loaders_lantern.py`

A `RuneDataset` over the numpy runes, returning `float32 (8, 8)` images and `int64` labels (and curing float64 at the border). A `make_loader` whose shuffle is reproducible from a seed through a `torch.Generator`. Then the two halves of every training script: `train_one_epoch` (the five lines, in train mode, returning a float) and `evaluate` (eval mode, no graph, counting examples not batches). The trial replays shuffles, weighs the last partial batch, spies on the mode flag and autograd state from inside a model, and trains two epochs on the runes.

```
dungeon trial 4 room_3
```

### 4.4 The Cursed Training Loop — `rooms/room_4_cursed_training_loop.py`

No stubs. A complete training loop with exactly seven planted bugs: one wrong dtype, one wrong axis, one wrong scale, one thing applied twice, one memory leak, one thing never called, one mode never switched. The trial names a symptom per curse, with the observed value, and finally trains the whole loop: the fixed version reaches at least 90% validation accuracy on the runes in three epochs and a couple of seconds. Lift each curse with a one-line fix; do not rewrite the file.

```
dungeon trial 4 room_4
```

### 4.5 The Device Ferry — `rooms/room_5_device_ferry.py`

`pick_device(prefer)` (an override wins; otherwise `cuda`, then `mps`, then `cpu`), `to_device(obj, device)` recursing through lists, tuples and dicts while leaving other objects untouched, `ensure_float32(batch)` for the numpy float64 trap (integers stay integers), and `model_device(model)` read from parameters, then buffers, else `ValueError`. Everything runs on CPU; the trial uses the `meta` device as a far bank that exists on every machine.

```
dungeon trial 4 room_5
```

### 4.6 The Half-Light — `rooms/room_6_the_half_light.py`

The torches thin out. `dtype_report(dtype)` reads `bits`, `max`, `eps` and `tiny` off `torch.finfo` and supplies the two widths finfo does not know; the trial checks that the columns agree with each other (`eps` really is 2^-mantissa, `tiny` really is 2^(2-2^(e-1))). Then the **Half-Light Prophecy**: eight snippets, predict `"inf"` or `"finite"`, `"zero"` or `"nonzero"`, `True` or `False`, from the table alone. `overflow_demo` sums squares naively in float16 and is supposed to return `inf`; `safe_sum_of_squares` casts first and does not. `forward_autocast_bf16` runs a model under `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` while a spy inside the model checks the mode and a second model reports its loss dtype. `cast_for_inference` and `parameter_bytes` halve a model's weight for inference and prove it. Finally `LossScaler`: `scale`, `unscale_` and `step`, driven by the trial with a clean gradient, an injected `inf`, and a schedule it must reproduce exactly, then with a real 1e-8 gradient that only survives the float16 hop when scaled.

```
dungeon trial 4 room_6
```

---

## Boss: The Reproducibility Revenant

```
                    .-~~~~-.
                   /  _  _  \        "You trained me yesterday. You wrote
                  |  (o)(o)  |        down the accuracy. Train me again.
                  |    __    |        Go on. I will be someone else."
                   \  '--'  /
                 .-'`-....-'`-.       It comes back DIFFERENT every time.
                /   |  ||  |   \      Weight init. Shuffle order. Dropout
               /    |  ||  |    \     masks. Every die in the building rolls
              /_____|__||__|_____\    without you, and it counts on that.
```

**Weakness:** someone who seeds every source of randomness and checkpoints every piece of state, so that a run can be replayed, or interrupted and resumed, to the bit.

- **Phase 1:** `seed_everything(seed, deterministic=False)`: Python, numpy, torch, and the deterministic-algorithms flag set either way so it never leaks.
- **Phase 2:** `build_run(seed)` (a model with `Dropout(0.1)`, a `TensorDataset` of the runes, a loader with its own generator, SGD with momentum), `run_epoch`, and `train_deterministically(seed, epochs)`. Two calls, one seed: `torch.equal` on every tensor and identical loss lists.
- **Phase 3:** `save_checkpoint`/`load_checkpoint` (model, optimizer, epoch, RNG states; loads under `weights_only=True`) and `train_with_resume(seed, epochs=3, stop_after=2, path)`: train two epochs, checkpoint, rebuild everything from scratch, restore all four pieces of state, finish the third epoch, and match a straight three-epoch run exactly.

```
dungeon fight 4
```

## Secret room: The Custom Whisper *(optional)*

Behind the crypt wall: a `torch.autograd.Function` for the soft threshold `sign(x) * max(|x| - lam, 0)` with a hand-written backward (`1` outside the band, `0` inside), the same op composed from ordinary tensor operations as a witness, and `gradcheck` in float64 at points safely away from the kinks. Not required to clear the floor; required to really believe you understand what `backward()` has been doing for you all along.

```
dungeon trial 4 --secret
```

## Loot

Clear the six rooms and lay the Revenant to rest to unlock:

- **The PyTorch Debugging Checklist** — `loot/pytorch_debugging_checklist.md`. Sixty-second triage, the seven curses and ten more classic bugs with symptoms and fixes, the shape/dtype table for the common losses, what `train()`/`eval()` actually change, when to detach, a mixed-precision page (the three formats, the autocast recipe, the symptoms of overflow and underflow), and the reproducibility checklist.
- **Training Loop Template (torch)** — `loot/training_loop_template_torch.py`. A clean, device-agnostic, fully seeded loop with checkpointing and exact resume. Runs as a script on the runes; copy it into every project and delete what you do not need.

## Stuck?

- `dungeon hint 4 room_4` reveals one hint at a time (three per room). The boss and the secret room have hints too.
- The trial failure messages say *what* is wrong and what they observed, not just *that* something failed. Read them; several were written to be more useful than the torch error they replace.
- `solutions/` exists. Use it after an honest attempt, to compare rather than to copy. The solution to 4.4 lists the seven curses in its docstring.

When `dungeon map` shows the Torchlit Passage cleared, take the stairs. Below, someone is cutting words into pieces.
