# The PyTorch Debugging Checklist

*Loot from Floor 4. When a training loop runs and learns nothing, start here.*

## Sixty-second triage

1. Does the loss go **down** on a *single batch* repeated 50 times? If not, the bug is in the loop, not the data.
2. Print `logits.shape`, `y.shape`, `y.dtype`, and the loss value before the first `backward()`.
3. Is `optimizer.zero_grad()` there? Is `optimizer.step()` there? In that order around `backward()`?
4. `model.train()` before training, `model.eval()` plus `torch.no_grad()` before measuring.
5. Halve the learning rate. Then halve it again. If the loss stops exploding, that was it.
6. Seed everything and run twice. If the two runs differ, you cannot debug anything else yet.

## The seven curses of Room 4.4

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Loss gets *worse* every step, or becomes `nan`/`inf` | `lr` far too large (the loop shipped with `10.0`) | `0.01`–`0.1` for SGD with momentum; `1e-3` for Adam as a first guess |
| 2 | `RuntimeError: expected target dtype to be Long` | labels cast to `float32` | class indices are `int64`: `y.long()` / `torch.as_tensor(y, dtype=torch.int64)` |
| 3 | Loss plateaus around 1.4–2.3 and never goes lower, even when accuracy is fine | `softmax` applied before `CrossEntropyLoss` (which applies `log_softmax` itself) | feed raw logits to the loss |
| 4 | Shape error comparing predictions with labels, or accuracy that is nonsense only when `B == C` | `argmax(dim=0)` on `(B, C)` logits | `argmax(dim=1)`: one winner per example |
| 5 | Loss falls for a few steps then climbs; gradients grow every batch | no `optimizer.zero_grad()` | call it before every `backward()` (`set_to_none=True` is the default and the cheapest) |
| 6 | Memory grows every batch; the epoch loss is a tensor with a `grad_fn` | `total += loss` keeps every batch's graph alive | `total += loss.item()` (or `.detach()`) |
| 7 | Validation accuracy noisy and low; the same input gives different outputs | never `model.eval()` (dropout active while measuring) or never `model.train()` again afterwards | `model.train()` at the top of the training epoch, `model.eval()` at the top of evaluation |

## Ten more classic bugs

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: cannot assign module before Module.__init__() call` | forgot `super().__init__()` | first line of every `__init__` |
| `ValueError: optimizer got an empty parameter list` | layers stored in a plain Python `list` | `nn.ModuleList` / `nn.Sequential`; check `sum(p.numel() for p in model.parameters())` |
| `RuntimeError: Expected all tensors to be on the same device` | batch on CPU, model on GPU (or the reverse) | move *every* tensor in the batch: recurse through the containers (Room 4.5) |
| `mat1 and mat2 must have the same dtype` (`Double` vs `Float`) | `torch.from_numpy` on a float64 numpy array | `.float()` at the border, once; `torch.as_tensor(x, dtype=torch.float32)` |
| `UserWarning: Using a target size that is different to the input size` and a loss that will not fall | `(N,)` predictions against `(N, 1)` targets in `MSELoss`/`L1Loss`: broadcast to `(N, N)`, averaged, wrong (`BCEWithLogitsLoss` at least raises a `ValueError` for the same mistake) | make the shapes identical: `pred.squeeze(1)` or `y.unsqueeze(1)` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | `x += ...`, `relu(inplace=True)`, or `x[mask] = 0` on a tensor autograd still needs | use the out-of-place version; `x = x + ...` |
| `view size is not compatible with input tensor's size and stride` | `.view()` on a non-contiguous tensor (after `transpose`, slicing) | `.reshape()` or `.contiguous().view()` |
| Every data-loader worker produces the *same* random augmentation | `num_workers > 0` and numpy randomness in `__getitem__`; torch reseeds each worker, numpy is not reseeded | `worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32)` or draw from a `torch.Generator` |
| `RuntimeError: Attempting to deserialize object on a CUDA device` when loading on a CPU box | checkpoint tensors remember their device | `torch.load(path, map_location="cpu")` |
| Loading a checkpoint fails after a refactor (`ModuleNotFoundError`, `AttributeError`) | `torch.save(model)` pickled the *class path* | save `model.state_dict()`, rebuild the module, `load_state_dict` |

Also worth knowing:

- `torch.load` defaults to `weights_only=True` since 2.6: tensors, numbers, strings and plain containers load; arbitrary pickled objects (numpy RNG state, dataclasses) do not. Keep checkpoints to tensors and primitives.
- `lr_scheduler.step()` goes *after* `optimizer.step()`, once per epoch (or per step, depending on the scheduler).
- Gradient clipping goes *between* `backward()` and `step()`: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
- Optimizer state (momentum, Adam moments) is created lazily on the parameters' device at the first `step()`. Move the model *before* training starts, not after.
- `x.max(dim=1)` returns a `(values, indices)` pair; `x.argmax(dim=1)` returns only the indices.

## Shapes and dtypes for the common losses

`N` examples, `C` classes. "logits" means raw, unsquashed outputs.

| Loss | Input | Target | Notes |
|---|---|---|---|
| `nn.CrossEntropyLoss` | `(N, C)` float logits | `(N,)` **int64** class indices in `[0, C)` | applies `log_softmax` itself; also accepts `(N, C)` float probabilities as soft targets; `ignore_index=-100`; `label_smoothing=` |
| `nn.NLLLoss` | `(N, C)` **log**-probabilities | `(N,)` int64 | `CrossEntropyLoss == NLLLoss(log_softmax(x))` |
| `nn.BCEWithLogitsLoss` | `(N, *)` float logits | same shape, float in `[0, 1]` | applies sigmoid itself, stably; `pos_weight=` for imbalance |
| `nn.BCELoss` | `(N, *)` probabilities in `[0, 1]` | same shape, float | needs a sigmoid first; prefer `BCEWithLogitsLoss` |
| `nn.MSELoss`, `nn.L1Loss`, `nn.SmoothL1Loss`, `nn.HuberLoss` | `(N, *)` float | **same shape**, float | mismatched shapes broadcast with a warning: fix them |
| `nn.KLDivLoss` | `(N, *)` log-probabilities | probabilities (or log-probs with `log_target=True`) | use `reduction="batchmean"` |

For sequences (Floor 7) `CrossEntropyLoss` takes `(N, C, T)` logits with `(N, T)` targets, or you flatten to `(N*T, C)` and `(N*T,)`.

## What `train()` and `eval()` actually change

| Module | `model.train()` | `model.eval()` |
|---|---|---|
| `nn.Dropout(p)` | zeroes each activation with probability `p`, scales the rest by `1/(1-p)` | identity |
| `nn.BatchNorm*` | normalises with the *batch* statistics and updates running mean/var | normalises with the *running* statistics |
| `nn.Linear`, `nn.ReLU`, `nn.LayerNorm`, `nn.Embedding`, ... | unaffected | unaffected |

The mode is a flag on every submodule (`module.training`); `.train()`/`.eval()` set it recursively. **`eval()` does not stop autograd and `no_grad()` does not change modes.** Evaluation needs both:

```python
model.eval()
with torch.no_grad():          # or torch.inference_mode(), which is stricter and faster
    logits = model(x)
model.train()                  # if training continues afterwards
```

## When to `detach()`

Detach (or `.item()`, which detaches and converts) whenever a tensor leaves the training computation:

- **Logging and metrics**: `running_loss += loss.item()`; `correct += (pred == y).sum().item()`.
- **Storing history**: a list of losses, a confusion matrix, anything that lives longer than one step.
- **Converting**: `.numpy()` refuses tracked tensors; write `t.detach().cpu().numpy()`.
- **Targets computed by a model**: teacher outputs in distillation, bootstrapped targets, pseudo-labels. You do not want gradients flowing into the target.
- **Stop-gradient tricks**: `x + (q - x).detach()` passes `q` forward but `x`'s gradient backward (the straight-through estimator).

Do **not** detach the loss before `backward()`, nor activations you still need gradients through. And remember: `detach()` shares memory with the original; writing into it writes into the original. `detach().clone()` if you intend to modify.

## The reproducibility checklist (from the boss)

1. `random.seed(s)`, `np.random.seed(s)`, `torch.manual_seed(s)`, once, before building the model.
2. Give every `DataLoader` its own `torch.Generator().manual_seed(s)`.
3. `torch.use_deterministic_algorithms(True)` when you need CUDA kernels to stop being clever (it can raise for ops with no deterministic implementation; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA).
4. Bitwise identity on CPU also assumes the same thread count (`torch.get_num_threads()`): reduction order changes the last bits.
5. A checkpoint that can resume exactly holds `model.state_dict()`, `optimizer.state_dict()`, the epoch, `torch.get_rng_state()` and every loader generator's `get_state()` (plus `torch.cuda.get_rng_state_all()` and scheduler state when you have them).
