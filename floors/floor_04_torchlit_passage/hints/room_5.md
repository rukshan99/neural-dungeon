`pick_device`: if `prefer` is not None, `return torch.device(prefer)`. Otherwise check `torch.cuda.is_available()`, then `torch.backends.mps.is_available()`, and fall through to `torch.device("cpu")`. Constructing a `torch.device("cuda")` never needs a GPU; only moving a tensor there does.
---
`to_device` is a recursive function with four branches: `torch.Tensor` -> `obj.to(device)`; `list` -> a list of recursive calls; `tuple` -> `tuple(...)` of recursive calls; `dict` -> `{k: recurse(v)}`; anything else -> `return obj`. `ensure_float32` is the same shape with `np.ndarray` -> `torch.as_tensor(arr)` first, and the tensor branch returning `t.float()` only `if t.is_floating_point()`.
---
`model_device`: `for p in model.parameters(): return p.device`, then the same over `model.buffers()`, then `raise ValueError(...)`. A generator that yields nothing skips the loop body, which is exactly how you fall through to the next source.
