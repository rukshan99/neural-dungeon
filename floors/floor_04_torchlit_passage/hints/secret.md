Both methods are `@staticmethod` and take `ctx` first. In `forward`, compute `mask = x.abs() > lam`, call `ctx.save_for_backward(mask)`, and return `torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)`. `lam` is a Python float, so nothing about it is saved or differentiated.
---
In `backward`, `(mask,) = ctx.saved_tensors` (it is always a tuple, even of one). The gradient of `sign(x) * (|x| - lam)` with respect to `x` is 1 wherever `|x| > lam` and 0 inside the band, so `grad_x = grad_output * mask.to(grad_output.dtype)`. Return `grad_x, None`: one entry per input of `forward`, `None` for `lam`.
---
`soft_threshold(x, lam)` is `SoftThreshold.apply(x, lam)`; never call `forward` directly, or autograd never hears about it. `soft_threshold_composed` is the same one-liner as forward without the ctx. gradcheck wants float64 inputs with `requires_grad=True`; the trial builds them for you, away from the kinks at `|x| == lam` where the function has no derivative.
