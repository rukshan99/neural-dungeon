`loss.backward()` does not overwrite `.grad`; it adds to it. That is the whole trick: run several micro-batches without calling `zero_grad` and the gradients pile up into a sum.
---
A sum of k micro-batch gradients is k times the mean you want. Scale before you accumulate: `(loss / accum_steps).backward()`. Then `if (i + 1) % accum_steps == 0: optimizer.step(); optimizer.zero_grad(set_to_none=True)`.
---
Whole thing: check `len(batches) % accum_steps == 0` (else `ValueError`); `model.train()`; `optimizer.zero_grad(set_to_none=True)`; `for i, (x, y) in enumerate(batches): _, loss = model(x, y); (loss / accum_steps).backward(); total += loss.item(); if (i + 1) % accum_steps == 0: optimizer.step(); optimizer.zero_grad(set_to_none=True)`; return `total / len(batches)`. `count_tokens_seen` is `sum(int(y.numel()) for _, y in batches)`.
