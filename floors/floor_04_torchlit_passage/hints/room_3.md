A `Dataset` is two methods: `__len__` and `__getitem__`. Convert the numpy arrays once in `__init__` with `torch.as_tensor(X, dtype=torch.float32)` and `torch.as_tensor(y, dtype=torch.int64)`; then `__getitem__(i)` is `return self.X[i], self.y[i]`. The loader stacks the items for you.
---
`make_loader`: `g = torch.Generator().manual_seed(seed)` and `DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)`. Nothing else. The generator is what makes two loaders with the same seed walk the same order and lets epoch 2 differ from epoch 1.
---
`train_one_epoch`: `model.train()`, then for each batch `optimizer.zero_grad()`, `loss = loss_fn(model(xb), yb)`, `loss.backward()`, `optimizer.step()`, `total += loss.item()`; return `total / len(loader)`. `evaluate`: `model.eval()`, `with torch.no_grad():`, `preds = model(xb).argmax(dim=1)`, add `(preds == yb).sum().item()` to `correct` and `len(yb)` to `total`; return `correct / total`.
