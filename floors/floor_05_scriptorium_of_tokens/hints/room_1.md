In `__init__`, the whole vocabulary is one list: `self.vocab = SPECIAL_TOKENS + list(chars)`. Then `stoi` and `itos` both come from `enumerate(self.vocab)`, and the four ids are the constants at the top of the file. `encode` is a comprehension with `self.stoi.get(ch, UNK_ID)`; `decode` skips any `int(i) < 4` when `skip_special` is set.
---
`batch_encode` is four small steps: encode every text (with specials); if `max_len` is set, shorten any long sequence to `s[:max_len - 1] + [EOS_ID]`; pick the target length (`max(len(s) for s in seqs)` or `max_len`); then right-pad each with `[PAD_ID] * (target - len(s))` and build the mask as `[1] * len(s) + [0] * (target - len(s))`. Check `pad_to_longest=False and max_len is None` first and raise `ValueError`.
---
```python
seqs = [self.encode(t) for t in texts]
if max_len is not None:
    seqs = [s if len(s) <= max_len else s[:max_len - 1] + [EOS_ID] for s in seqs]
if not seqs:
    return {"input_ids": [], "attention_mask": []}
target = max(len(s) for s in seqs) if pad_to_longest else max_len
ids = [s + [PAD_ID] * (target - len(s)) for s in seqs]
mask = [[1] * len(s) + [0] * (target - len(s)) for s in seqs]
```
`to_tensors`: `import torch` inside the function, then `{k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}`.
