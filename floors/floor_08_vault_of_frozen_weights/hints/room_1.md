`load_pretrained()` returns three things: `(model, tokenizer, extra)`. Keep the first two. Everything else in this room walks `model.named_parameters()`, which yields `(name, parameter)` pairs with dotted names like `blocks.2.mlp.fc.weight`, and yields the tied `lm_head`/`wte` matrix only once.
---
Freezing is `param.requires_grad_(False)` (or `param.requires_grad = False`) on each matching parameter; a pattern matches when `pattern in name`. `count_parameters` is `sum(p.numel() for p in ...)` with an `if p.requires_grad` filter when `trainable_only`. `trainable_parameters` is the same filter returning the `(name, p)` pairs.
---
`snapshot`: `{n: p.detach().clone() for n, p in model.named_parameters()}`. `changed_parameters`: for each `(name, p)` in the live model, look the name up in the snapshot; report it if it is missing, if the shapes differ, or if `not torch.all((p.detach() - before).abs() <= atol)`. Then append any snapshot names the model no longer has.
