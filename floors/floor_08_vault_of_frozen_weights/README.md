# Floor 8 — The Vault of Frozen Weights

> *A finished model sleeps here under ice. Teach it a new tongue without waking what it already knows.*

```
            ▼ stairs down from Floor 7
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │  8.1 THE VAULT  │─────│  8.2 THE LOW-RANK│─────│  8.3 THE ADAPTATION  │
   │      DOOR       │     │      SIGIL       │     │                      │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐                │
   │ ☠ THE FORGETTER │─────│  8.4 THE         │────────────────┘
   │   ANTECHAMBER   │     │     FORGETTING   │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a cold draught: the Replay Well
            ▼  stairs down to Floor 9
```

On Floor 7 you built a transformer and watched it learn. This floor hands you one that is already finished: the **Chronicler**, a 4-layer, 818,944-weight character-level GPT that reads the dungeon's chronicles at a loss of 0.23. It is kept in a vault, frozen, because it took real compute to make and because every adventurer who "just tweaks it a little" leaves it worse.

Your job is the job of almost every applied ML engineer today. You did not train the base model. You need it to do something new: read the goblin quartermaster's ledger, a text of pipes, colons and prices that the Chronicler currently scores at a loss of 6.8, which is to say it cannot read it at all. You have a small corpus, a CPU, and a few minutes. And you must not break what already works.

That last constraint is the whole floor. Fine-tuning is easy. Fine-tuning *without forgetting* is engineering. The boss down here is what happens when you skip it.

**You will learn:** checkpoints and `requires_grad` · freezing · LoRA from scratch, its arithmetic and its merge · the fine-tuning loop · measuring catastrophic forgetting · the three mitigations (parameter-efficient tuning, lower learning rate, replay) · adapter save/load/switch/merge · elastic weight consolidation (secret).

**You need:** PyTorch (CPU is plenty). `dungeon.artifacts.tiny_gpt` provides `load_pretrained()`, the `GPT` class and `read_corpus("chronicles" | "ledger")`. Every trial on this floor runs in well under a minute; most in a few seconds.

---

## The lore of frozen weights (read this before the rooms)

### A checkpoint is a dict of tensors plus the recipe to rebuild their owner

`model.state_dict()` is an ordered `{name: tensor}` of every parameter and every persistent buffer, keyed by dotted module path: `blocks.2.attn.c_attn.weight`, `ln_f.bias`. It is *only* the numbers. To use them you must rebuild the same module tree, which means you also need the config (`vocab_size=72, n_layer=4, n_head=4, n_embd=128, block_size=128`) and, for a language model, the tokenizer that maps characters to the ids the embedding table expects. The Chronicler's checkpoint stores all three:

```python
payload = torch.load("chronicler.pt", map_location="cpu")
cfg = GPTConfig(**payload["config"])
model = GPT(cfg)
model.load_state_dict(payload["state_dict"], strict=True)   # every key must match, both ways
tokenizer = CharTokenizer(payload["chars"])
```

`strict=True` is your friend: it refuses a state dict with missing or unexpected keys, which is how you find out that the architecture you rebuilt is not the one that was saved. You will exploit exactly this in the boss's Phase 3: a merged LoRA model must load into a fresh plain `GPT` with `strict=True`, proving nothing extra is left behind.

Two details worth knowing. Tied weights (`lm_head.weight is wte.weight`) appear under both names in a `state_dict()` but only once in `named_parameters()`. And buffers marked `persistent=False` (the Chronicler's causal mask) are not saved at all; they are rebuilt by `__init__`.

### `requires_grad` is the switch; freezing is flipping it

Every tensor carries a `requires_grad` flag. Autograd records an operation only if at least one input requires grad; `backward()` writes `.grad` only into leaf tensors that require it. For an `nn.Parameter` the flag defaults to True. Set it to False and three things follow:

1. no `.grad` is ever written to it (it stays `None`);
2. an optimizer that was never handed it cannot move it;
3. if *every* parameter feeding the loss is frozen, `loss.requires_grad` is False and `backward()` raises.

```python
for name, p in model.named_parameters():
    if "blocks.0." in name:
        p.requires_grad_(False)

params = [p for p in model.parameters() if p.requires_grad]   # hand the optimizer ONLY these
opt = torch.optim.AdamW(params, lr=lr)
```

The filter on the last line matters. AdamW keeps two extra tensors (first and second moment) per parameter it is given, so handing it frozen parameters wastes memory. And with `weight_decay > 0`, AdamW would shrink frozen weights even though their gradient is None (decoupled decay does not look at the gradient). Freeze, then filter.

Freezing is cheap insurance, not a guarantee. This floor makes you *prove* what moved: take a `snapshot()` of every tensor before training and diff afterwards with `changed_parameters()`. Trust nothing you have not diffed.

### LoRA: a trainable correction of low rank

A fine-tune changes a weight matrix `W` (shape out × in) to `W + ΔW`. LoRA (Hu et al., 2021) observes that useful `ΔW`s have low effective rank and simply *constrains* it:

```
ΔW = (alpha / r) · B A        A: (r × in)   B: (out × r)   r << min(in, out)

y = W x + b + (alpha / r) · B (A x)
```

`W` and `b` stay frozen. Only `A` and `B` train. They hold `r · (in + out)` numbers instead of `in · out`.

**Initialisation is the whole trick.** `A` is random (Kaiming-uniform, or normal with small std); **`B` is zero**. So at step 0, `B A = 0` and the wrapped layer computes *exactly* the pretrained function. The fine-tune starts from the checkpoint, not from a perturbed copy of it. If both were zero, both gradients would be zero (∂/∂A ∝ Bᵀ, ∂/∂B ∝ Aᵀ) and nothing would ever train. If both were random, step 0 would already be a different model.

**`alpha / r` is a fixed scale, not a parameter.** It exists so that when you change `r` you do not have to re-tune the learning rate: with `alpha` fixed, the size of the update stays roughly constant across ranks. The common rule of thumb is `alpha = 2r` (this floor uses r=8, alpha=16, so scaling = 2). The original paper set `alpha` to the first `r` it tried and never tuned it again.

**Where the sigils go.** On `nn.Linear` layers inside the blocks. In the Chronicler those are `c_attn` (the fused q, k, v projection, 128→384), `c_proj` (128→128), `mlp.fc` (128→512) and `mlp.proj` (512→128): four per block, sixteen in all. `lm_head` is not a target; its weight is tied to the token embedding.

**The arithmetic.** Per block at r=4: `4·(128+384) + 4·(128+128) + 4·(128+512) + 4·(512+128) = 2048 + 1024 + 2560 + 2560 = 8,192`. Four blocks: **32,768 trainable weights, 4.0%** of the 818,944-weight model. At r=8: 65,536, 8%. The fraction is much smaller on real models because `r · (in + out)` grows linearly with width while `in · out` grows quadratically: a 7B-class model with r=8 on its q and v projections trains about 4M weights, under 0.1%. The adapter file is kilobytes to megabytes next to a base of gigabytes, and one base can serve many adapters.

**Merging.** Because the correction is just a matrix, you can fold it back in:

```
W_merged = W + (alpha / r) · B @ A          bias unchanged
```

The result is an ordinary `nn.Linear` with no LoRA code and no runtime overhead. Keep the adapter separate and you get the opposite virtue: switch it off (the pristine model is back, bit for bit), swap in another one, ship it as a small file. Room 2 builds all of this; the boss uses both faces of it.

### The fine-tuning loop is the pretraining loop with two changes

The weights start from the checkpoint, and the optimizer sees only the trainable parameters. Everything else is what you wrote on Floor 7:

```python
batches = make_batches(text, tokenizer, block_size, batch_size, generator)   # endless (x, y) stream
model.train()
for step in range(steps):
    x, y = next(batches)             # (B, T) ids and (B, T) next-character targets
    _, loss = model(x, y)            # the GPT computes cross-entropy for you
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
model.eval()
```

Two hygiene points the trials check. The Chronicler was trained with dropout 0.1, so `train()` makes it stochastic and `eval()` deterministic: train in train mode, *evaluate in eval mode*, and restore whatever mode you found. And evaluation goes under `torch.no_grad()`, leaves no `.grad` behind and changes no weight.

### Catastrophic forgetting is the optimizer doing its job

Fine-tune the Chronicler on the ledger and its ledger loss falls from 6.8 to under 1 in thirty steps. Its chronicles loss rises from 0.23 to about 2.5 in the same thirty steps. Nothing went wrong. The loss you minimised contains only ledger text; there is no term anywhere that cares about the chronicles, so gradient descent overwrote whatever weights it needed. This is **catastrophic forgetting**, and it is the default outcome of sequential training on different distributions.

Measure it honestly: old-text loss before and after, evaluated **on the same batches**. Sample different windows for before and after and the sampling noise will show up as fake forgetting (or fake remembering). Room 4's report draws one seed and reuses it.

The three standard mitigations, with what they cost on this floor (30 steps on the ledger, chronicles loss rise in nats):

| Strategy | What moves | Rise in chronicles loss | Ledger learned? |
|---|---|---|---|
| Full fine-tune, lr 1e-3 | everything, far | **+2.3** | yes (0.8) |
| Full fine-tune, lr 1e-4 | everything, a tenth as far | +0.5 | partly (1.5) |
| LoRA r=8, lr 1e-3 | 8% of weights, rank-8 subspace | +1.1 | partly (1.7) |
| Full fine-tune, lr 1e-3, 50% replay | everything, but the loss still contains chronicles | **+0.2** | yes (0.8) |
| LoRA r=8, lr 3e-3, 50% replay (the boss) | adapters only, mixed batches | **+0.07** | yes (1.2) |

1. **Parameter-efficient tuning** (LoRA, adapters, prompt tuning). Fewer weights move, in a constrained subspace, so there is less to break. It *limits* forgetting relative to a full fine-tune at the same learning rate. It does not abolish it: an adapter that has learned to predict pipes and digits everywhere changes the function on chronicles text too, and nothing told it not to.
2. **A lower learning rate.** Every weight moves, but less far. You forget less and you learn less; it is a dial, not a fix.
3. **Replay / mixing.** Put old data back into every batch. Now the loss *does* have a term for the old behaviour, and the optimizer is asked to satisfy both at once. On this floor it is by far the most effective single measure, and it stacks with LoRA.

The boss adds a fourth escape that is not a mitigation but a change of architecture: keep the base frozen and make the new behaviour a *detachable* adapter. Then "forgetting" cannot happen to the base at all; the worst case is an adapter that is bad on old text, and you can switch it off.

The secret room adds a fifth, **elastic weight consolidation** (Kirkpatrick et al., 2017): estimate how much each weight mattered to the old task with the diagonal Fisher information `F_i ≈ mean over old batches of (∂L/∂w_i)²`, then add `(λ/2) Σ F_i (w_i − w*_i)²` to the new loss. Important weights become stiff springs anchored at their old values `w*`; unimportant ones stay free. Using true labels for the gradients gives the "empirical Fisher", which is what most implementations do. On this model the Fisher averages about 1e-6 per weight, so λ has to be around 1e5 before the spring competes with the task loss.

### Checkpoint hygiene

Save everything needed to *resume*, not just to *run*: the `state_dict`, the config, the tokenizer (or its vocabulary), the optimizer state if you may continue training, the step count and learning-rate schedule position, and the RNG seed. Load with `map_location` so a GPU checkpoint opens on a CPU box. Use `strict=True` and treat any missing or unexpected key as a bug in your understanding of the architecture. For adapters, save the LoRA tensors and the LoRA config (`r`, `alpha`, targets) together; a bag of tensors without `r` cannot be re-injected. The cheat sheet in the loot has the checklist.

---

## Rooms

Run `dungeon enter 8` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial. Rooms build on each other: 3 imports 2, 4 imports 2 and 3, the boss imports 2 and 3.

### 8.1 The Vault Door — `rooms/room_1_the_vault_door.py`

A door of frosted iron and a vault-keeper who hands you the key with a warning: *touch only what you mean to touch, then prove it.* Six functions: `load_chronicler` (wrap `load_pretrained`), `freeze` by substring pattern (returns how many tensors matched), `trainable_parameters`, `count_parameters`, `snapshot`, and `changed_parameters`, which diffs a live model against a snapshot with a tolerance.

The trial edits a single scalar in one tensor and expects you to name exactly that tensor. It also runs a backward pass and checks that frozen parameters have `grad is None`.

```
dungeon trial 8 room_1
```

### 8.2 The Low-Rank Sigil — `rooms/room_2_the_low_rank_sigil.py`

You cannot carve runes into a frozen door, but you can draw a sigil on the ice in front of it, as long as it starts out invisible. Build `LoRALinear` (frozen `base`, `lora_A`, zero `lora_B`, `scaling = alpha / r`, optional dropout on the LoRA input, an `enabled` flag, `merge()`), then `inject_lora` (walk the module tree and `setattr` on the parents), `lora_modules`, `lora_state_dict` and `lora_parameter_count`.

The trial checks that injection changes not one logit, that the merged `nn.Linear` matches the adapter to 1e-5, that only 32 tensors are trainable afterwards, and that your count for r=4 is exactly the arithmetic above.

```
dungeon trial 8 room_2
```

### 8.3 The Adaptation — `rooms/room_3_the_adaptation.py`

The goblin ledger. Sixty steps and the sigils. `make_batches` (an endless, seeded stream of shifted windows), `finetune` (AdamW over trainable parameters only, returns the per-step losses) and `evaluate_loss` (no grad, eval mode, restores the mode).

The trial fine-tunes LoRA r=8 at lr 3e-3 for 60 steps and expects the ledger loss under 1.5 (reference ~0.8, from 6.8). It diffs the frozen base against a snapshot, and it watches which tensors you hand to `torch.optim.AdamW`.

```
dungeon trial 8 room_3
```

### 8.4 The Forgetting — `rooms/room_4_the_forgetting.py`

Ask the goblin-speaking Chronicler about the chronicles. It hesitates. `forgetting_report` (before/after on identical batches), and three ways to make a fine-tuned *copy*: `full_finetune_copy`, `lora_finetune_copy`, `replay_finetune_copy` (each batch mixes rows from both corpora).

Then **the Prophecy of Forgetting**, to fill in *before* running: which of the four strategies forgets most, which least, whether LoRA forgets less than a full fine-tune at the same learning rate, and for each whether its chronicles loss rises by more than 0.35. Commit first. The trial measures.

```
dungeon trial 8 room_4
```

---

## Boss: The Catastrophic Forgetter

```
                 .  .   .    .
              .  ( \ | / )  .          "Teach them anything you like.
            .    (  o  o  )    .          I will keep what they lose."
           .   . (   __   ) .   .
             . . .`-----'. . .          It drifts between the shelves of the
            .  :  :  :  :  :  .         vault touching checkpoints. Where its
           :   :  :  :  :  :   :        fingers pass, the models forget. Every
          :    :  :  :  :  :    :       gradient step you take on the ledger is
               `--'  `--'  `--'         a door it walks through.
```

**Weakness:** parameter-efficient tuning, replay, and adapters that can be switched off, swapped or merged, so that the base weights are never touched at all.

- **Phase 1:** `adapt_without_forgetting(model, tokenizer, new_text, old_text, steps, generator)`: in 25 steps, ledger loss ≤ 1.6 *and* chronicles loss within 0.35 of the pristine model. A naive full fine-tune at lr 1e-3 raises the chronicles loss by 2.3. LoRA alone raises it by 1.3. LoRA with half of every batch replayed from the chronicles raises it by 0.1 and reads the ledger at 1.3. Return a copy carrying `LoRALinear` adapters; the original must be untouched.
- **Phase 2:** `AdapterSwitch`: `save_adapter` (r, alpha, targets, the LoRA tensors), `load_adapter` onto a pristine copy (inject first if needed), `disable_adapters` (logits equal the pristine Chronicler to 1e-6; the ledger is unreadable again), `enable_adapters` (the ledger is back). Runtime specialisation without touching a base weight.
- **Phase 3:** `merge_and_export(model)`: a state dict a fresh `GPT(cfg)` loads with `strict=True`, whose logits match the adapted model to 1e-5. Work on a copy; the adapted model keeps its sigils.
- **Phase 4:** `FORGETTER_PROPHECY`: which of four strategies satisfy *both* constraints. Only two do.

```
dungeon fight 8
```

## Secret room: The Replay Well *(optional)*

Under the antechamber, a well that returns memories on schedule. `mixed_batches` interleaves whole batches from two corpora with an exact ratio (among any first N batches, exactly ⌊N·ratio⌋ are replay). Then EWC: `fisher_diagonal` from gradients on old text, `ewc_penalty = (λ/2) Σ F (w − w*)²` (zero at the snapshot, quadratic in the displacement, linear in λ), and `finetune_with_ewc`. At the same learning rate and steps, EWC with λ = 1e5 leaves the chronicles about 0.9 nats better off than a plain fine-tune while still learning the ledger.

```
dungeon trial 8 --secret
```

## Loot

Clear the four rooms and defeat the Forgetter to unlock:

- **LoRA From Scratch** — `loot/lora_from_scratch.py`. A clean, commented `LoRALinear` + `inject_lora` + `merge_lora` you can drop into any PyTorch project.
- **Fine-Tuning Cheat Sheet** — `loot/finetuning_cheat_sheet.md`. Full FT vs LoRA vs adapters vs prompt tuning; the r/alpha rule of thumb; which modules to target; the forgetting mitigations; merge vs runtime adapters; checkpoint hygiene.

## Stuck?

- `dungeon hint 8 room_2` reveals one hint at a time (three per room).
- Failure messages report the observed losses and counts and say which concept went missing. Read them; they were written for exactly this.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, to compare, not to copy.

When `dungeon map` shows the Vault cleared, take the stairs. Below, a library answers every question you ask it, whether or not it has the book.
