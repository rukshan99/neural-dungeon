# Fine-Tuning Cheat Sheet

*Loot from Floor 8. Everything you need to decide how to adapt a pretrained model without wrecking it.*

## The four ways to adapt a model

| Method | What trains | Extra params | Inference cost | Forgetting risk | Use when |
|---|---|---|---|---|---|
| **Full fine-tune** | every weight | 0 (but full optimizer state: ~2x model size for AdamW) | none | **high** | you have lots of in-domain data, compute, and a way to evaluate the old behaviour |
| **LoRA** | low-rank `B A` next to chosen Linears | `r · (in + out)` per layer, ~0.1-2% | none after merge; two small matmuls if kept separate | medium (less than full FT at the same lr) | the default for adapting LLMs on one GPU; multi-tenant serving with one base |
| **Adapters** (bottleneck modules) | small MLPs inserted between layers | ~1-5% | a little, always (they cannot be merged) | medium | you want per-task modules with their own nonlinearity |
| **Prompt / prefix tuning** | a few learned token embeddings or per-layer prefixes | tiny (tokens x d) | uses context length | low for the base (nothing changes), but the prefix can be brittle | very small budgets; the base is shared and immutable |

Rule of thumb: start with LoRA. Move to full fine-tuning only when LoRA plateaus and you have measured that the old behaviour survives.

## LoRA settings

```
y = W x + b + (alpha / r) · B (A x)      A: (r, in) random     B: (out, r) ZERO
W_merged = W + (alpha / r) · B @ A       bias unchanged
```

- **r** (rank): 4-16 for most adaptation; 32-64 when the new task is far from pretraining or the corpus is large. Doubling r doubles adapter size and rarely doubles quality.
- **alpha**: fix `alpha = 2r` (scaling 2) or `alpha = r` (scaling 1) and leave it. Its job is to keep the update size stable when you change r so the learning rate does not need re-tuning.
- **Learning rate**: LoRA tolerates and wants a *higher* lr than full fine-tuning, typically 1e-4 to 3e-4 on LLMs (this floor used 3e-3 on an 800K-param model). Full fine-tuning of the same model: 1e-5 to 5e-5.
- **Dropout on the LoRA input** (0.05-0.1) helps on small datasets; leave it 0 otherwise.
- **B must start at zero.** Random B means step 0 is already a different model. Zero A *and* zero B means no gradient ever flows.

## Which modules to target

| Family | Attention projections | MLP | Notes |
|---|---|---|---|
| GPT-2 style (nanoGPT, the Chronicler) | `c_attn` (fused q,k,v), `c_proj` | `fc`, `proj` | never `lm_head` (tied to `wte`) |
| Llama / Mistral style | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `gate_proj`, `up_proj`, `down_proj` | |
| Encoder (BERT style) | `query`, `key`, `value`, `attention.output.dense` | `intermediate.dense`, `output.dense` | |

Minimum that usually works: **q and v**. Better for a fixed budget: **all Linears at a lower r** rather than q,v at a higher r. Embeddings and LayerNorm are usually left frozen; unfreezing LayerNorm gains and biases is a cheap extra if you need it.

Parameter count check: `sum(r * (in + out))` over targeted layers. For a 7B model, r=8 on q and v: 32 layers x 2 x 8 x (4096 + 4096) ≈ 4.2M, under 0.1%.

## Catastrophic forgetting and its mitigations

Fine-tuning on new data minimises a loss with **no term for the old data**. The optimizer will overwrite whatever it needs. Measure it: old-data loss (or a task metric) before and after, **on identical batches**.

| Mitigation | Mechanism | Trade-off | On this floor (30 steps, rise in old loss) |
|---|---|---|---|
| Parameter-efficient tuning (LoRA) | fewer weights move, low-rank subspace | limits but does not abolish forgetting | full FT +2.3 vs LoRA +1.1 at the same lr |
| Lower learning rate | every weight moves less far | learns less too | lr 1e-3 +2.3 vs lr 1e-4 +0.5 |
| **Replay / data mixing** | old data back in every batch, so the loss protects old behaviour | needs (some of) the old data; 10-50% of each batch | full FT + 50% replay **+0.2**; LoRA + replay **+0.07** |
| Regularisation toward the checkpoint (EWC, L2-SP) | penalty `(λ/2) Σ F_i (w_i − w*_i)²`, F = diagonal Fisher from old-data gradients | one more hyper-parameter (λ, often huge because F is tiny); approximate | EWC λ=1e5: ~0.9 nats better than plain FT |
| Keep the base frozen, ship adapters | forgetting cannot touch the base; switch the adapter off to get it back | adapter may still be bad on old inputs while on | disabled adapters == pristine model, bit for bit |

Combine them. LoRA + replay + a modest lr is the boring, effective default.

## Merge vs runtime adapters

| | Merged (`W + ΔW` folded in) | Runtime adapters (kept separate) |
|---|---|---|
| Inference cost | zero extra | two small matmuls per wrapped layer |
| Serving many tasks on one base | one full copy per task | one base + one small adapter per task, swappable per request |
| Turn it off | reload the base | `enabled = False` |
| Stacking / composing adapters | no | yes |
| Quantised base (QLoRA) | cannot merge into 4-bit weights cleanly; merge into a full-precision copy | works as is |
| Precision | merge in fp32 then cast; merging in bf16 loses adapter detail | n/a |

Export for deployment: merge into a copy, `state_dict()`, load into a fresh model with `strict=True`, verify outputs match the adapted model to float32 rounding (~1e-5 relative; a wrong merge is off by orders of magnitude more) before you delete anything.

## Supervised fine-tuning (SFT)

Pretraining continued on dialogues rendered with a fixed template. Same loss, three new habits.

**Template.** One function renders a conversation; the same function renders the inference prompt (ending with the empty assistant header, `add_generation_prompt=True` in HF terms). Never hand-write the prefix at inference. On this floor: `--- User:\n{q}\n\n--- Assistant:\n{a}\n\n`.

**Masking.** Labels are the next token; positions whose target is a prompt token get `-100`, and so does every pad and the last position.

```python
ids    = prompt_ids + response_ids            # response_ids includes the end tag
labels = [-100] * len(ids)
for t in range(len(ids) - 1):
    if t + 1 >= len(prompt_ids):               # the TARGET is a response token
        labels[t] = ids[t + 1]
loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
```

`-100` is PyTorch's default `ignore_index`. Check `supervised_fraction(labels)` once: if it is 1.0 you forgot the mask; if it is 0.0 you shifted the wrong way.

| Choice | Do | Because |
|---|---|---|
| Padding side | right, labels `-100` | causal attention cannot see a pad that comes after the real tokens |
| Truncation | from the prompt side, keep the answer whole (or drop the example) | a headless answer with no end tag teaches the model to never stop |
| Padding vs packing | pad for small runs; pack (concatenate with EOS, block-diagonal mask if you can) for large ones | packing wastes no compute but lets examples attend across boundaries |
| Epochs | 1-3 | small sets memorise fast |
| Evaluation | held-out loss **per answer token** + read the generations | training loss on a small set says nothing; a low loss can hide "answers everything the same way" |
| Loss floor | estimate it | unpredictable content (a price's digits) puts a floor well above 0 |

Prompt loss on purpose is a valid choice on tiny datasets; by accident it is a bug.

## Preference optimisation (DPO)

Chosen `y_w`, rejected `y_l`, a frozen reference `pi_ref`, and no reward model:

```
r_hat(y) = beta * ( log pi(y|x) - log pi_ref(y|x) )              implicit reward
L        = -log sigmoid( r_hat(y_w) - r_hat(y_l) )               mean over pairs
```

`log pi(y|x)` is the **sum** of token log-probs over the answer positions only. At step 0 the policy is the reference, every `r_hat` is 0 and the loss is `ln 2 = 0.693`; if your first step is not, dropout is on or the adapters did not start at zero.

| Knob | Default | Effect |
|---|---|---|
| `beta` | 0.1 | scales the logit; larger = shorter leash to the reference (saturates sooner), smaller = more drift |
| lr | 10x lower than SFT (5e-7 to 5e-6 on LLMs; 1e-3 for LoRA on the 800K Chronicler) | DPO is sensitive; a big lr collapses the chosen log-prob |
| dropout | **off** for policy and reference | the loss is a difference of log-probs; noise in one side reads as reward |
| steps / epochs | 1 epoch, often less | over-optimisation shows up fast |

What to log: loss, mean implicit reward margin, **reward accuracy** (fraction of pairs with `r_hat(y_w) > r_hat(y_l)`; starts at 0), mean chosen and rejected log-probs (both often fall: watch the chosen one), old-data loss, answer length.

Failure modes: (1) *likelihood displacement*: the chosen log-prob falls too and probability mass leaves the data entirely; stop early or use a variant that anchors the chosen log-prob (DPO-Positive). (2) *Length bias*: summed log-probs favour long answers; compare lengths before/after or use a length-normalised objective (SimPO). (3) *Data*: the model learns exactly the contrast the pairs contain, so pairs that differ in irrelevant ways teach irrelevant features. RLHF with a reward model + PPO is the heavier alternative; the reference-model KL term plays the same anchoring role there.

## Checkpoint hygiene

Save together, always:

- [ ] `model.state_dict()` (or the adapter state dict, if the base ships separately)
- [ ] the **config** that rebuilds the architecture (vocab size, layers, heads, width, block size, activation, norm type)
- [ ] the **tokenizer** (vocabulary and merges); mismatched ids are silent garbage
- [ ] for adapters: **`r`, `alpha`, targets, dropout**; a bag of `lora_A`/`lora_B` tensors without `r` cannot be re-injected
- [ ] to *resume* training: **optimizer state**, LR-scheduler state, step count, RNG state, and the data position or seed
- [ ] the base checkpoint's identity (hash or version) an adapter was trained against

Load with `map_location="cpu"` (GPU checkpoints open on any machine), `strict=True` (missing or unexpected keys are a bug in your understanding), then `model.eval()` before evaluation. Run a smoke test on a fixed input and compare with the value recorded at save time.

## Freezing checklist

1. `for p in model.parameters(): p.requires_grad_(False)` **before** injecting adapters (they set their own factors trainable).
2. Hand the optimizer `[p for p in model.parameters() if p.requires_grad]`. Not `model.parameters()`. AdamW's moments would waste 2x the frozen weights' memory, and a weight frozen after a backward pass still carries a stale `.grad` the optimizer would step on (a `.grad` of `None` is skipped; do not rely on it).
3. After training, diff against a snapshot. The only tensors that changed should be the ones you meant.
4. Train in `train()`, evaluate in `eval()` under `torch.no_grad()`, and put the mode back.
