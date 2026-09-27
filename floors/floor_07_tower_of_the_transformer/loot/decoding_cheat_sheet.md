# Decoding Cheat Sheet

*Loot from Floor 7. A trained language model gives you a distribution over the next token. Everything below is about what to do with it.*

## The setting

At each step the model produces logits `z` of shape `(V,)` for the next token. `softmax(z)` is a probability distribution. Decoding = pick one token, append it, repeat. The choice of *how* to pick changes the text far more than most people expect, and it costs no training at all.

Two things you can measure about the output:

- **Diversity**: distinct n-grams / total n-grams (n = 4 is a good default). Loops score near 0; fresh text scores near 1.
- **Fluency**: mean negative log-likelihood per token under the model (teacher-forced). Text the model finds likely scores low; noise scores high (about `ln V` for uniform noise).

Every strategy is a point on the diversity-fluency trade-off. Judge with both; either one alone is trivially gamed (greedy maxes fluency, noise maxes diversity).

## The strategies

| Strategy | What it does | Knob | Effect |
|---|---|---|---|
| **Greedy** | `argmax(z)` | none | Deterministic. Highest per-step probability. Small or over-confident models loop: the most likely continuation of a sentence is often the sentence again. |
| **Temperature** | `softmax(z / T)` then sample | `T` | `T < 1` sharpens (toward greedy), `T > 1` flattens (toward uniform). `T -> 0` is greedy. Typical: 0.7-1.0. |
| **Top-k** | keep the k largest logits, others `-inf`, renormalise, sample | `k` | Cuts the long tail of unlikely tokens. `k = 1` is greedy. Blind to how peaked the distribution is: k=40 keeps 40 tokens whether 1 or 40 are plausible. Typical: 40-50. |
| **Top-p (nucleus)** | sort by probability, keep the smallest prefix whose cumulative mass reaches p, sample from it | `p` | Adapts to the distribution: keeps 1 token when the model is sure, many when it is not. Always keeps at least one. Typical: 0.9-0.95. |
| **Repetition penalty** | for tokens already generated: positive logit `/= r`, negative logit `*= r` | `r` | Pushes down anything already said (CTRL-style). Gentle: 1.1-1.3. Harsh on small vocabularies (character models): it penalises the letter *e*. |
| **No-repeat n-gram** | if the last n-1 tokens have appeared before, set the tokens that followed them to `-inf` | `n` | Makes an exact n-gram repeat impossible: distinct-n-gram ratio becomes 1.0 by construction. Small n forces unnatural choices; n = 3-4 for word/BPE tokens, 6+ for characters. |
| **Beam search** | keep the B highest-scoring partial sequences at every step, expand each, keep the best B again | `B` | Finds higher total log-probability than greedy. Good for translation and summarisation, where one right answer exists. For open-ended text it makes the loop problem *worse* (the highest-probability long sequence is very repetitive). Not for chat. |

## Combining them (the usual order per step)

```
logits = model(context)[:, -1, :]
logits = repetition_penalty(logits, generated, r)        # optional
logits = block_repeated_ngrams(logits, generated, n)     # optional
logits = logits / temperature
logits = top_k_filter(logits, k)                         # optional
logits = top_p_filter(logits, p)                         # optional
next   = multinomial(softmax(logits))
```

Penalties and blocks come first because they change *which* tokens are plausible; temperature and truncation then decide how boldly to draw among them. `-inf` survives every later step as probability exactly 0.

## When to use which

| Situation | Reach for |
|---|---|
| A single "correct" output (code completion, structured extraction, translation) | greedy, or beam search with a small B |
| Open-ended prose, dialogue, brainstorming | temperature 0.7-1.0 + top-p 0.9-0.95 |
| A small or over-fitted model that loops | add no-repeat n-gram blocking or a mild repetition penalty |
| Output is bland and safe | raise temperature, raise p |
| Output is incoherent | lower temperature, lower p, or add top-k |
| You need reproducibility | greedy, or a seeded generator |
| Judging a decoding change | measure diversity AND fluency on several seeds; never one number, never one sample |

## The prompt-cropping rule

The model has a fixed `block_size` (context length). Position embeddings only exist for `0 .. block_size - 1`, so before every forward pass:

```
idx_cond = idx[:, -block_size:]
```

Keep the *last* `block_size` tokens, not the first. The model then predicts from the most recent context and forgets the beginning of a long generation. This is why plain autoregressive decoding is O(T^2) in total: every step re-reads the whole (cropped) prefix. A KV cache (Floor 12) removes the re-reading, not the forgetting.

## Numbers from the Chronicler (Floor 7's model, 72-character vocabulary)

| Strategy | distinct-4-gram | mean NLL |
|---|---|---|
| greedy | 0.23 | 0.16 |
| greedy + repetition penalty 1.3 | 0.89 | 0.16 |
| greedy + no-repeat 4-gram | 1.00 | 0.50 |
| temperature 0.7 + top-k 40 | 0.85-0.94 | 0.15 |
| temperature 0.8 + top-p 0.9 + no-repeat 6-gram | 0.88-0.96 | 0.23-0.58 |
| temperature 2.0 | 0.9-1.0 | 1.2-2.0 |
| uniform noise | 1.00 | 11.2 |

Greedy is the most fluent and the least diverse. Noise is the most diverse and the least fluent. The middle rows are speech.
