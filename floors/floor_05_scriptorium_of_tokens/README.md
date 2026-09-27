# Floor 5 — The Scriptorium of Tokens

> *Before a model can read a word, someone has to decide what a word is. The scribes down here have opinions.*

```
            ▼ stairs from Floor 4
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ 5.1 THE         │─────│ 5.2 BYTE-PAIR    │─────│ 5.3 THE          │
   │  CHARACTER CODEX│     │     BINDING      │     │  EMBEDDING WELL  │
   └─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
   ┌─────────────────┐     ┌──────────────────┐              │
   │ ☠ THE GOLEM'S   │─────│ 5.4 TOKEN        │──────────────┘
   │     VAULT       │     │     ECONOMICS    │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a ledger slides under the door: the Merchant's Alphabet
            ▼  stairs down to Floor 6
```

The stairs open onto a long vaulted hall lit by candles. Rows of desks, and at every desk a scribe copying the Chronicles. Not one of them copies letters. The first numbers each character. The second binds pairs of strips into longer strips and keeps a ledger of every binding. The third drops numbers down a well and pulls vectors back up. The fourth counts everything and charges by the token. In the vault at the end sits the Babel Golem, a construct assembled from a codex, which shatters when shown a character it has never seen.

A language model never sees text. It sees a sequence of integers, and the map from text to integers is a design decision with consequences everywhere: how long your sequences are (and attention cost grows with the square of that), how big the embedding table and the output softmax are, what you pay per request, how much fits in the context window, whether a name with an accent survives the round trip, and whether a sentence in one script costs three times as much as the same sentence in another. Tokenization is also where a surprising number of production bugs live: a prompt counted with the wrong tokenizer, a chunker that splits a sentence in half, a `<pad>` row that quietly learned something.

This floor builds three tokenizers from scratch, looks at what the model does with the ids, and ends with the practical arithmetic of budgets and cost.

**You will learn:** what a token is · character tokenizers with special tokens, padding and attention masks · byte-pair encoding (training, encoding by rank, decoding) · byte-level BPE and why nothing is unknown to it · embeddings as lookup tables, `padding_idx`, sinusoidal and learned positions · counting, truncating, chunking and pricing tokens.

**You need:** Python 3.11+, numpy, and PyTorch (room 5.3 and one helper in 5.1). Everything runs on CPU in seconds.

---

## The lore of tokens (read this before the rooms)

### A token is an integer with a dictionary

A **tokenizer** is a pair of functions: `encode(text) -> list[int]` and `decode(ids) -> str`, plus the **vocabulary** that defines them: `V` distinct token strings, numbered `0..V-1`. The model owns a `(V, D)` embedding table indexed by those numbers and a `(D, V)` output layer that scores them. `V` is therefore a model dimension, not a preprocessing detail.

Everything on this floor is a trade between vocabulary size and sequence length:

| Kind | `V` | ids per English word | Unknowns |
|---|---|---|---|
| character | tens to hundreds | about 5 | every unseen character |
| word | 100k+ and never complete | 1 | every unseen word |
| subword (BPE) | you choose, 30k to 200k | 1 to 2 | rare |
| byte-level BPE | 256 + merges | 1 to 2 | **none** |

### Character tokenizers, special tokens, masks

The simplest tokenizer that works: sort the distinct characters of the training text and number them. The Chronicles have 69 distinct characters. Round trips are lossless *for text made of those characters*. Everything else gets one shared id, `<unk>`, and is gone forever after decoding.

Four **special tokens** sit in front of the ordinary characters, and their ids are fixed by convention on this floor and used by every floor below:

```
0  <pad>   filler that makes a batch rectangular
1  <unk>   the id of any character the codex lacks
2  <bos>   beginning of sequence, prepended by encode()
3  <eos>   end of sequence, appended by encode()
4..        the characters, sorted
```

Putting specials first (not after the characters) means `<pad>` is 0 regardless of corpus, so a padded batch is zeros on the right whichever codex produced it.

**Padding and attention masks.** Models take `(B, T)` tensors, but sentences have different lengths. `batch_encode` right-pads every sequence with `<pad>` to a common length and returns an **attention mask** of the same shape: 1 over real tokens (specials included), 0 over padding.

```
input_ids       [[2, 5, 9, 3, 0, 0],      <bos> a e <eos> <pad> <pad>
                 [2, 5, 6, 7, 8, 3]]      <bos> a b c d <eos>
attention_mask  [[1, 1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 1, 1]]
```

The mask, not the pad id, is what tells attention and the loss to ignore the filler; without it a model attends to padding like any other token. Truncation to `max_len` should keep the markers: cut content, then put `<eos>` back as the last id.

### Byte-pair encoding

BPE (Sennrich, Haddow and Birch, 2016) learns a vocabulary from data by repeatedly merging the most frequent adjacent pair of symbols. The whole algorithm:

```
words  = count whitespace-split words, each as a tuple of characters + "</w>"
repeat num_merges times:
    pairs = count adjacent symbol pairs, weighted by word count
    best  = the most frequent pair; on a tie, the lexicographically smallest
    rewrite every word, replacing best with the concatenated symbol
    append best to the merge list
```

The **merge list is the tokenizer**. To encode new text, split it into words, start each from its characters plus `</w>`, and apply the merges **in the order they were learned**. Order matters: `(es, t)` can only fire after `(e, s)` has created the symbol `es`. Decoding is concatenation with every `</w>` turned back into a space.

The textbook example (`low` ×5, `lower` ×2, `newest` ×6, `widest` ×3): `(e, s)` appears 9 times, ties with `(s, t)` and `(t, </w>)`, and wins the tie-break. The first five merges are `(e,s) (es,t) (est,</w>) (l,o) (lo,w)`, after which `lowest` is `[low, est</w>]`.

The end-of-word marker is what makes `hug` and `hugs` distinct: `hug</w>` can never appear inside `hugs`. Every merge can only bind two symbols into one, so more merges never means more tokens. Characters never seen in training simply stay as single-character tokens; this scheme has no `<unk>`, but it *does* need every character to exist as a base symbol, which is where the Golem gets into trouble.

### Byte-level BPE: nothing is unknown

UTF-8 encodes every code point as 1 to 4 bytes: ASCII is one byte; `α` is `CE B1`; a CJK character is three bytes; most emoji are four. Bytes `80..FF` never form a valid sequence on their own; they are lead or continuation bytes of a longer character.

Run BPE over **bytes** instead of characters and start the vocabulary from all 256 byte values. Then any string whatsoever encodes (worst case: one id per byte), merges buy compression on top, and there is no `<unk>` because there is nothing to be unknown. GPT-2 did exactly this: 256 bytes + 50,000 merges + one `<|endoftext|>` = 50,257 tokens.

Two consequences to keep straight:

- A merged token is a **byte string**, and it may contain *part* of a character (the `CE` of an `α` with the `B1` in the next token). Decoding a whole sequence is exact. Decoding a lone token may hit an incomplete sequence, so decode with `errors="replace"` and accept U+FFFD (`�`) for fragments instead of an exception.
- Pre-tokenization matters. Splitting the text into pieces first (a word with its leading whitespace, or a whitespace run) keeps merges from crossing word boundaries. The pieces must concatenate back to the text exactly, or the round trip breaks.

**Compression ratio** `bytes / tokens` measures how much the merges bought you: exactly 1.0 with no merges, about 2.4 at 512 tokens trained on the Chronicles, about 4 for production tokenizers on English. Measure it per script: a tokenizer trained on English has few merges for Devanagari, so the same message costs more tokens there. That is a fairness and a cost problem at once.

### Embeddings are lookup tables

An embedding is a `(V, D)` table `W`. Looking up ids of shape `(B, T)` gives `(B, T, D)`. Two ways to write it:

```python
W[ids]                                        # advanced indexing: pick rows
one_hot(ids, V).to(W.dtype) @ W               # a matmul with a vector of zeros and a single 1
```

Same output, same gradient. The gradient with respect to `W` is a **scatter-add**: the upstream gradient at each position is added into the row of that position's id. A row used twice receives two contributions; a row never used receives zeros. Indexing is just the version that skips materializing a `(B, T, V)` tensor.

`nn.Embedding(V, D, padding_idx=0)` freezes one row: it is initialized to zeros and its gradient is always zero, so the pad token never learns anything and never leaks into the loss through its embedding.

### Positions

Attention treats its input as a set: without extra information it cannot distinguish `dog bites man` from `man bites dog`. The fix is a second `(max_len, D)` table, indexed by position and **added** to the token embeddings.

**Sinusoidal** (Vaswani et al., 2017), no parameters:

```
PE[pos, 2i]     = sin(pos / 10000^(2i / D))
PE[pos, 2i + 1] = cos(pos / 10000^(2i / D))
```

Column pair `i` is a sine and cosine at angular rate `w_i = 1 / 10000^(2i/D)`: fast for small `i`, slow for large `i`, like the hands of a clock with `D/2` dials. A property worth knowing: `PE[p] · PE[p + k] = Σ_i (sin(w_i p) sin(w_i (p+k)) + cos(w_i p) cos(w_i (p+k))) = Σ_i cos(w_i k)`. It depends only on the offset `k`, not on `p`, so a dot product can read "how far apart" directly. At position 0 every angle is 0 and the row is `[0, 1, 0, 1, ...]`.

**Learned**: an `nn.Embedding(max_len, D)` indexed by `torch.arange(T)`. More flexible, one row per position, and a sequence longer than `max_len` has no row to look up. The Chronicler in `dungeon/artifacts/tiny_gpt.py` uses this (`wpe`), as GPT-2 did.

### Token economics

- **Counting**: `len(tokenizer.encode(text))`, with the *model's own* tokenizer. Specials count; they occupy context.
- **Cost**: prices are quoted per million tokens, input and output separately. `cost = n_in / 1e6 · p_in + n_out / 1e6 · p_out`.
- **Truncating a conversation**: drop the **oldest** non-system messages, whole, until the total fits. Never split a message, never reorder, keep the system prompt.
- **Chunking with overlap**: windows of `max_tokens` ids that step by `max_tokens − overlap`, so consecutive windows share exactly `overlap` ids and a sentence that straddles a boundary is intact in at least one of them. Stop as soon as a window reaches the end; the last one may be shorter. For `n` ids: `1` window if `n ≤ max_tokens`, else `1 + ceil((n − max_tokens) / step)`.

---

## Rooms

Run `dungeon enter 5` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial.

### 5.1 The Character Codex — `rooms/room_1_character_codex.py`

The first scribe hands you a codex with four lines in red ink and the rest in black. Build `CharTokenizer`: `__init__` (vocabulary, `stoi`, `itos`, the four ids), `encode` with `<bos>`/`<eos>` and `<unk>` for strangers, `decode` that strips the red ink unless asked not to, and `batch_encode` that produces right-padded `input_ids` and an `attention_mask` as lists of lists, truncating while keeping `<eos>`. `to_tensors` turns the lists into int64 tensors.

The trial round-trips a page of the Chronicles, checks the numbering, shows the codex an Omega, and inspects every pad and every mask bit.

```
dungeon trial 5 room_1
```

### 5.2 Byte-Pair Binding — `rooms/room_2_byte_pair_binding.py`

The second scribe binds strips. Six functions: `word_counts`, `pair_counts`, `merge_pair`, `train_bpe` (most frequent pair, ties to the lexicographically smallest, stop when nothing is left), `encode_bpe` (replay the ledger in rank order) and `decode_bpe`.

Then **the Binding Prophecy**: a seven-word corpus, `hug hug hug pug pun pun bun`. Which pair binds first? Second? Third? How many tokens is `hugs` after three merges? Work it out by hand, including the tie-break, before running anything.

```
dungeon trial 5 room_2
```

### 5.3 The Embedding Well — `rooms/room_3_embedding_well.py`

Drop an id in, get a vector out. `embedding_lookup` by indexing, `embedding_as_matmul` by one-hot matmul (the trial checks outputs *and* gradients agree, and that a row used twice gets a doubled gradient), `token_embedding` with `padding_idx` (the pad row must stay at zero gradient), `sinusoidal_positional_encoding` exactly per the formula (checked against a reference and for the offset property), and a `LearnedPositionalEmbedding` module.

```
dungeon trial 5 room_3
```

### 5.4 Token Economics — `rooms/room_4_token_economics.py`

The bursar's desk, pure Python. `count_tokens`, `estimate_cost`, `truncate_messages` (oldest first, system protected, never split, order kept) and `chunk_by_tokens` (exact overlap, full coverage, short last window, nothing empty).

Then **the Ledger Prophecy**: how many tokens do `the thing`, `then`, `singing` and `thing thing thing` cost under the Character Codex versus a six-merge BPE ledger? One of them is a trap about the end-of-word marker.

```
dungeon trial 5 room_4
```

---

## Boss: The Babel Golem

```
                 ______
              .-'      '-.           "Show me a letter and I will
             /   ______   \           number it. Show me a letter I have
            |   |  __  |   |          not seen and I will... I will..."
            |   | |  | |   |
            |   |_|__|_|   |         The Golem was built from a codex. Every
             \  ________  /          character it knows has a number. Every
              |  |    |  |           character it does not know is <unk>, and
              |  |    |  |           enough <unk> and it SHATTERS. It was
             _|__|____|__|_          shown Greek once. They are still sweeping.
            |______________|
```

**Weakness:** a vocabulary that already contains every possible input. There are only 256 byte values.

- **Phase 1:** `ByteLevelBPE` with `train(text, vocab_size)`, `encode`, `decode`, plus `pretokenize` and `merge_ids`. Base vocabulary of 256 single bytes; merges create ids `256, 257, ...` whose byte strings are the concatenation of their parts; encoding merges the lowest-ranked present pair first.
- **Phase 2:** round trip on a battery of strings: Greek, Devanagari, CJK, Hangul, Arabic, Hebrew, emoji with ZWJ sequences, flags, combining marks, astral-plane letters, exotic digits, odd whitespace, 3000 repeated characters, words the Golem never saw, and the empty string. Byte for byte.
- **Phase 3:** `unknown_rate(tokenizer, text)`. Zero for you on everything. Strictly positive for the Character Codex of room 5.1 on any script it did not train on. The Codex shatters; the Golem does not.
- **Phase 4:** `compression_ratio(tokenizer, text)`. Exactly 1.0 untrained; above 2.0 on the training page at `vocab_size=512`; still above 1.6 on a page the Golem never saw.

```
dungeon fight 5
```

## Secret room: The Merchant's Alphabet *(optional)*

A ledger slides under the vault door. GPT-2 stores its byte-level vocabulary as text, which requires a printable stand-in for every byte value: `bytes_to_unicode()`, a bijection from 256 bytes to 256 printable, non-whitespace characters, in which a space is `Ġ`. Then the pre-tokenizer: GPT-2's real pattern needs `\p{L}` from the third-party `regex` module; write the closest stdlib `re` version that splits words, numbers, punctuation runs and whitespace so that the pieces always concatenate back to the text.

```
dungeon trial 5 --secret
```

## Loot

Clear the four rooms and defeat the Golem to unlock:

- **Tokenization Cheat Sheet** — `loot/tokenization_cheat_sheet.md`. Char vs word vs subword vs byte-level, BPE in six lines, special-token conventions, padding side and masks, why tokenization decides cost, context and multilingual fairness, positional encodings compared, and the chunking recipe.

## Stuck?

- `dungeon hint 5 room_2` reveals one hint at a time (three per room).
- The trial failure messages say *which* concept went wrong and show the values they saw. Read them before changing code.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, and to compare, not to copy.

When `dungeon map` shows the Scriptorium cleared, take the stairs. Below, something is paying attention.
