# Tokenization Cheat Sheet

*Loot from Floor 5. One page. Everything a model ever reads passed through one of these.*

## Four kinds of tokenizer

| Kind | Vocabulary | Sequence length | Unknowns | Where you meet it |
|---|---|---|---|---|
| **Character** | tiny (tens to hundreds) | one id per character, long | any unseen character is `<unk>` | toy models, the Chronicler |
| **Word** | huge (100k+) and never complete | short | every new word, typo or name | classic NLP |
| **Subword (BPE, WordPiece, Unigram)** | chosen (30k to 200k) | medium | rare, but possible if built on characters | most modern LLMs |
| **Byte-level BPE** | 256 bytes + merges | medium; 1 to 4 bytes per character before merges | **none**, ever | GPT-2 onward, Llama 3 and friends |

The trade is always vocabulary size (embedding table rows, softmax width) against sequence length (context used, attention cost, which grows with the square of the length).

## BPE in six lines

```
words  = count whitespace-split words, each as (chars..., "</w>")
repeat num_merges times:
    pairs = count adjacent symbol pairs, weighted by word count
    best  = most frequent pair (tie: lexicographically smallest)
    rewrite every word, replacing best with the concatenated symbol
    append best to the merge list                       # the merge list IS the tokenizer
encode(word) = start from chars, apply merges in the order learned
```

Byte-level: replace "chars" with UTF-8 bytes and start the vocabulary at all 256 byte values. Decode with `bytes.decode("utf-8", errors="replace")`: a token can hold part of a character, and a lone fragment prints as U+FFFD instead of raising.

## Special token conventions

| Token | Typical id | Job |
|---|---|---|
| `<pad>` | 0 | filler so a batch is rectangular; masked out of attention and loss |
| `<unk>` | 1 | what a character-level or word-level tokenizer emits for the unseen |
| `<bos>` / `<s>` | 2 | start of sequence; gives the first real token something to attend to |
| `<eos>` / `</s>` / `<|endoftext|>` | 3 | end of sequence; the model learns to emit it to stop |

Put specials *first* so `<pad>` is 0 no matter which corpus built the vocabulary. Specials occupy context and are counted by `len(encode(text))`. Chat models add role markers (`<|user|>`, `<|assistant|>`) as further specials; they are ordinary ids to the model.

## Padding side and attention masks

- `attention_mask` has the shape of `input_ids`: 1 over real tokens (specials included), 0 over padding. The pad **id** alone is not enough; attention would happily attend to it.
- **Right padding** (this floor): natural for training and for encoders; the loss is masked with the same mask.
- **Left padding**: common for batched generation with decoder-only models, so that every sequence's *last* position is a real token and the next token is appended at the same index for all rows. Position ids must then start at the first real token, not at column 0.
- Truncation should keep `<eos>` (and `<bos>`): cut the middle content, not the markers.

## Why tokenization decides cost, context and fairness

- **Cost**: hosted models bill per million tokens, input and output separately. `cost = n_in / 1e6 * p_in + n_out / 1e6 * p_out`.
- **Context**: the window is measured in tokens. Fewer tokens per byte means more text fits.
- **Multilingual fairness**: a BPE trained mostly on one script learns few merges for the others. A Devanagari or CJK character is 3 UTF-8 bytes, so with few merges it costs 1 to 3 tokens where an English word costs 1. The same message then costs more and fits less. When you evaluate a tokenizer, measure bytes per token *per script*, not just on your training corpus.
- **Compression ratio** `bytes / tokens`: 1.0 for raw bytes, roughly 2 to 2.5 for a 512-token byte-level BPE on English prose, roughly 4 for production tokenizers with 100k+ tokens on English.

## Embeddings

- An embedding is a `(V, D)` table; `W[ids]` picks rows. `one_hot(ids) @ W` is the same function with the same gradient, just slower.
- The gradient is a **scatter-add**: each position's upstream gradient lands in the row of its id. Unused rows get zeros.
- `nn.Embedding(V, D, padding_idx=0)`: the pad row is initialized to zeros and its gradient is always zero.
- Weight tying (`lm_head.weight = wte.weight`) shares the table with the output layer; the Chronicler does this.

## Positional encodings

| | Sinusoidal | Learned |
|---|---|---|
| Parameters | none | `max_len * d_model` |
| Formula | `PE[p, 2i] = sin(p / 10000^(2i/d))`, `PE[p, 2i+1] = cos(...)` | a row per position |
| Beyond `max_len` | any position has a value (extrapolation quality varies) | no row: error |
| Relative offsets | `PE[p] . PE[p+k]` depends only on `k` | must be learned |
| Used by | the original Transformer | GPT-2, BERT |

Modern models mostly use **RoPE** (rotary): rotate query/key pairs by a position-dependent angle so that dot products depend on relative position by construction. Same sin/cos machinery, applied inside attention instead of added to the input.

## Chunking with overlap

```
step = max_tokens - overlap                 # 0 <= overlap < max_tokens
windows: ids[0:max], ids[step:step+max], ids[2*step:2*step+max], ...
stop as soon as a window reaches the end    # the last window may be shorter
number of windows = 1 if n <= max else 1 + ceil((n - max) / step)
```

Consecutive windows share exactly `overlap` ids. Chunk on **token** boundaries measured by the *same* tokenizer the model will use; character counts are a guess. For retrieval, an overlap of 10 to 20 percent of the window is the usual starting point.

## Truncating a conversation

Drop the **oldest** non-system messages first, whole, until the total fits. Never split a message, never reorder, keep the system prompt. If the system prompt alone does not fit, that is a configuration error, not something to solve at runtime.

## Things that bite

1. Counting tokens with a different tokenizer than the model's. Every model family has its own; numbers are not portable.
2. Forgetting that specials count. `encode("hi")` is 4 ids on this floor, not 2.
3. Character-level `<unk>` on real text: emoji, curly quotes and accented names all vanish silently. Byte-level cannot do this.
4. Decoding a single BPE token in isolation and getting U+FFFD: the token was part of a character. Decode sequences, not tokens.
5. Positions beyond `max_len` with a learned table. Check the sequence length before the lookup, not after the traceback.
6. Treating whitespace as free. In GPT-2-style tokenizers the space belongs to the *following* word (`Ġthe`), so `"hello"` and `" hello"` are different tokens.
