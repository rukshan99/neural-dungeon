`count_tokens` is `len(tokenizer.encode(text))`. `estimate_cost` is `n_input / 1e6 * price_in + n_output / 1e6 * price_out`. `truncate_messages`: copy the list, then `while total > budget`: find the index of the first message that is droppable (`not (keep_system and m["role"] == "system")`); if there is none, stop; otherwise `del kept[idx]`.
---
`chunk_by_tokens`: validate, return `[]` for empty input, then `step = max_tokens - overlap` and
```python
start = 0
while True:
    chunks.append(ids[start:start + max_tokens])
    if start + max_tokens >= len(ids):
        break
    start += step
```
The break condition is what stops you emitting a final window that contains nothing new.
---
Prophecy. `char` counts characters plus two specials: `"the thing"` is 9 + 2 = 11, `"thing thing thing"` is 17 + 2 = 19. `bpe` replays TOY_MERGES per word: `the` becomes `th e </w>` then `the </w>` then `the</w>` (1 token); `thing` becomes `th ing</w>` (2); `then` becomes `the n </w>` (3, because `(the, </w>)` needs `</w>` right after `the`); `singing` becomes `s ing ing</w>` (3); three `thing`s cost 6.
