The 188 bytes that keep their own character are three ranges: `range(ord("!"), ord("~") + 1)`, `range(ord("¡"), ord("¬") + 1)` and `range(ord("®"), ord("ÿ") + 1)`. Start the table with `{b: chr(b) for b in those}`, then walk `for b in range(256)` and give every missing byte `chr(256 + n)` with `n` counting up from 0. Byte 0 gets U+0100, byte 32 (space) gets U+0120, which is the `Ġ` in `Ġthe`.
---
The stdlib pattern, one branch per kind of piece:

```python
PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)
```

`[^\W\d_]` is "a word character that is neither a digit nor an underscore", the closest `re` gets to `\p{L}`. The `(?:[^\s\w]|_)` branch catches punctuation *and* the underscore, which is otherwise a word character that no branch claims. `\s+(?!\S)` eats a whitespace run except its last character when a word follows, so that character can attach to the word.
---
`unicode_to_bytes` is `{ch: b for b, ch in bytes_to_unicode().items()}`. `bytes_to_visible(data)` is `"".join(table[b] for b in data)` (iterating `bytes` yields ints). `visible_to_bytes(text)` is `bytes(inverse[ch] for ch in text)`. `pretokenize` is `PRETOKENIZE_PATTERN.findall(text)`; with no capturing groups, `findall` returns whole matches.
