`Retriever.__init__` embeds every chunk text in one call, `index.add(vectors, [c.id for c in chunks])`, and keeps `{c.id: c}`. `retrieve` embeds `[query]`, takes row 0, calls `index.search(q, k)` and maps each returned id back to its Chunk, keeping the score as a float. Same embedder for chunks and queries, always.
---
`build_prompt` returns `[Message.system(SYSTEM_INSTRUCTION), Message.user(body)]` where `body` is `"Sources:"`, one line `f"[{n}] {chunk.text}"` per retrieved chunk numbered from 1, a blank line, then `f"Question: {question}"`, joined with newlines. `parse_citations`: `re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)`, split each match on commas, `int()`, and append to a list only if not already present.
---
```python
def answer(self, question, k=None):
    retrieved = self.retriever.retrieve(question, k or self.k)
    text = self.llm.complete(build_prompt(question, retrieved)).text
    sources = [c for c, _ in retrieved]
    citations = [sources[n - 1].id for n in parse_citations(text) if 1 <= n <= len(sources)]
    return {"answer": text, "citations": citations, "sources": sources}
```
