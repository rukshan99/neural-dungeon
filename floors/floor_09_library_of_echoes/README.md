# Floor 9 — The Library of Echoes

> *Every book answers when you call to it. The echo is only as good as the question, the shelving, and your willingness to check the footnote.*

```
            ▼ stairs from the Vault of Frozen Weights
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  9.1 THE ECHO   │─────│  9.2 THE CARD    │─────│  9.3 THE SCRIBE'S│
   │     METRIC      │     │    CATALOGUE     │     │      CHUNKS      │
   └─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
   ┌─────────────────┐     ┌──────────────────┐              │
   │☠ THE LIBRARIAN'S│─────│ 9.4 THE RETRIEVAL│──────────────┘
   │      DESK       │     │       RITE       │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ behind the desk, a second ledger: the Reranker's Ledger
            ▼  stairs down to Floor 10
```

The stairs open onto a hall of shelves that goes up further than the torchlight. You say a sentence and the books answer: a murmur from every shelf, loud where a book agrees with you, silent where it does not. It is the most useful room in the dungeon and the most dangerous, because at the far end sits a Librarian who will answer *any* question, at once, with a page number. Whether the page exists is a separate matter.

This floor is about the engineering around a language model rather than the model itself. A model knows what it was trained on, as of when it was trained, and states all of it with the same tone. Retrieval-augmented generation (RAG) fixes the first two problems by finding the relevant text at question time and putting it in the prompt. It does nothing about the third. A RAG system that retrieves the wrong passage, or none, and still lets the model answer has made the confident-nonsense problem *worse*, because now the nonsense comes with a citation. The rooms build the retrieval; the boss builds the discipline.

Nothing here needs a real model. Every trial uses the deterministic mocks in `dungeon/artifacts/llm.py` (`RuleLLM` in particular) and a toy embedder in `assets/embedder.py`. The mocks are not the lesson; the plumbing around them is, and the plumbing is identical for a real provider (`dungeon/artifacts/providers.py` has one, for your own experiments).

**You will learn:** embeddings as vectors · cosine similarity and why normalisation matters · top-k selection · exact and approximate nearest-neighbour search and the recall/latency trade · chunking · the RAG pipeline · grounding, relevance gates and citation verification · BM25 and hybrid search (secret room).

**You need:** Python 3.11+ and numpy. No PyTorch on this floor.

---

## The lore of retrieval (read this before the rooms)

### An embedding is a vector, and only its direction means anything

An embedding model maps a text to a vector in R^D (D is 256 here; 768 to 3072 for real models). The model is trained so that texts with related meaning land in nearby directions. Nothing about an individual coordinate is interpretable; only the geometry between vectors is.

The measure of "same direction" is the cosine of the angle between two vectors:

```
cos(a, b) = (a · b) / (|a| |b|)          1 = same direction, 0 = orthogonal, -1 = opposite
```

The plain dot product `a · b = |a| |b| cos(a, b)` mixes in the lengths. If one document's vector is twice as long, its dot product with every query doubles, which is not a statement about relevance. Cosine removes that. Two facts make it cheap:

- If every row is L2-normalised (`x / |x|`), cosine *is* the dot product, and an entire (N, M) similarity matrix is one matmul: `normalize(A) @ normalize(B).T`.
- On unit vectors, Euclidean distance ranks identically: `|a − b|² = 2 − 2 cos(a, b)`. That is why an index can cluster with Euclidean k-means and still serve cosine queries.

Normalise once, at insert time, and normalise the query. The `eps` in `x / max(|x|, eps)` is not decoration: a zero vector (an empty string, an all-stopword string) otherwise turns every similarity into NaN, and NaN sorts wherever it likes.

Many real embedding APIs return normalised vectors already. Check; do not assume.

### Selecting the top k without sorting everything

Retrieval ends with "the k highest of M scores". `np.argsort` costs O(M log M) and sorts everything. `np.argpartition(-scores, k-1)` costs O(M) and guarantees only that the first k positions hold the k largest, unordered; you then sort those k. At M = 50 it is a rounding error. At M = 50 million it is the query.

Ties matter more than they seem: two chunks with identical text have identical scores, and a retriever whose output order changes between runs is a retriever you cannot test. Break ties by index.

### Exact search, approximate search, and recall@k

**Flat (exact) search** compares the query with every stored vector: O(N·D) per query, and by definition it finds the true k nearest. Up to a few hundred thousand vectors on one machine it is the right answer, and it is the yardstick for everything else.

**Approximate nearest-neighbour (ANN) search** reads less than everything and sometimes misses. The trade is measured by **recall@k**: of the true k nearest (from flat search), what fraction did the approximate index return?

```
recall@k = |approx_top_k ∩ exact_top_k| / k          averaged over queries
```

The **inverted-file index (IVF)** is the simplest ANN structure and the one you build here:

1. *Train:* run k-means on the vectors to get `n_clusters` centroids ("shelves").
2. *Add:* assign every vector to its nearest centroid; store it on that shelf.
3. *Search:* find the `nprobe` centroids nearest the query, score only the vectors on those shelves, return the best k.

Cost per query is about `n_clusters + (nprobe / n_clusters) · N` comparisons instead of N. Recall depends on whether the true neighbours sit on a probed shelf; a neighbour just across a cluster boundary is missed. `nprobe` is the dial: more shelves, more recall, more cost. `nprobe == n_clusters` is flat search with extra steps, and the trial checks that it is *exactly* flat search. Graph indexes (HNSW) are the other big family: better recall at the same cost, more memory, harder to write in an afternoon.

**k-means** (Lloyd's algorithm): assign each point to its nearest centroid, move each centroid to the mean of its points, repeat. Each half-step can only decrease the total squared distance (the *inertia*), so a correct implementation's inertia never climbs; the trial checks it. Two details bite: an empty cluster must keep its old centroid (re-seeding it at random can raise inertia), and initialisation matters. **k-means++** picks each new centre with probability proportional to its squared distance from the centres chosen so far, which spreads the starts out and avoids two centres landing in one blob.

### Chunking: the unit of retrieval

You do not embed whole documents. You embed **chunks**, and the chunk is what is retrieved and what the model reads. Three trade-offs:

- **Size.** Small chunks embed precisely (one topic per vector) but lose surrounding context; large chunks keep context but average several topics into one direction, and every retrieved hit costs `chunk_size` prompt tokens. A few hundred tokens is the usual starting point.
- **Overlap.** Repeating the last `overlap` characters of a chunk at the start of the next means a sentence that straddles a boundary survives whole in at least one chunk. It costs storage and a little duplicate retrieval.
- **Boundaries.** Cutting at sentence (or paragraph, or heading) ends keeps thoughts intact at the price of uneven sizes.

Two invariants matter more than the numbers. Every chunk carries exact character offsets so it can be traced back to its source: `text[start:end] == chunk.text`, always. And every chunk has an id unique across the corpus, because on this floor a **citation is a chunk id**. Deduplicate before indexing (by normalised text, so whitespace and case differences do not survive); otherwise the same passage fills three of your k slots.

### The RAG pipeline, step by step

```
question ──embed──▶ query vector ──search(k)──▶ k chunks ──build prompt──▶ model ──parse──▶ answer + citations
```

1. **Embed the question with the same embedder that embedded the chunks.** A different model, or a different version, gives vectors in a different space; the cosine between them is noise.
2. **Search for the k nearest chunks.** k is a budget. More sources raise the chance the answer is present and raise the tokens, the noise and the chance the model is distracted.
3. **Build the prompt.** A system message with three instructions (answer only from the numbered sources; cite as `[n]`; if the sources do not contain the answer, say so), then the sources numbered `[1]..[k]` with their text, then the question.
4. **Generate, then parse.** Pull the `[n]` markers out of the answer and resolve each to `sources[n − 1].id`. The chunk id is the citation. `[2]` on its own means nothing once the prompt is gone.

Everything a model on Floor 10 will do with tools, and everything Floor 11 will measure, sits on top of this loop.

### Grounding: a citation is a claim, and claims are verified

An answer is **grounded** when every claim in it can be traced to a retrieved source. The model's job is to produce grounded answers; your job is to assume it did not. There are two ways an ungrounded answer reaches a user, and each has a defence:

**Failure 1: nothing relevant was retrieved, and the model answered anyway.** Ask a RAG system about the boiling point of water when the library holds only dungeon lore, and the top-3 chunks are the *least irrelevant* of a hopeless set. A model told "answer from these sources" will often comply with something. The defence is a **relevance gate**: if the best retrieval score is below a threshold, refuse *without calling the model*. Set the threshold from data: collect best scores for questions the corpus can answer and for questions it cannot, and pick a point between the two distributions, using percentiles (the 10th of on-topic, the 90th of off-topic) rather than min and max so one strange question cannot drag the gate to the wall.

**Failure 2: the model cited a source that does not say what the model said.** A wrong number, a source that says something adjacent, or a marker attached to an invention. The defence is to **verify citations, not trust them**: split the answer into sentences, and for each sentence citing `[n]`, check that enough of the sentence's content words (lowercase, punctuation stripped, stopwords removed) actually appear in source n. Numbers outside `1..k` are invalid. Sentences whose citation fails lose the marker; an answer with no surviving citation is refused.

Be honest about the limits of a lexical check. It is *necessary, not sufficient*: a sentence copied from the source with one number changed passes; an accurate paraphrase in different words fails. Production systems layer an entailment model or a judge model over it and then evaluate the judge against human labels, which is Floor 11's business. The check here catches the cheap, common failures: the confident invention, the misattributed source, the number pulled from the air.

And the gate has its own blind spot, which you will see in the trials: the toy embedder is a bag of words, so an off-topic question that happens to reuse a passage's vocabulary can score above the threshold. The gate stops the model being asked; the verifier stops what the model said. Neither replaces the other.

### The two ledgers (secret room)

Dense retrieval matches meaning and survives paraphrase but can miss an exact rare token: a name, an error code, `bij,bjk->bik`. **BM25** is the classical sparse ranking: exact tokens weighted by rarity, with saturating term frequency and length normalisation.

```
score(q, d) = Σ_{t ∈ q} idf(t) · tf(t,d)·(k1 + 1) / (tf(t,d) + k1·(1 − b + b·|d| / avgdl))
idf(t)      = ln( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

`k1 ≈ 1.2–2.0` controls how quickly repeated occurrences stop adding score; `b ∈ [0, 1]` how much long documents are penalised. Dense and sparse scores live on different scales, so you do not add them; you fuse the *rankings* with **reciprocal rank fusion**: `rrf(id) = Σ 1 / (k + rank)` over the lists that contain the id, `k = 60` by convention. Hybrid search is dense top-n and BM25 top-n fused, first k returned.

### About the toy embedder

`assets/embedder.py` is a hashed bag of words: lowercase, tokenize, drop stopwords, strip plurals, hash every unigram and adjacent bigram to one of 256 buckets with a random sign, L2-normalise. It knows nothing about meaning, only vocabulary, and it is deterministic on every machine (blake2b, not Python's `hash`). It is enough to make every mechanic on this floor real: retrieval works when questions share words with their passage, near-duplicates score high, decoys that share words but not meaning sneak in. Swap in a real embedding model and every room's code runs unchanged; only the scores get better.

---

## Rooms

Run `dungeon enter 9` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run the room's trial. Rooms build on each other: room 2 imports room 1, room 4 imports rooms 2 and 3, the boss imports room 4.

### 9.1 The Echo Metric — `rooms/room_1_echo_metric.py`

You speak; the shelves answer. Before trusting the loudness, define it. Four functions: `normalize_rows` (eps-safe), `cosine_similarity` as one matmul for (N, D) × (M, D) → (N, M), `top_k` with `argpartition` and index-ordered ties, and `nearest_neighbors` that composes them.

The trial checks you against a slow loop, checks that a vector echoes itself at exactly 1, that shouting (`2a`) does not change the echo, that silence (`0`) echoes 0 rather than NaN, and reads your `top_k` source for `argpartition`.

```
dungeon trial 9 room_1
```

### 9.2 The Card Catalogue — `rooms/room_2_card_catalogue.py`

Ten thousand books, one question. Walk every aisle (`FlatIndex`, exact, counts every comparison), or ask the catalogue which shelves to visit (`KMeans` from scratch with k-means++, then `IVFIndex` that trains, assigns and probes `nprobe` shelves). `recall_at_k` measures what the shortcut costs.

The trial demands that IVF with `nprobe == n_clusters` equals flat search exactly, and that with 2 of 16 shelves probed on 2000 clustered vectors it keeps recall@10 ≥ 0.85 while scanning under 40% of what flat scans (the reference does 1.00 and 13%). k-means inertia must never climb.

```
dungeon trial 9 room_2
```

### 9.3 The Scribe's Chunks — `rooms/room_3_scribes_chunks.py`

The scribes copy books onto cards. `chunk_text` by characters with exact offsets and exact overlap; `chunk_by_sentences` that never cuts a thought in half and packs greedily; `dedupe` by normalised text; `chunk_corpus` over the library's passages with corpus-unique ids.

The trial reassembles the text from the non-overlapping parts, checks `text[start:end] == chunk.text` for every chunk, and checks that only a sentence longer than `max_chars` may exceed it.

```
dungeon trial 9 room_3
```

### 9.4 The Retrieval Rite — `rooms/room_4_retrieval_rite.py`

Embed, search, prompt, generate, resolve. `Retriever` wraps your `FlatIndex` and the toy embedder; `build_prompt` produces the system instruction and the numbered sources; `parse_citations` reads `[n]`; `RAGPipeline.answer` returns the answer, the citations as chunk ids, and the sources.

The model at the desk is a *faithful* `RuleLLM`: it answers a golden question only when the passage holding the answer is in its prompt, and cites the number it read it under. Twelve golden questions; the gold passage must be in your top 3 for at least ten of them (the reference gets all twelve, at rank 1).

```
dungeon trial 9 room_4
```

---

## Boss: The Hallucinating Librarian

```
               _______
              /       \          "Of course the book exists. I have
             | (o) (o) |          cited it, have I not? Page forty-two.
             |    ^    |          Third shelf. You may thank me later."
             |  \___/  |
          ___|_________|___       The Librarian answers EVERY question,
         /  |  [1] [1]  |  \      instantly, confidently, with a citation.
        /   |___________|   \     Ask about the Basilisk: a citation. Ask
       |    |  ~~~~~~~  |    |    how to boil an egg: a citation. The
       |    |  ~~~~~~~  |    |    citation is always "[1]". The book is
       |____|___________|____|    sometimes there.
```

**Weakness:** a reader who refuses when retrieval is weak, and who verifies every citation against the source it names instead of trusting the number.

The trial gives you a *gullible* `RuleLLM`: correct when the gold passage is in front of it, and otherwise inventing a plausible dungeon fact and appending `[1]`. You do not fix the Librarian. You make sure nothing it invents reaches the reader.

- **Phase 1, the gate:** `relevance_gate(scores, threshold)` and `calibrate_threshold(on_topic, off_topic)` from percentiles. The trial calibrates on the twelve golden questions against eight questions about eggs, moons and violins.
- **Phase 2, the verifier:** `content_words`, `citation_supported(sentence, chunk_text, min_overlap)` and `verify_citations(answer, sources)` sorting every `[n]` into valid, invalid and unsupported.
- **Phase 3, grounded RAG:** `GroundedRAG.answer` refuses below the gate without calling the model, verifies after, strips markers that failed, and refuses when nothing survived. Reasons come from a fixed set.
- **Phase 4, the prophecy:** eight failure modes; say which defence stops each, and which two slip past both.

```
dungeon fight 9
```

## Secret room: The Reranker's Ledger *(optional)*

Behind the desk, a second ledger that counts words instead of meanings. `BM25` from the formula, matched to a reference to 1e-9; `reciprocal_rank_fusion`, order-invariant with deterministic ties; `hybrid_search` that fuses your dense index with BM25 and must be no worse than either alone on the golden questions.

```
dungeon trial 9 --secret
```

## Loot

Clear the four rooms and defeat the Librarian to unlock:

- **The RAG Checklist** — `loot/rag_checklist.md`. Chunking, normalisation, index choice by scale, k and thresholds, prompt structure, citation verification, evaluation with recall@k and grounding checks, and the failure modes that bite in production.
- **The Vector Index (portable)** — `loot/vector_index.py`. Clean, numpy-only `FlatIndex`, `KMeans`, `IVFIndex`, `BM25`, `reciprocal_rank_fusion` and `recall_at_k` to drop into a project.

## Stuck?

- `dungeon hint 9 room_2` reveals one hint at a time (three per room).
- Failure messages name the concept, not just the assertion. When retrieval@3 misses, the message lists which questions missed and what came back instead; read it before touching the prompt.
- `solutions/` exists. After an honest attempt, compare; do not copy.

When `dungeon map` shows the Library cleared, take the stairs. Below, someone is weaving prompts, and something small is hiding instructions inside them.
