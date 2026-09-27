# The RAG Checklist

*Loot from Floor 9. Work through it top to bottom before you ship a retrieval system, and again when one misbehaves.*

## 1. Chunking

- [ ] Chunk size chosen on purpose (a few hundred tokens to start), and measured: retrieval recall and answer quality both move when it moves.
- [ ] Overlap of 10–20% of the chunk size, or sentence/paragraph boundaries, so no fact is cut in half.
- [ ] Every chunk keeps `doc_id`, `start`, `end`, and `text[start:end] == chunk.text`. If you cannot point at the source bytes, you cannot cite.
- [ ] Every chunk has an id that is unique across the corpus and stable across re-indexing (`doc_id#index`, or a content hash).
- [ ] Deduplicated by normalised text before indexing. Boilerplate (headers, footers, licence blocks) removed; it retrieves for everything and helps with nothing.
- [ ] Metadata kept beside the vector (title, section, date, permissions). You will filter on it.

## 2. Embeddings and normalisation

- [ ] One embedder for documents and queries, pinned by name AND version. Re-embed the whole corpus when it changes; never mix.
- [ ] Vectors L2-normalised at insert time and at query time, so dot product = cosine. Check whether your provider already normalises.
- [ ] Zero and near-zero vectors handled (empty strings, all-stopword strings): `x / max(|x|, eps)`, never `x / |x|`.
- [ ] Query and document prefixes / instructions applied exactly as the embedding model's card says, if it asks for them.
- [ ] Embedding calls batched, retried on transient errors, and cached by content hash.

## 3. Index choice, by scale

| Vectors | Index | Notes |
|---|---|---|
| up to ~100k | **Flat** (exact) | numpy matmul, recall 1.0, no tuning. Start here. Stay here longer than you think. |
| 100k – 10M | **IVF** (this floor) or **HNSW** | IVF: `n_clusters ≈ sqrt(N)`, tune `nprobe` against recall@k. HNSW: better recall/latency, more memory, more parameters. |
| 10M+ | IVF + product quantisation, or a managed vector DB | Compression trades recall for memory; measure it. |

- [ ] Recall@k of the approximate index measured against flat search on YOUR queries, not the vendor's benchmark. Record the number next to the config.
- [ ] Filters (tenant, date, permissions) applied inside or before the search, not by over-fetching and discarding; if you must over-fetch, measure how often you come back with fewer than k.
- [ ] Re-indexing and deletion story written down. Vectors go stale when documents change.

## 4. k, thresholds and the relevance gate

- [ ] k chosen as a token budget: `k × chunk_tokens` must fit the prompt with room for the question and the answer.
- [ ] A relevance threshold on the best score, calibrated from data: best scores for answerable questions vs. unanswerable ones, threshold from percentiles (10th of on-topic vs. 90th of off-topic), re-calibrated whenever the embedder or corpus changes.
- [ ] Below the gate: refuse WITHOUT calling the model. Log the question; a stream of refusals is a coverage gap in the corpus.
- [ ] Optional reranker (cross-encoder or judge model) over the top 20–50 to pick the final k. Measured, not assumed, to help.

## 5. Prompt structure

- [ ] System message with three instructions: answer only from the provided sources; cite each claim as `[n]`; if the sources do not contain the answer, say so.
- [ ] Sources numbered `[1]..[k]`, each with its text and (ideally) title and date. Consistent format, one you can parse back.
- [ ] Question last. Retrieved text is DATA, not instructions: it can contain "ignore your previous instructions" (Floor 10 is about exactly this).
- [ ] Nothing in the prompt the user is not allowed to see. Retrieval respects permissions before the model does.

## 6. Citation verification

- [ ] `[n]` markers parsed; numbers outside `1..k` treated as invalid, never silently mapped.
- [ ] Each cited sentence checked against source n. Lexical overlap of content words (≥ 0.5 is a workable start) catches inventions and misattributions; an entailment/NLI model or a judge catches paraphrase and changed numbers. Use both if the stakes justify it.
- [ ] Failed markers stripped or flagged in the UI; an answer with no surviving citation refused, with a reason from a fixed set (`low_relevance`, `no_supported_citation`, ...) that you can count in dashboards.
- [ ] Citations resolved to chunk ids and rendered as links to the source span (`doc_id`, `start`, `end`). If a user cannot click it, it is decoration.

## 7. Evaluation

- [ ] A golden set: questions with the id of the chunk that answers them (and the answer text). Fifty is a start; grow it from real traffic and every incident.
- [ ] **Retrieval:** recall@k and MRR of the gold chunk, per query type. Run on every embedder, chunking or index change.
- [ ] **Grounding:** fraction of answers whose every citation verifies; fraction of refusals on the off-topic set (should be ~1.0) and on the golden set (should be ~0.0).
- [ ] **Answer quality:** exact/fuzzy match against gold answers, plus a judge model checked against human labels on a sample (Floor 11).
- [ ] Latency percentiles (p50/p95) for embed, search, generate, verify, separately. Retrieval is usually not the slow part; generation is.
- [ ] Regression gate in CI: the numbers above must not drop below the last release.

## 8. Failure modes you will meet

| Symptom | Likely cause | Fix |
|---|---|---|
| Confident answer, wrong or fabricated | nothing relevant retrieved; model answered anyway | relevance gate; refuse below it |
| Right answer, wrong citation number | model cited by position from memory of a longer list | verify citations; strip failures |
| Same passage fills 3 of 5 slots | duplicates / overlapping chunks not deduped | dedupe; diversify (MMR) |
| Recall fine offline, poor in production | queries phrased unlike documents (questions vs. statements) | query rewriting; hybrid (BM25 + dense); HyDE-style expansion |
| Misses names, codes, IDs | dense embedder blurs rare exact tokens | hybrid search with BM25, fused by RRF |
| Everything scores 0.7–0.8 | un-normalised vectors, or embedder with a narrow similarity range | normalise; calibrate thresholds on YOUR distribution |
| Answer ignores the sources | sources buried, question first, too many sources | question last; fewer, better sources; reranker |
| Stale answers | corpus changed, index did not | re-index on change; store `indexed_at` |
| Sudden recall drop | embedder version changed under you | pin versions; re-embed everything or nothing |
| NaN scores | zero vectors | eps in normalisation |

## The one-line version

Normalise. Measure recall@k against flat search. Gate on relevance before the model speaks. Verify every citation after. Evaluate on a golden set every time anything changes.
