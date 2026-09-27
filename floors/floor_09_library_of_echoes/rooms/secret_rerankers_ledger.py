"""SECRET - THE RERANKER'S LEDGER   (optional)

    Behind the Librarian's desk, a second ledger. It does not care what a book
    is *about*; it counts which exact words a book contains, and how rare they
    are. The two ledgers disagree often. Read both.

Dense retrieval (embeddings) matches meaning and paraphrase but can miss an
exact rare token: a name, an error code, ``bij,bjk->bik``. Sparse retrieval
(BM25) is the opposite: exact tokens, weighted by rarity, blind to synonyms.
Production systems run both and fuse the rankings.

BM25, for a query q and document d of length |d| in a corpus of N documents with
average length avgdl:

    score(q, d) = sum over terms t in q of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

k1 (about 1.2 to 2.0) controls how quickly repeated occurrences of a term stop
adding score; b (0 to 1) controls how much long documents are penalised.

Reciprocal rank fusion merges rankings without comparing their scores, which
live on different scales:

    rrf(id) = sum over rankings that contain id of 1 / (k + rank),  rank starting at 1

with k = 60 by convention. An id near the top of any list gets a strong vote;
one that appears in several lists gets several.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from ..assets.embedder import embed, tokenize  # noqa: F401  (hybrid_search needs both)


class BM25:
    """Sparse lexical scoring over a fixed, tokenised corpus.

    ``BM25(k1=1.5, b=0.75)``
    - ``fit(tokenized_docs, ids=None)``: ``tokenized_docs`` is a list of token lists;
      ``ids`` (optional) one id per document, default ``list(range(N))``. Store
      what ``score`` needs: N, avgdl, per-document term counts and lengths,
      document frequency per term. Set ``self.ids``. Return ``self``.
    - ``score(query_tokens)``: (N,) float64 array, one BM25 score per document in
      fit order. Sum over the query tokens AS GIVEN (a repeated token counts
      twice); a token in no document contributes 0.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        raise NotImplementedError("BM25.__init__() is unwritten")

    def fit(self, tokenized_docs: Sequence[Sequence[str]], ids: Sequence[Hashable] | None = None) -> "BM25":
        raise NotImplementedError("BM25.fit() is unwritten")

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("BM25.score() is unwritten")


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    """Fuse several ranked id lists into one: ``[(id, rrf_score), ...]``, best first.

    ``rrf(id) = sum(1 / (k + rank))`` over every ranking that contains ``id``,
    with rank starting at 1. Ties are broken by ``str(id)`` ascending so the
    output is deterministic. The result does not depend on the order in which
    the rankings are given.
    """
    raise NotImplementedError("reciprocal_rank_fusion() is unwritten")


def hybrid_search(query: str, dense_index, bm25: BM25, k: int, depth: int | None = None) -> list[tuple[Hashable, float]]:
    """Dense + sparse, fused: the top ``k`` ids by reciprocal rank fusion.

    ``depth`` (default ``max(10, 2 * k)``) is how many candidates each side
    contributes. Dense ranking: ``dense_index.search(embed([query])[0], depth)``
    ids in order. Sparse ranking: the ``depth`` highest ``bm25.score(tokenize(query))``
    positions (use a stable argsort of the negated scores), mapped through
    ``bm25.ids``; skip documents scoring 0, they match no query term and should
    not vote. Fuse with ``reciprocal_rank_fusion`` and return the first k.
    """
    raise NotImplementedError("hybrid_search() is unwritten")
