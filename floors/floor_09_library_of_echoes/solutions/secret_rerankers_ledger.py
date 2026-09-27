"""SECRET - THE RERANKER'S LEDGER  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

BM25 as Robertson/Lucene define it, reciprocal rank fusion, and a hybrid search
that fuses a dense ranking with a sparse one.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np

from ..assets.embedder import embed, tokenize


class BM25:
    """Sparse lexical scoring over a fixed corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[Hashable] = []
        self._tf: list[Counter] = []
        self._lengths: np.ndarray = np.zeros(0)
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def fit(self, tokenized_docs: Sequence[Sequence[str]], ids: Sequence[Hashable] | None = None) -> "BM25":
        docs = [list(d) for d in tokenized_docs]
        self.ids = list(ids) if ids is not None else list(range(len(docs)))
        if len(self.ids) != len(docs):
            raise ValueError("one id per document")
        self._tf = [Counter(d) for d in docs]
        self._lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avgdl = float(self._lengths.mean()) if docs else 0.0
        return self

    def idf(self, term: str) -> float:
        n, df = len(self._tf), self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        n = len(self._tf)
        scores = np.zeros(n, dtype=np.float64)
        if n == 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self._lengths / self._avgdl)  # (N,)
        for term in query_tokens:
            if term not in self._df:
                continue
            tf = np.array([c.get(term, 0) for c in self._tf], dtype=np.float64)
            scores += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
        return scores


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    """sum(1 / (k + rank)) per id across rankings; best first; ties by str(id)."""
    fused: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], str(kv[0])))


def hybrid_search(query: str, dense_index, bm25: BM25, k: int, depth: int | None = None) -> list[tuple[Hashable, float]]:
    """Fuse the dense top-depth with the BM25 top-depth; return the top k."""
    depth = depth or max(10, 2 * k)
    dense = [doc_id for doc_id, _ in dense_index.search(embed([query])[0], depth)]
    scores = bm25.score(tokenize(query))
    sparse = [bm25.ids[i] for i in np.argsort(-scores, kind="stable")[:depth] if scores[i] > 0]
    return reciprocal_rank_fusion([dense, sparse])[:k]
