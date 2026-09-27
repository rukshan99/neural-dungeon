"""The Vector Index (portable) - loot from Floor 9 of the Neural Dungeon.

Clean, dependency-free (numpy only) implementations of the retrieval building
blocks from the Library of Echoes, packaged to be copied into a project:

    FlatIndex               exact cosine search; the yardstick
    KMeans                  Lloyd's algorithm with k-means++ initialisation
    IVFIndex                inverted-file approximate search (cluster, then probe)
    BM25                    sparse lexical ranking
    reciprocal_rank_fusion  merge rankings without comparing their scores
    recall_at_k             how much of the exact top-k an approximate search found

Every index takes (N, D) float arrays and a parallel sequence of ids, and
returns ``[(id, score), ...]`` best first. Vectors are L2-normalised on insert,
so "score" is cosine similarity everywhere.

Usage:

    index = FlatIndex()
    index.add(vectors, ids)                 # (N, D) array, N ids
    hits = index.search(query_vector, k=5)  # [(id, cosine), ...]

    ivf = IVFIndex(n_clusters=64, nprobe=4)
    ivf.train(vectors)                      # once, on a representative sample
    ivf.add(vectors, ids)
    hits = ivf.search(query_vector, k=5)

    print(recall_at_k([ivf.search(q, 10) for q in Q], [index.search(q, 10) for q in Q], 10))

Run this file directly for a small self-check.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np

# ------------------------------------------------------------------ helpers


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-L2 rows; zero rows stay zero instead of becoming NaN."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row of (N, M): indices and values of the k largest, descending, ties by index. O(M) selection."""
    scores = np.asarray(scores)
    n, m = scores.shape
    k = min(k, m)
    if k < m:
        cand = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        cand = np.broadcast_to(np.arange(m), (n, m)).copy()
    vals = np.take_along_axis(scores, cand, axis=1)
    order = np.lexsort((cand, -vals), axis=1)
    return np.take_along_axis(cand, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(N, D), (K, D) -> (N, K) squared Euclidean distances without an (N, K, D) tensor."""
    d2 = (X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2.0 * X @ C.T
    return np.maximum(d2, 0.0)


# ---------------------------------------------------------------- FlatIndex


class FlatIndex:
    """Exact cosine search. Correct up to a few hundred thousand vectors; the reference for everything else."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[Hashable] = []
        self.comparisons = 0

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        vectors = normalize_rows(vectors)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"{vectors.shape[0]} vectors but {len(ids)} ids")
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        if self._vectors is None:
            return []
        q = normalize_rows(np.asarray(query, dtype=np.float64)[None, :])
        self.comparisons += self._vectors.shape[0]
        idx, vals = top_k(q @ self._vectors.T, k)
        return [(self._ids[i], float(s)) for i, s in zip(idx[0], vals[0])]

    def __len__(self) -> int:
        return len(self._ids)


# ------------------------------------------------------------------- KMeans


class KMeans:
    """Lloyd's algorithm, k-means++ init, empty clusters keep their centroid (inertia never climbs)."""

    def __init__(self, n_clusters: int, iters: int = 25, seed: int = 0) -> None:
        self.n_clusters, self.iters, self.seed = n_clusters, iters, seed
        self.centroids: np.ndarray | None = None
        self.inertia_history: list[float] = []

    def _init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        centroids = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = _sq_dists(X, np.array(centroids)).min(axis=1)
            total = d2.sum()
            centroids.append(X[rng.choice(n, p=d2 / total)] if total > 0 else X[rng.integers(n)])
        return np.array(centroids, dtype=np.float64)

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"{X.shape[0]} points cannot form {self.n_clusters} clusters")
        rng = np.random.default_rng(self.seed)
        C = self._init(X, rng)
        self.inertia_history = []
        labels = None
        for _ in range(self.iters):
            d2 = _sq_dists(X, C)
            new_labels = d2.argmin(axis=1)
            self.inertia_history.append(float(d2[np.arange(X.shape[0]), new_labels].sum()))
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members):
                    C[j] = members.mean(axis=0)
        self.centroids = C
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("call fit() first")
        return _sq_dists(np.asarray(X, dtype=np.float64), self.centroids).argmin(axis=1)


# ----------------------------------------------------------------- IVFIndex


class IVFIndex:
    """Inverted file: k-means shelves, probe only the nprobe nearest per query.

    Cost per query is about n_clusters + (nprobe / n_clusters) * N comparisons.
    Recall drops when a true neighbour sits on an unprobed shelf; raise nprobe to pay for it.
    """

    def __init__(self, n_clusters: int, nprobe: int, iters: int = 25, seed: int = 0) -> None:
        self.n_clusters, self.nprobe, self.iters, self.seed = n_clusters, nprobe, iters, seed
        self.kmeans: KMeans | None = None
        self._vectors: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
        self._ids: list[list[Hashable]] = [[] for _ in range(n_clusters)]
        self._stacks: list[np.ndarray | None] = [None] * n_clusters
        self.comparisons = 0

    def train(self, vectors: np.ndarray) -> None:
        self.kmeans = KMeans(self.n_clusters, self.iters, self.seed).fit(normalize_rows(vectors))

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        if self.kmeans is None:
            raise RuntimeError("train() before add()")
        X = normalize_rows(vectors)
        for x, i, lab in zip(X, ids, self.kmeans.assign(X)):
            self._vectors[lab].append(x)
            self._ids[lab].append(i)
            self._stacks[lab] = None

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        if self.kmeans is None:
            raise RuntimeError("train() before search()")
        q = normalize_rows(np.asarray(query, dtype=np.float64)[None, :])
        probe = np.argsort(_sq_dists(q, self.kmeans.centroids)[0], kind="stable")[: self.nprobe]
        ids: list[Hashable] = []
        scores: list[np.ndarray] = []
        for c in probe:
            if not self._ids[c]:
                continue
            if self._stacks[c] is None:
                self._stacks[c] = np.array(self._vectors[c])
            M = self._stacks[c]
            scores.append((q @ M.T)[0])
            ids.extend(self._ids[c])
            self.comparisons += M.shape[0]
        if not ids:
            return []
        idx, vals = top_k(np.concatenate(scores)[None, :], k)
        return [(ids[i], float(s)) for i, s in zip(idx[0], vals[0])]


# --------------------------------------------------------------------- BM25


class BM25:
    """Okapi BM25 over a tokenised corpus. score(query_tokens) -> (N,) array, one per document."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[Hashable] = []
        self._tf: list[Counter] = []
        self._lengths = np.zeros(0)
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def fit(self, tokenized_docs: Sequence[Sequence[str]], ids: Sequence[Hashable] | None = None) -> BM25:
        docs = [list(d) for d in tokenized_docs]
        self.ids = list(ids) if ids is not None else list(range(len(docs)))
        self._tf = [Counter(d) for d in docs]
        self._lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self._df = Counter(t for tf in self._tf for t in tf)
        self._avgdl = float(self._lengths.mean()) if docs else 0.0
        return self

    def idf(self, term: str) -> float:
        n, df = len(self._tf), self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        n = len(self._tf)
        out = np.zeros(n)
        if n == 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self._lengths / self._avgdl)
        for term in query_tokens:
            if term not in self._df:
                continue
            tf = np.array([c.get(term, 0) for c in self._tf], dtype=np.float64)
            out += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def search(self, query_tokens: Sequence[str], k: int) -> list[tuple[Hashable, float]]:
        scores = self.score(query_tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]


# ---------------------------------------------------------------- fusion / eval


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    """sum(1 / (k + rank)) per id across rankings, rank from 1. Best first; ties by str(id)."""
    fused: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], str(kv[0])))


def recall_at_k(approx_results: Sequence[Sequence], exact_results: Sequence[Sequence], k: int) -> float:
    """Mean over queries of |approx top-k ids & exact top-k ids| / k. Accepts (id, score) pairs or bare ids."""

    def ids(results):
        return {item[0] if isinstance(item, tuple) else item for item in list(results)[:k]}

    if not approx_results:
        return 0.0
    return float(np.mean([len(ids(a) & ids(e)) / k for a, e in zip(approx_results, exact_results)]))


# ------------------------------------------------------------------ self-check

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    centers = normalize_rows(rng.standard_normal((16, 32)))
    labels = rng.integers(0, 16, 4000)
    X = normalize_rows(centers[labels] + 0.2 * rng.standard_normal((4000, 32)))
    Q = normalize_rows(centers[rng.integers(0, 16, 100)] + 0.2 * rng.standard_normal((100, 32)))
    ids = [f"doc_{i}" for i in range(4000)]

    flat = FlatIndex()
    flat.add(X, ids)
    exact = [flat.search(q, 10) for q in Q]
    for nprobe in (1, 2, 4, 16):
        ivf = IVFIndex(n_clusters=16, nprobe=nprobe)
        ivf.train(X)
        ivf.add(X, ids)
        approx = [ivf.search(q, 10) for q in Q]
        print(f"IVF nprobe={nprobe:2d}: recall@10={recall_at_k(approx, exact, 10):.3f}  "
              f"comparisons={ivf.comparisons / flat.comparisons:.1%} of flat")

    docs = [["the", "basilisk", "waits"], ["the", "lich", "decays", "slowly"], ["a", "basilisk", "and", "a", "lich"]]
    bm25 = BM25().fit(docs, ["a", "b", "c"])
    print("BM25 'basilisk':", [(i, round(s, 3)) for i, s in bm25.search(["basilisk"], 3)])
    print("RRF:", reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])[:3])
