"""ROOM 9.2 - THE CARD CATALOGUE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

FlatIndex is the exact yardstick. KMeans is Lloyd's algorithm with k-means++
initialisation. IVFIndex clusters once, then probes only ``nprobe`` clusters per
query and counts how many vectors it actually scored.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from .room_1_echo_metric import normalize_rows, top_k


class FlatIndex:
    """Exact cosine search over everything stored."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[Hashable] = []
        self.comparisons = 0

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        vectors = normalize_rows(np.asarray(vectors, dtype=np.float64))
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"{vectors.shape[0]} vectors but {len(ids)} ids")
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = normalize_rows(np.asarray(query, dtype=np.float64)[None, :])
        scores = q @ self._vectors.T  # (1, N): unit vectors, so this IS cosine
        self.comparisons += self._vectors.shape[0]
        idx, vals = top_k(scores, k)
        return [(self._ids[i], float(s)) for i, s in zip(idx[0], vals[0])]

    def __len__(self) -> int:
        return len(self._ids)


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(N, D), (K, D) -> (N, K) squared Euclidean distances via |x|^2 + |c|^2 - 2 x.c."""
    d2 = (X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2.0 * X @ C.T
    return np.maximum(d2, 0.0)


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation."""

    def __init__(self, n_clusters: int, iters: int = 25, seed: int = 0) -> None:
        self.n_clusters = n_clusters
        self.iters = iters
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.inertia_history: list[float] = []

    def _init_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        centroids = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = _sq_dists(X, np.array(centroids)).min(axis=1)
            total = d2.sum()
            if total <= 0:  # every point already sits on a centroid
                centroids.append(X[rng.integers(n)])
                continue
            centroids.append(X[rng.choice(n, p=d2 / total)])
        return np.array(centroids, dtype=np.float64)

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"{X.shape[0]} points cannot form {self.n_clusters} clusters")
        rng = np.random.default_rng(self.seed)
        C = self._init_plus_plus(X, rng)
        self.inertia_history = []
        labels = None
        for _ in range(self.iters):
            d2 = _sq_dists(X, C)
            new_labels = d2.argmin(axis=1)
            self.inertia_history.append(float(d2[np.arange(X.shape[0]), new_labels].sum()))
            if labels is not None and np.array_equal(new_labels, labels):
                break  # converged: the update would change nothing
            labels = new_labels
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members):  # an empty cluster keeps its old centroid
                    C[j] = members.mean(axis=0)
        self.centroids = C
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("call fit() first")
        return _sq_dists(np.asarray(X, dtype=np.float64), self.centroids).argmin(axis=1)


class IVFIndex:
    """Inverted file: k-means shelves, probe only the nearest nprobe of them."""

    def __init__(self, n_clusters: int, nprobe: int, iters: int = 25, seed: int = 0) -> None:
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.iters = iters
        self.seed = seed
        self.kmeans: KMeans | None = None
        self._vectors: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
        self._ids: list[list[Hashable]] = [[] for _ in range(n_clusters)]
        self._matrices: list[np.ndarray | None] = [None] * n_clusters  # per-cluster stacks
        self.comparisons = 0

    def train(self, vectors: np.ndarray) -> None:
        X = normalize_rows(np.asarray(vectors, dtype=np.float64))
        self.kmeans = KMeans(self.n_clusters, self.iters, self.seed).fit(X)

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        if self.kmeans is None:
            raise RuntimeError("IVFIndex.add() before train(): there are no shelves yet")
        X = normalize_rows(np.asarray(vectors, dtype=np.float64))
        labels = self.kmeans.assign(X)
        for x, i, lab in zip(X, ids, labels):
            self._vectors[lab].append(x)
            self._ids[lab].append(i)
            self._matrices[lab] = None  # invalidate the cached stack

    def _matrix(self, cluster: int) -> np.ndarray:
        if self._matrices[cluster] is None:
            self._matrices[cluster] = np.array(self._vectors[cluster])
        return self._matrices[cluster]

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        if self.kmeans is None:
            raise RuntimeError("IVFIndex.search() before train()")
        q = normalize_rows(np.asarray(query, dtype=np.float64)[None, :])
        d2 = _sq_dists(q, self.kmeans.centroids)[0]
        probe = np.argsort(d2, kind="stable")[: self.nprobe]
        cand_ids: list[Hashable] = []
        cand_scores: list[np.ndarray] = []
        for c in probe:
            if not self._ids[c]:
                continue
            M = self._matrix(int(c))
            cand_scores.append((q @ M.T)[0])
            cand_ids.extend(self._ids[c])
            self.comparisons += M.shape[0]
        if not cand_ids:
            return []
        scores = np.concatenate(cand_scores)[None, :]
        idx, vals = top_k(scores, k)
        return [(cand_ids[i], float(s)) for i, s in zip(idx[0], vals[0])]


def _ids_of(results: Sequence, k: int) -> set:
    """First k ids of a result list, whether it holds ids or (id, score) pairs."""
    out = []
    for item in list(results)[:k]:
        out.append(item[0] if isinstance(item, tuple) else item)
    return set(out)


def recall_at_k(approx_results: Sequence[Sequence], exact_results: Sequence[Sequence], k: int) -> float:
    """Mean over queries of |approx top-k & exact top-k| / k."""
    if len(approx_results) != len(exact_results):
        raise ValueError("one result list per query, for both arguments")
    if not approx_results:
        return 0.0
    hits = [len(_ids_of(a, k) & _ids_of(e, k)) / k for a, e in zip(approx_results, exact_results)]
    return float(np.mean(hits))
