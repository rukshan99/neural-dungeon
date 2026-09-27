"""ROOM 9.2 - THE CARD CATALOGUE

    Ten thousand books, one question. You could walk every aisle and read every
    spine: that is exact search, and it always finds the right book. Or you
    could ask the catalogue which shelves are *about* your question and walk
    only those. Faster. Usually right.

Exact (flat) search compares the query with every stored vector: O(N) per query,
recall 1.0 by definition. It is the correct choice up to a few hundred thousand
vectors, and it is the yardstick everything else is measured against.

Approximate search trades a little recall for a lot of speed. The inverted-file
index (IVF) does it with clustering:

    train:  run k-means on the vectors -> n_clusters centroids ("shelves")
    add:    assign each vector to its nearest centroid; store it on that shelf
    search: find the nprobe centroids nearest the query; score ONLY the vectors
            on those shelves; return the k best

If the true neighbour sits on a shelf you did not probe, you miss it. Raising
``nprobe`` raises recall and cost together; ``nprobe == n_clusters`` is exact
search with extra steps. The trial measures both sides of that trade with
``recall_at_k`` and a ``comparisons`` counter.

k-means itself is Lloyd's algorithm: assign every point to its nearest centroid,
move every centroid to the mean of its points, repeat. Each step can only lower
the total squared distance (the *inertia*), so the trial checks it never climbs.
k-means++ initialisation (pick the next centre with probability proportional to
its squared distance from the centres so far) avoids the degenerate starts that
random picks sometimes produce.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from .room_1_echo_metric import normalize_rows, top_k  # noqa: F401  (you will want these)


class FlatIndex:
    """Exact search: every query is compared with every stored vector.

    - ``add(vectors, ids)``: ``vectors`` is (N, D), ``ids`` a sequence of N hashable
      ids (strings, ints, ...). Store L2-normalised copies. May be called repeatedly.
    - ``search(query, k)``: ``query`` is (D,). Return ``[(id, score), ...]``, the k
      highest cosine scores, best first (fewer if fewer vectors are stored).
    - ``comparisons``: running total of stored vectors scanned across all searches.
    - ``len(index)``: number of stored vectors.
    """

    def __init__(self) -> None:
        raise NotImplementedError("FlatIndex.__init__() is unwritten")

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        raise NotImplementedError("FlatIndex.add() is unwritten")

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        raise NotImplementedError("FlatIndex.search() is unwritten")

    def __len__(self) -> int:
        raise NotImplementedError("FlatIndex.__len__() is unwritten")


class KMeans:
    """Lloyd's k-means in numpy.

    ``KMeans(n_clusters, iters=25, seed=0)``

    - ``fit(X)``: ``X`` is (N, D). Sets ``self.centroids`` (n_clusters, D) and
      ``self.inertia_history``, a list with one float per iteration: the sum of
      squared Euclidean distances from every point to its assigned centroid,
      measured right after that iteration's assignment step. Returns ``self``.
    - ``assign(X)``: (N, D) -> (N,) int array, the index of the nearest centroid.

    Initialise with k-means++ (or n_clusters distinct random rows), drawing from
    ``np.random.default_rng(seed)``. Then at most ``iters`` times: assign, record
    inertia, move each centroid to the mean of its points. A cluster that loses
    all its points KEEPS its old centroid; re-seeding it randomly can make
    inertia climb, and the trial checks that it never does. You may stop early
    when the assignment stops changing.
    """

    def __init__(self, n_clusters: int, iters: int = 25, seed: int = 0) -> None:
        raise NotImplementedError("KMeans.__init__() is unwritten")

    def fit(self, X: np.ndarray) -> "KMeans":
        raise NotImplementedError("KMeans.fit() is unwritten")

    def assign(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("KMeans.assign() is unwritten")


class IVFIndex:
    """Inverted-file index: cluster the vectors, then search only the nearest clusters.

    ``IVFIndex(n_clusters, nprobe, iters=25, seed=0)``

    - ``train(vectors)``: L2-normalise, fit a ``KMeans(n_clusters, iters, seed)``,
      keep it. Must be called before ``add``.
    - ``add(vectors, ids)``: normalise, assign every vector to its nearest centroid
      and store vector + id on that cluster's list. ``RuntimeError`` if untrained.
    - ``search(query, k)``: normalise the query; pick the ``nprobe`` centroids
      nearest to it (Euclidean, the same metric ``assign`` uses); score ONLY the
      vectors in those clusters (dot product = cosine); return the k best as
      ``[(id, score), ...]``, best first. Add the number of vectors scored to
      ``self.comparisons``. Fewer than k results are fine when the probed
      clusters hold fewer than k vectors.
    - ``comparisons``: running total, like ``FlatIndex``.
    """

    def __init__(self, n_clusters: int, nprobe: int, iters: int = 25, seed: int = 0) -> None:
        raise NotImplementedError("IVFIndex.__init__() is unwritten")

    def train(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("IVFIndex.train() is unwritten")

    def add(self, vectors: np.ndarray, ids: Sequence[Hashable]) -> None:
        raise NotImplementedError("IVFIndex.add() is unwritten")

    def search(self, query: np.ndarray, k: int) -> list[tuple[Hashable, float]]:
        raise NotImplementedError("IVFIndex.search() is unwritten")


def recall_at_k(approx_results: Sequence[Sequence], exact_results: Sequence[Sequence], k: int) -> float:
    """How much of the exact top-k did the approximate search find? Mean over queries.

    Both arguments hold one result list per query, as ``search`` returns them:
    ``[(id, score), ...]`` best first. Plain lists of ids are accepted too. For
    each query take the first k ids of each list; recall is
    ``|approx & exact| / k``. Return the mean as a Python float in [0, 1].
    """
    raise NotImplementedError("recall_at_k() is unwritten")
