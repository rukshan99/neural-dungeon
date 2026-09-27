"""ROOM 9.1 - THE ECHO METRIC  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Cosine similarity is a dot product of unit vectors, so the whole (N, M) matrix
is one matmul after normalising rows. Top-k selection uses argpartition (O(M))
and sorts only the k survivors.
"""

from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-L2 rows; a zero row stays zero instead of becoming NaN."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(N, D) x (M, D) -> (N, M). Normalise, then one matrix product."""
    return normalize_rows(a, eps) @ normalize_rows(b, eps).T


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-k, descending, ties by ascending index. argpartition + a small sort."""
    scores = np.asarray(scores)
    n, m = scores.shape
    k = min(k, m)
    if k < m:
        # argpartition puts the k largest of -scores... i.e. the k largest scores,
        # in the first k slots, in no particular order. O(M) per row.
        candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        candidates = np.broadcast_to(np.arange(m), (n, m)).copy()
    values = np.take_along_axis(scores, candidates, axis=1)
    # lexsort sorts by the LAST key first: descending value, then ascending index.
    order = np.lexsort((candidates, -values), axis=1)
    return np.take_along_axis(candidates, order, axis=1), np.take_along_axis(values, order, axis=1)


def nearest_neighbors(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(Q, D), (N, D) -> indices and cosine scores of the k nearest docs per query, (Q, k)."""
    return top_k(cosine_similarity(query_vecs, doc_vecs), k)
