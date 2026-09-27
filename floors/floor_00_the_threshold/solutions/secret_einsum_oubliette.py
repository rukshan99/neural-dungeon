"""SECRET - THE EINSUM OUBLIETTE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.
Every function body is a single np.einsum call. Nothing else is permitted.
"""

from __future__ import annotations

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(B, I, J) @ (B, J, K) -> (B, I, K)."""
    return np.einsum("bij,bjk->bik", a, b)


def trace_of_each(a: np.ndarray) -> np.ndarray:
    """(B, N, N) -> (B,) traces."""
    return np.einsum("bii->b", a)


def diagonal_of_each(a: np.ndarray) -> np.ndarray:
    """(B, N, N) -> (B, N) diagonals."""
    return np.einsum("bii->bi", a)


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,), (M,) -> (N, M) outer product."""
    return np.einsum("i,j->ij", a, b)


def bilinear(x: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(B, I), (I, J), (B, J) -> (B,) where out[b] = x[b] @ W @ y[b]."""
    return np.einsum("bi,ij,bj->b", x, W, y)


def attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(B, Q, D), (B, K, D) -> (B, Q, K) dot products between every query and every key."""
    return np.einsum("bqd,bkd->bqk", q, k)


def row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D), (N, D) -> (N,) dot product of matching rows."""
    return np.einsum("nd,nd->n", a, b)
