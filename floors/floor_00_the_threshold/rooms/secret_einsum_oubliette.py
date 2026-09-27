"""SECRET - THE EINSUM OUBLIETTE   (optional)

    A trapdoor under the Basilisk's lair drops you into a stone cell. Scratched
    into the wall, over and over: "bij,bjk->bik". There is exactly one spell
    that works down here.

``np.einsum`` names every axis with a letter and lets you say what to multiply
and what to sum. Repeated letters across inputs are multiplied and (if absent
from the output) summed. Letters only in the output are kept.

    "ij,jk->ik"      matrix product
    "bij,bjk->bik"   batched matrix product
    "ii->"           trace (repeated letter on ONE input = diagonal)
    "i,j->ij"        outer product
    "bqd,bkd->bqk"   every query against every key: attention scores

Rules of the cell: each function body must be ONE einsum call. The trial reads
your source and refuses matmul, dot, @, trace, diagonal, outer, sum and friends.
"""

from __future__ import annotations

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(B, I, J), (B, J, K) -> (B, I, K)."""
    raise NotImplementedError("batched_matmul() is unwritten")


def trace_of_each(a: np.ndarray) -> np.ndarray:
    """(B, N, N) -> (B,) the trace of every matrix in the batch."""
    raise NotImplementedError("trace_of_each() is unwritten")


def diagonal_of_each(a: np.ndarray) -> np.ndarray:
    """(B, N, N) -> (B, N) the diagonal of every matrix in the batch."""
    raise NotImplementedError("diagonal_of_each() is unwritten")


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,), (M,) -> (N, M) outer product."""
    raise NotImplementedError("outer() is unwritten")


def bilinear(x: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(B, I), (I, J), (B, J) -> (B,) with out[b] = x[b] @ W @ y[b]. One einsum, three operands."""
    raise NotImplementedError("bilinear() is unwritten")


def attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(B, Q, D), (B, K, D) -> (B, Q, K) dot product of every query with every key."""
    raise NotImplementedError("attention_scores() is unwritten")


def row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D), (N, D) -> (N,) dot product of matching rows."""
    raise NotImplementedError("row_dot() is unwritten")
