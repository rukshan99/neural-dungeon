"""ROOM 13.2 - THE DRIFT GLASS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every metric here compares a CURRENT sample with a REFERENCE sample. PSI and
KS work on one numeric feature at a time, JS on a categorical distribution,
and the embedding score on a cloud of vectors. The monitor precomputes the
reference's bin edges once and re-uses them, so the current batch is always
measured against the same yardstick.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
DEFAULT_THRESHOLDS = {"psi": PSI_MAJOR, "ks": 0.15}


# --------------------------------------------------------------- histograms


def histogram_bins(reference, n_bins: int = 10) -> np.ndarray:
    """``n_bins + 1`` edges: -inf, the reference's 1/n .. (n-1)/n quantiles, +inf."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if ref.size == 0:
        raise ValueError("the reference sample is empty")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    inner = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_fractions(values, edges) -> np.ndarray:
    """Fraction of ``values`` in each bin; bin i is [edges[i], edges[i+1])."""
    x = np.asarray(values, dtype=np.float64).ravel()
    edges = np.asarray(edges, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot bin an empty sample")
    idx = np.searchsorted(edges[1:-1], x, side="right")
    counts = np.bincount(idx, minlength=len(edges) - 1)
    return counts / x.size


# ---------------------------------------------------------------------- psi


def psi(reference, current, edges=None, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index: sum_i (c_i - r_i) * ln(c_i / r_i), fractions clipped to >= eps."""
    if edges is None:
        edges = histogram_bins(reference, n_bins)
    r = np.clip(bin_fractions(reference, edges), eps, None)
    c = np.clip(bin_fractions(current, edges), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_verdict(value: float) -> str:
    """The rule of thumb: < 0.1 stable, 0.1..0.25 moderate, >= 0.25 major."""
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_MAJOR:
        return "moderate"
    return "major"


# ----------------------------------------------------------------------- ks


def ks_statistic(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance: max over x of |F_a(x) - F_b(x)|."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.concatenate([a, b])
    f_a = np.searchsorted(a, grid, side="right") / a.size
    f_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(f_a - f_b)))


# ----------------------------------------------------------------------- js


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence (nats) of two categorical distributions given as probabilities or counts."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape or p.size == 0:
        raise ValueError("p and q must be non-empty and the same length")
    if np.any(p < 0) or np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("p and q must be non-negative with positive mass")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def category_fractions(labels: Sequence, categories: Sequence) -> np.ndarray:
    """Fraction of ``labels`` equal to each category, in ``categories`` order. Unknown labels raise."""
    index = {category: i for i, category in enumerate(categories)}
    counts = np.zeros(len(categories), dtype=np.float64)
    for label in labels:
        if label not in index:
            raise ValueError(f"label {label!r} is not one of the categories {list(categories)}")
        counts[index[label]] += 1
    if len(labels) == 0:
        raise ValueError("no labels")
    return counts / len(labels)


# --------------------------------------------------------------- embeddings


def embedding_drift(ref_vectors, cur_vectors) -> float:
    """(1 - cosine of the two centroids) + |mean norm change| / reference mean norm."""
    ref = np.asarray(ref_vectors, dtype=np.float64)
    cur = np.asarray(cur_vectors, dtype=np.float64)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError("expected two (N, D) arrays with the same D")
    if ref.shape[0] == 0 or cur.shape[0] == 0:
        raise ValueError("both clouds must be non-empty")
    centroid_ref, centroid_cur = ref.mean(axis=0), cur.mean(axis=0)
    denom = np.linalg.norm(centroid_ref) * np.linalg.norm(centroid_cur)
    cosine_distance = max(0.0, 1.0 - float(centroid_ref @ centroid_cur) / denom) if denom > 0 else 0.0
    norm_ref = float(np.linalg.norm(ref, axis=1).mean())
    norm_cur = float(np.linalg.norm(cur, axis=1).mean())
    norm_change = abs(norm_cur - norm_ref) / norm_ref if norm_ref > 0 else 0.0
    return float(cosine_distance + norm_change)


# ------------------------------------------------------------------ monitor


class DriftMonitor:
    """Per-feature PSI and KS of the last ``window`` batches against a fixed reference."""

    def __init__(self, reference, window: int = 1, n_bins: int = 10, thresholds: dict | None = None) -> None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref[:, None]
        if ref.ndim != 2 or ref.shape[0] == 0:
            raise ValueError("reference must be (N,) or (N, d) with N > 0")
        if window < 1:
            raise ValueError("window must be at least 1")
        self.reference = ref
        self.window = window
        self.n_bins = n_bins
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.edges = [histogram_bins(ref[:, j], n_bins) for j in range(ref.shape[1])]
        self._recent: deque[np.ndarray] = deque(maxlen=window)
        self.history: list[dict] = []

    @property
    def n_features(self) -> int:
        return int(self.reference.shape[1])

    def observe(self, batch) -> dict:
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] != self.n_features or x.shape[0] == 0:
            raise ValueError(f"batch must be (N, {self.n_features}) with N > 0, got {x.shape}")
        self._recent.append(x)
        current = np.concatenate(list(self._recent), axis=0)
        psi_values = [psi(self.reference[:, j], current[:, j], self.edges[j]) for j in range(self.n_features)]
        ks_values = [ks_statistic(self.reference[:, j], current[:, j]) for j in range(self.n_features)]
        drifted = [
            j
            for j in range(self.n_features)
            if psi_values[j] > self.thresholds["psi"] or ks_values[j] > self.thresholds["ks"]
        ]
        report = {
            "n_current": int(current.shape[0]),
            "psi": psi_values,
            "ks": ks_values,
            "max_psi": max(psi_values),
            "max_ks": max(ks_values),
            "drifted_features": drifted,
            "drifted": bool(drifted),
        }
        self.history.append(report)
        return report
