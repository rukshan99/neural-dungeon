"""ROOM 13.2 - THE DRIFT GLASS

    A tall pane of smoked glass on the north wall. Hold yesterday's traffic
    behind it and today's in front, and where the two disagree the glass
    clouds over. It cannot tell you whether the model is wrong. It can tell
    you, today, that the world it was built for is not the world it is in.

Labels arrive late or never. Inputs arrive now. So the first thing a
watchtower measures is whether the DISTRIBUTION of what comes in still looks
like the distribution the model was validated on. Four instruments, all of
which compare a CURRENT sample with a fixed REFERENCE sample:

POPULATION STABILITY INDEX (numeric features). Cut the reference into
``n_bins`` quantile bins (each holds ~1/n of the reference), count the fraction
of the current sample in each bin, and sum

    PSI = sum_i (c_i - r_i) * ln(c_i / r_i)

Clip fractions to ``eps`` first so an empty bin does not produce ln(0). PSI is
zero when the fractions match and grows with the shift. Rule of thumb, from
credit scoring and still useful: < 0.1 stable, 0.1..0.25 moderate, >= 0.25
major. Always bin with the REFERENCE's edges, computed once; if you re-bin on
the current sample the yardstick moves with the thing you are measuring.

KOLMOGOROV-SMIRNOV DISTANCE (numeric, binning-free): the largest vertical gap
between the two empirical CDFs, max_x |F_a(x) - F_b(x)|. It lives in [0, 1]:
0 for identical samples, 1 when they do not overlap. Evaluate the CDFs on the
pooled sorted values; ``np.searchsorted(sorted_a, grid, side="right") / len(a)``
is F_a on the grid.

JENSEN-SHANNON DIVERGENCE (categorical: tool-call names, refusal vs answer,
intent labels). With m = (p + q) / 2,

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),   KL(p || m) = sum_i p_i ln(p_i / m_i)

Unlike KL it is symmetric and bounded: 0 for identical distributions, ln 2 for
disjoint ones. Terms with p_i = 0 contribute 0. Accept counts and normalise.

EMBEDDING DRIFT (vectors: query embeddings, response embeddings). Compare the
two clouds' centroids by cosine, and their mean norms:

    score = (1 - cos(centroid_ref, centroid_cur)) + |mean_norm_cur - mean_norm_ref| / mean_norm_ref

Rotation shows up in the first term, scaling in the second.

``DriftMonitor`` ties these together: it takes a reference (N, d), precomputes
each feature's edges, keeps the last ``window`` batches, and on ``observe``
reports PSI and KS per feature and which features crossed the thresholds. Its
default thresholds (PSI 0.25, KS 0.15) sit far above the sampling noise of a
few hundred requests; the boss will tune them.
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
    """``n_bins + 1`` edges: -inf, then the reference's quantiles at 1/n, 2/n, ..., (n-1)/n, then +inf.

    ``np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])`` gives the inner edges. Raise ValueError
    for an empty reference or n_bins < 1. The open ends guarantee every current value has a bin.
    """
    raise NotImplementedError("histogram_bins() is unwritten")


def bin_fractions(values, edges) -> np.ndarray:
    """Fraction of ``values`` falling in each bin; bin i is [edges[i], edges[i+1]). Sums to 1.

    ``np.searchsorted(edges[1:-1], values, side="right")`` is the bin index of every value;
    ``np.bincount(..., minlength=len(edges) - 1)`` counts them. Raise ValueError for an empty sample.
    """
    raise NotImplementedError("bin_fractions() is unwritten")


# ---------------------------------------------------------------------- psi


def psi(reference, current, edges=None, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of ``current`` against ``reference``. Python float.

    ``edges=None`` means ``histogram_bins(reference, n_bins)``. Clip both fraction vectors to at
    least ``eps`` before the log. A sample against itself is exactly 0.0.
    """
    raise NotImplementedError("psi() is unwritten")


def psi_verdict(value: float) -> str:
    """"stable" below PSI_MODERATE, "moderate" below PSI_MAJOR, else "major"."""
    raise NotImplementedError("psi_verdict() is unwritten")


# ----------------------------------------------------------------------- ks


def ks_statistic(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance in [0, 1]. Raise ValueError if either sample is empty."""
    raise NotImplementedError("ks_statistic() is unwritten")


# ----------------------------------------------------------------------- js


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence in nats of two categorical distributions of the same length.

    Accept probabilities or counts (normalise). Raise ValueError for mismatched lengths, empty input,
    negative entries or zero total mass. Bounded by ln 2; symmetric.
    """
    raise NotImplementedError("js_divergence() is unwritten")


def category_fractions(labels: Sequence, categories: Sequence) -> np.ndarray:
    """Fraction of ``labels`` equal to each category, in ``categories`` order.

    A label outside ``categories`` raises ValueError: a tool name you have never seen IS drift, and
    you want to hear about it rather than have it silently dropped. Build ``categories`` as the union
    of the reference's and the current sample's labels before calling this on both.
    """
    raise NotImplementedError("category_fractions() is unwritten")


# --------------------------------------------------------------- embeddings


def embedding_drift(ref_vectors, cur_vectors) -> float:
    """(1 - cosine(centroid_ref, centroid_cur)) + |mean_norm_cur - mean_norm_ref| / mean_norm_ref.

    Both inputs are (N, D) with the same D and N > 0, else ValueError. Clamp the cosine term at 0 so
    identical clouds give exactly 0.0 despite floating point; treat a zero centroid as cosine term 0.
    """
    raise NotImplementedError("embedding_drift() is unwritten")


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
        self.edges = [histogram_bins(ref[:, j], n_bins) for j in range(ref.shape[1])]  # once, from the reference
        self._recent: deque[np.ndarray] = deque(maxlen=window)
        self.history: list[dict] = []

    @property
    def n_features(self) -> int:
        return int(self.reference.shape[1])

    def observe(self, batch) -> dict:
        """Add a batch ((N,) or (N, d)), pool the last ``window`` batches, compare with the reference.

        Return and append to ``history``:
        {"n_current": rows pooled, "psi": [per feature], "ks": [per feature], "max_psi", "max_ks",
         "drifted_features": [indices where psi > thresholds["psi"] or ks > thresholds["ks"]],
         "drifted": bool(drifted_features)}
        Raise ValueError if the batch has the wrong number of features or no rows.
        """
        raise NotImplementedError("DriftMonitor.observe() is unwritten")
