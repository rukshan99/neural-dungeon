"""SECRET - THE CALIBRATION MIRROR   (optional)

    Behind a loose stone in the arena wall: a tall mirror that shows every
    champion exactly as sure of itself as it deserves to be. Most of them look
    smaller in it.

Accuracy says how often a model is right. CALIBRATION says whether its
confidence means anything: among all the predictions made with 90% confidence,
about 90% should be right. A model can be accurate and badly calibrated (most
neural networks are over-confident), and a downstream system that acts on the
confidence, routing to a human, abstaining, ranking, will inherit the lie.

Expected calibration error bins predictions by confidence (bin b covers
(b/n, (b+1)/n], with bin 0 also taking confidence 0) and weights each bin's
gap by how full it is:

    ECE = sum_b (n_b / N) * | accuracy_b - mean_confidence_b |

Temperature scaling is the simplest fix: divide the logits by one scalar T,
chosen to minimise the mean negative log-likelihood on held-out data, before
the softmax. T > 1 softens the probabilities, T < 1 sharpens them, and no
argmax ever changes, so accuracy is untouched. Search a grid; it is one number.

Confidence = the largest softmax probability; correct = argmax(probabilities) == label.
"""

from __future__ import annotations

import numpy as np


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of logits / temperature over the last axis, numerically stable (subtract the row max)."""
    raise NotImplementedError("softmax() is unwritten")


def mean_nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean over rows of -log softmax(logits / temperature)[row, labels[row]]. Work in log space; do not
    take log(softmax(...)). Return a Python float."""
    raise NotImplementedError("mean_nll() is unwritten")


def reliability_table(confidences, correct, n_bins: int = 10) -> np.ndarray:
    """(n_bins, 3) float array: column 0 = count, 1 = mean confidence, 2 = accuracy, per bin.

    Empty bins have count 0 and NaN in the other two columns. Bin edges are
    np.linspace(0, 1, n_bins + 1); a confidence goes into the bin whose interval (lower, upper]
    contains it (``np.digitize(conf, edges[1:-1], right=True)`` does exactly this).
    """
    raise NotImplementedError("reliability_table() is unwritten")


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """sum over non-empty bins of (count / N) * |accuracy - mean confidence|. 0.0 for empty input."""
    raise NotImplementedError("expected_calibration_error() is unwritten")


def temperature_scale(logits, labels, grid) -> float:
    """The temperature in ``grid`` giving the smallest mean_nll(logits, labels, T). Return a Python float
    taken from the grid. Raise ValueError for an empty grid."""
    raise NotImplementedError("temperature_scale() is unwritten")
