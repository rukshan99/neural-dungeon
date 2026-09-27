"""ROOM 11.1 - THE SCALES  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Every metric here is a ratio of counts from the confusion matrix, plus the two
perplexity helpers, which are one exp away from a mean negative log-likelihood.
The zero-division conventions are the part people get wrong: a metric whose
denominator is zero is 0.0 here, never NaN, never an exception.
"""

from __future__ import annotations

import numpy as np


def _as_int_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    if t.shape != p.shape:
        raise ValueError(f"y_true has {t.shape[0]} labels and y_pred has {p.shape[0]}; they must pair up")
    return t, p


def _infer_classes(t: np.ndarray, p: np.ndarray, n_classes: int | None) -> int:
    if n_classes is not None:
        return int(n_classes)
    if t.size == 0:
        return 0
    return int(max(t.max(), p.max())) + 1


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) int64 matrix with cm[i, j] = #examples of true class i predicted as j."""
    t, p = _as_int_arrays(y_true, y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)  # one increment per (true, predicted) pair, duplicates included
    return cm


def accuracy(y_true, y_pred) -> float:
    """Fraction of labels predicted exactly right. The trace of the confusion matrix over its sum."""
    t, p = _as_int_arrays(y_true, y_pred)
    if t.size == 0:
        return 0.0
    return float(np.mean(t == p))


def precision_recall_f1(y_true, y_pred, positive: int = 1) -> tuple[float, float, float]:
    """Binary precision, recall and F1 for the class ``positive``.

    precision = TP / (TP + FP)   0.0 when nothing was predicted positive
    recall    = TP / (TP + FN)   0.0 when nothing was actually positive
    f1        = 2PR / (P + R)    0.0 when P + R == 0
    """
    t, p = _as_int_arrays(y_true, y_pred)
    is_true = t == positive
    is_pred = p == positive
    tp = int(np.sum(is_true & is_pred))
    fp = int(np.sum(~is_true & is_pred))
    fn = int(np.sum(is_true & ~is_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def macro_f1(y_true, y_pred, n_classes: int | None = None) -> float:
    """Mean over classes 0..n_classes-1 of the one-vs-rest F1. Every class weighs the same."""
    t, p = _as_int_arrays(y_true, y_pred)
    k = _infer_classes(t, p, n_classes)
    if k == 0:
        return 0.0
    return float(np.mean([precision_recall_f1(t, p, positive=c)[2] for c in range(k)]))


def balanced_accuracy(y_true, y_pred, n_classes: int | None = None) -> float:
    """Mean over classes of the per-class recall. A majority-class guesser scores 1 / n_classes."""
    t, p = _as_int_arrays(y_true, y_pred)
    k = _infer_classes(t, p, n_classes)
    if k == 0:
        return 0.0
    return float(np.mean([precision_recall_f1(t, p, positive=c)[1] for c in range(k)]))


def perplexity_from_nll(nll_values) -> float:
    """exp(mean(nll)). Average the log-losses FIRST, then exponentiate."""
    nll = np.asarray(nll_values, dtype=np.float64).ravel()
    if nll.size == 0:
        raise ValueError("perplexity of nothing is undefined")
    return float(np.exp(nll.mean()))


def token_level_perplexity(log_probs) -> float:
    """exp(-mean(log p(correct token))). log_probs are <= 0; a uniform model over V tokens gives V."""
    lp = np.asarray(log_probs, dtype=np.float64).ravel()
    if lp.size == 0:
        raise ValueError("perplexity of nothing is undefined")
    return float(np.exp(-lp.mean()))
