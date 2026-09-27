"""ROOM 11.1 - THE SCALES

    A brass balance the height of a door, and a steward who weighs every
    champion that enters the arena. "Ninety-five percent," she says of the
    last one, and then, quieter, "which is to say: nothing."

Every classification metric is a ratio of counts from the confusion matrix.
Learn the counts and the metrics stop being vocabulary:

    cm[i, j] = number of examples whose TRUE class is i and PREDICTED class is j

For a binary task with positive class 1: TP = cm[1,1], FP = cm[0,1],
FN = cm[1,0], TN = cm[0,0].

    accuracy  = (TP + TN) / N            fraction right; lies when classes are unbalanced
    precision = TP / (TP + FP)           of what you flagged, how much was real
    recall    = TP / (TP + FN)           of what was real, how much you flagged
    F1        = 2 P R / (P + R)          harmonic mean; punishes a lopsided pair
    macro-F1  = mean over classes of the one-vs-rest F1   (every class counts equally)
    balanced accuracy = mean over classes of the per-class recall

Zero-division conventions (the trial checks them; return 0.0, never NaN, never raise):
    precision with no predicted positives = 0.0
    recall with no actual positives = 0.0
    F1 when P + R == 0 = 0.0

Perplexity is the metric for language models. The training loss is the mean
negative log-likelihood (NLL) of the correct next token; perplexity is exp of
that mean. Average the log-losses FIRST and exponentiate once. exp(mean(nll))
is not mean(exp(nll)), and the second one is wrong. A model that spreads its
probability uniformly over V tokens has perplexity exactly V: it is "as confused
as a V-sided die".

Accept lists or arrays for every label argument. Return Python floats.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) integer matrix; rows are TRUE classes, columns PREDICTED classes.

    cm[i, j] counts examples with true label i and predicted label j. Labels are
    integers in 0..n_classes-1. Raise ValueError if y_true and y_pred differ in length.
    (``np.add.at`` handles repeated (i, j) pairs; a loop is also fine here.)
    """
    raise NotImplementedError("confusion_matrix() is unwritten")


def accuracy(y_true, y_pred) -> float:
    """Fraction of predictions that equal the truth. Return a Python float."""
    raise NotImplementedError("accuracy() is unwritten")


def precision_recall_f1(y_true, y_pred, positive: int = 1) -> tuple[float, float, float]:
    """Binary (precision, recall, f1) treating ``positive`` as the positive class.

    Apply the zero-division conventions from the module docstring. Return three
    Python floats.
    """
    raise NotImplementedError("precision_recall_f1() is unwritten")


def macro_f1(y_true, y_pred, n_classes: int | None = None) -> float:
    """Mean of the one-vs-rest F1 over classes 0..n_classes-1.

    ``n_classes=None`` means infer it as max(label) + 1. Every class counts once,
    however rare it is; a class with no true and no predicted examples has F1 = 0.0
    by the conventions above, so pass ``n_classes`` deliberately.
    """
    raise NotImplementedError("macro_f1() is unwritten")


def balanced_accuracy(y_true, y_pred, n_classes: int | None = None) -> float:
    """Mean of the per-class recall over classes 0..n_classes-1 (same inference rule as macro_f1).

    A predictor that always says the majority class scores exactly 1 / n_classes here,
    whatever its accuracy.
    """
    raise NotImplementedError("balanced_accuracy() is unwritten")


def perplexity_from_nll(nll_values) -> float:
    """exp(mean(nll_values)). ``nll_values`` are per-token (or per-example) negative log-likelihoods, >= 0.

    Raise ValueError for an empty input. Return a Python float.
    """
    raise NotImplementedError("perplexity_from_nll() is unwritten")


def token_level_perplexity(log_probs) -> float:
    """exp(-mean(log_probs)) where ``log_probs`` are the model's log-probabilities of the CORRECT tokens (<= 0).

    A uniform model over V tokens gives log p = -log V everywhere, so the answer is V.
    Raise ValueError for an empty input. Return a Python float.
    """
    raise NotImplementedError("token_level_perplexity() is unwritten")
