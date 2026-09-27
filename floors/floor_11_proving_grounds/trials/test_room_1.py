"""TRIAL 11.1 - THE SCALES

The steward weighs your metrics against counts done by hand, then hands you a
champion with 95% accuracy and asks whether you are impressed.
"""

import math

import numpy as np
import pytest

from dungeon.trials import load_room
from floors.floor_11_proving_grounds.assets.champions import MajorityClassifier, imbalanced_task

room = load_room(__file__, "room_1_the_scales")

# A six-example, three-class hand case. True on the rows, predicted on the columns:
#     pred:  0  1  2
# true 0:   [2, 0, 0]
# true 1:   [1, 0, 1]
# true 2:   [0, 0, 2]
Y3_TRUE = [0, 1, 2, 2, 1, 0]
Y3_PRED = [0, 2, 2, 2, 0, 0]
CM3 = np.array([[2, 0, 0], [1, 0, 1], [0, 0, 2]])

# Binary hand case: TP=2, FP=1, FN=2, TN=3.
YB_TRUE = [1, 1, 1, 0, 0, 0, 0, 1]
YB_PRED = [1, 0, 1, 1, 0, 0, 0, 0]


# ----------------------------------------------------------- confusion matrix
def test_the_confusion_matrix_puts_truth_on_rows_and_predictions_on_columns():
    cm = room.confusion_matrix(Y3_TRUE, Y3_PRED, 3)
    assert isinstance(cm, np.ndarray) and cm.shape == (3, 3), f"Expected a (3, 3) array, got {getattr(cm, 'shape', type(cm))}."
    assert np.issubdtype(cm.dtype, np.integer), f"Counts are integers; your matrix is {cm.dtype}."
    assert np.array_equal(cm, CM3), (
        f"Expected\n{CM3}\nbut got\n{cm}\ncm[i, j] counts examples with TRUE class i predicted as j. "
        "If yours is the transpose, you have rows and columns swapped."
    )


def test_the_confusion_matrix_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        room.confusion_matrix([0, 1, 1], [0, 1], 2)


def test_the_confusion_matrix_counts_repeated_pairs():
    cm = room.confusion_matrix([1, 1, 1, 1], [1, 1, 1, 0], 2)
    assert cm[1, 1] == 3 and cm[1, 0] == 1, (
        f"Three examples were (true 1, pred 1) and one was (true 1, pred 0); got {cm.tolist()}. "
        "Fancy-index assignment `cm[t, p] += 1` drops duplicates; use np.add.at or a loop."
    )


# ------------------------------------------------------------------ accuracy
def test_accuracy_is_the_fraction_right():
    acc = room.accuracy(Y3_TRUE, Y3_PRED)
    assert isinstance(acc, float), f"Return a Python float, not {type(acc).__name__}."
    assert math.isclose(acc, 4 / 6), f"4 of 6 are right, so accuracy is {4 / 6:.4f}; you said {acc:.4f}."


# ----------------------------------------------------------- precision/recall
def test_precision_recall_and_f1_match_the_hand_count():
    p, r, f1 = room.precision_recall_f1(YB_TRUE, YB_PRED, positive=1)
    assert math.isclose(p, 2 / 3), f"TP=2, FP=1: precision is 2/3 = {2 / 3:.4f}; you said {p:.4f}."
    assert math.isclose(r, 0.5), f"TP=2, FN=2: recall is 2/4 = 0.5; you said {r:.4f}."
    assert math.isclose(f1, 4 / 7), f"F1 = 2PR/(P+R) = {4 / 7:.4f}; you said {f1:.4f}."


def test_the_positive_class_can_be_zero():
    p, r, f1 = room.precision_recall_f1(YB_TRUE, YB_PRED, positive=0)
    # For class 0: TP=3, FP=2, FN=1.
    assert math.isclose(p, 3 / 5) and math.isclose(r, 3 / 4), (
        f"With positive=0: TP=3, FP=2, FN=1, so P=0.6, R=0.75; you said P={p:.4f}, R={r:.4f}."
    )


def test_a_champion_who_never_says_yes_has_zero_precision_not_a_crash():
    p, r, f1 = room.precision_recall_f1([1, 0, 1, 0], [0, 0, 0, 0])
    assert (p, r, f1) == (0.0, 0.0, 0.0), (
        f"No predicted positives: precision is 0.0 by convention (not NaN, not an error). Got {(p, r, f1)}."
    )
    assert not any(math.isnan(v) for v in (p, r, f1)), "NaN leaked out of a zero-division."


def test_recall_with_no_actual_positives_is_zero():
    p, r, f1 = room.precision_recall_f1([0, 0, 0], [1, 0, 1])
    assert r == 0.0 and p == 0.0 and f1 == 0.0, (
        f"Nothing was actually positive: recall is 0.0 by convention. Two false alarms: precision 0.0. Got {(p, r, f1)}."
    )


# --------------------------------------------------------- macro / balanced
def test_macro_f1_weighs_every_class_equally():
    # Per-class one-vs-rest F1 on the 3-class hand case: class 0: P=2/3, R=1 -> 0.8; class 1: 0; class 2: P=2/3, R=1 -> 0.8.
    expected = (0.8 + 0.0 + 0.8) / 3
    got = room.macro_f1(Y3_TRUE, Y3_PRED, n_classes=3)
    assert math.isclose(got, expected), f"Per-class F1 is [0.8, 0.0, 0.8]; the macro mean is {expected:.4f}. You said {got:.4f}."


def test_balanced_accuracy_is_the_mean_recall():
    # Recall per class: 2/2, 0/2, 2/2.
    got = room.balanced_accuracy(Y3_TRUE, Y3_PRED, n_classes=3)
    assert math.isclose(got, 2 / 3), f"Per-class recall is [1, 0, 1]; balanced accuracy is {2 / 3:.4f}. You said {got:.4f}."


def test_the_class_count_is_inferred_when_not_given():
    assert math.isclose(room.balanced_accuracy(Y3_TRUE, Y3_PRED), 2 / 3), "n_classes=None should mean max(label) + 1."
    assert math.isclose(room.macro_f1(Y3_TRUE, Y3_PRED), (0.8 + 0.0 + 0.8) / 3)


def test_ninety_five_percent_accuracy_and_nothing_learned():
    y = imbalanced_task(n=1000, positive_rate=0.05)
    y_hat = MajorityClassifier().fit(y).predict(len(y))
    acc = room.accuracy(y, y_hat)
    bal = room.balanced_accuracy(y, y_hat, n_classes=2)
    mf1 = room.macro_f1(y, y_hat, n_classes=2)
    assert math.isclose(acc, 0.95), f"The majority guesser should score 0.95 accuracy; you computed {acc:.4f}."
    assert math.isclose(bal, 0.5), (
        f"Its balanced accuracy must be exactly 0.5 (recall 1.0 on the majority, 0.0 on the minority); you computed {bal:.4f}. "
        "This is the number that tells the truth."
    )
    assert mf1 < 0.5, f"Macro-F1 should be under 0.5 (the minority class has F1 = 0); you computed {mf1:.4f}."
    assert acc - bal > 0.4, "Accuracy and balanced accuracy should disagree by a mile here. Read both, always."


# --------------------------------------------------------------- perplexity
def test_perplexity_of_a_certain_model_is_one():
    assert math.isclose(room.perplexity_from_nll([0.0, 0.0, 0.0]), 1.0), "Zero loss everywhere is perplexity 1: one choice, no confusion."


def test_perplexity_is_exp_of_the_mean_nll():
    ppl = room.perplexity_from_nll([math.log(2), math.log(8)])
    assert isinstance(ppl, float), f"Return a Python float, not {type(ppl).__name__}."
    assert math.isclose(ppl, 4.0), f"exp(mean(ln 2, ln 8)) = exp(ln 16 / 2) = 4; you said {ppl:.4f}."


def test_perplexity_is_not_the_mean_of_perplexities():
    ppl = room.perplexity_from_nll([0.0, math.log(100)])
    assert math.isclose(ppl, 10.0), (
        f"exp(mean(0, ln 100)) = 10. You said {ppl:.3f}. If you got 50.5 you averaged exp(nll) instead of "
        "exponentiating the average. Perplexity is exp(mean NLL), and Jensen's inequality says the two differ."
    )


def test_a_uniform_model_over_v_tokens_has_perplexity_v():
    V = 512
    lp = np.full(300, -math.log(V))
    ppl = room.token_level_perplexity(lp)
    assert math.isclose(ppl, V, rel_tol=1e-9), f"Uniform over {V} tokens is perplexity {V}: as confused as a {V}-sided die. You said {ppl:.3f}."


def test_token_perplexity_uses_log_probs_of_the_correct_tokens():
    lp = np.log(np.array([0.5, 0.25, 0.125]))
    ppl = room.token_level_perplexity(lp)
    expected = math.exp(-lp.mean())
    assert math.isclose(ppl, expected), f"exp(-mean(log p)) = {expected:.4f}; you said {ppl:.4f}. Mind the sign: log-probs are negative."


def test_perplexity_of_nothing_is_refused():
    with pytest.raises(ValueError):
        room.perplexity_from_nll([])
    with pytest.raises(ValueError):
        room.token_level_perplexity([])
