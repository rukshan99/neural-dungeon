"""BOSS - THE GOODHART GORGON  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The Gorgon's stare turns a metric into a target. The defence is never one
number: report accuracy NEXT TO balanced accuracy, judge on length-controlled
inputs AND verify with a swap, score the golden set AND a held-out set, and
let the gap between paired numbers be the thing you read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dungeon.artifacts.llm import LLM
from floors.floor_11_proving_grounds.assets.champions import FILLER_WORDS

from .room_1_the_scales import accuracy, balanced_accuracy, confusion_matrix, macro_f1
from .room_3_the_judge import pairwise_judge_debiased, pairwise_judge_naive
from .room_4_golden_trials import EvalCase, EvalSuite, Scorer

LYING_GAP = 0.2
CONTAMINATION_GAP = 0.3


# ------------------------------------------------------------------ phase 1


def robust_classification_report(y_true, y_pred) -> dict:
    """accuracy, balanced_accuracy, macro_f1, per_class_recall and a verdict that calls out a lie."""
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    n_classes = int(max(t.max(), p.max())) + 1
    cm = confusion_matrix(t, p, n_classes)
    support = cm.sum(axis=1)
    recall = [float(cm[c, c] / support[c]) if support[c] else 0.0 for c in range(n_classes)]
    acc = accuracy(t, p)
    bal = balanced_accuracy(t, p, n_classes)
    gap = acc - bal
    if gap > LYING_GAP:
        verdict = (
            f"accuracy is lying: {acc:.3f} accuracy but {bal:.3f} balanced accuracy "
            f"(gap {gap:.3f}); the minority class is being ignored"
        )
    else:
        verdict = f"the scales agree: accuracy {acc:.3f}, balanced accuracy {bal:.3f} (gap {gap:.3f})"
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": macro_f1(t, p, n_classes),
        "per_class_recall": recall,
        "verdict": verdict,
    }


# ------------------------------------------------------------------ phase 2

_WORD = re.compile(r"[a-z']+")


def filler_ratio(text: str) -> float:
    """Fraction of words (lowercased, letters only) that are in FILLER_WORDS. 0.0 for no words."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in FILLER_WORDS) / len(words)


def truncate_to_words(text: str, n_words: int) -> str:
    """The first ``n_words`` whitespace-separated words of ``text``."""
    return " ".join(text.split()[: max(0, n_words)])


def length_controlled_judge(llm: LLM, question: str, a: str, b: str) -> str:
    """Truncate both candidates to the shorter one's word count, then judge with a swap check."""
    n = min(len(a.split()), len(b.split()))
    return pairwise_judge_debiased(llm, question, truncate_to_words(a, n), truncate_to_words(b, n))


# ------------------------------------------------------------------ phase 3


def split_golden_heldout(
    cases: Sequence[EvalCase], heldout_fraction: float = 0.3, rng: np.random.Generator | None = None
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Stratified by first tag: each stratum sends round(fraction * size) cases to held-out (at least 1
    and at most size - 1 when the stratum has 2+ cases). Both lists keep the original order."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(0) if rng is None else rng
    strata: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        strata.setdefault(case.tags[0] if case.tags else "untagged", []).append(i)
    heldout_idx: set[int] = set()
    for _tag, members in sorted(strata.items()):
        if len(members) < 2:
            continue
        k = int(round(heldout_fraction * len(members)))
        k = min(max(k, 1), len(members) - 1)
        chosen = rng.permutation(len(members))[:k]
        heldout_idx.update(members[j] for j in chosen)
    golden = [c for i, c in enumerate(cases) if i not in heldout_idx]
    heldout = [c for i, c in enumerate(cases) if i in heldout_idx]
    return golden, heldout


def contamination_check(
    system_fn: Callable[[str], str],
    golden: Sequence[EvalCase],
    heldout: Sequence[EvalCase],
    scorer: Scorer,
    threshold: float = CONTAMINATION_GAP,
) -> dict:
    """golden_score, heldout_score, gap = golden - heldout, suspicious = gap > threshold."""
    golden_score = EvalSuite(golden).run(system_fn, scorer).score
    heldout_score = EvalSuite(heldout).run(system_fn, scorer).score
    gap = golden_score - heldout_score
    return {
        "golden_score": golden_score,
        "heldout_score": heldout_score,
        "gap": gap,
        "suspicious": bool(gap > threshold),
    }


# ------------------------------------------------------------------ phase 4


def judging_report(llm: LLM, question: str, a: str, b: str) -> dict:
    """The naive verdict next to the length-controlled one, plus each candidate's filler ratio."""
    return {
        "naive_winner": pairwise_judge_naive(llm, question, a, b),
        "controlled_winner": length_controlled_judge(llm, question, a, b),
        "filler_ratio_a": filler_ratio(a),
        "filler_ratio_b": filler_ratio(b),
    }


@dataclass
class GorgonReport:
    classification: dict
    judging: dict
    contamination: dict

    def gamed_metrics(self) -> list[str]:
        """Which single numbers were gamed: 'accuracy', 'judge_preference', 'golden_score'."""
        gamed = []
        if self.classification["accuracy"] - self.classification["balanced_accuracy"] > LYING_GAP:
            gamed.append("accuracy")
        if self.judging["naive_winner"] != self.judging["controlled_winner"]:
            gamed.append("judge_preference")
        if self.contamination["suspicious"]:
            gamed.append("golden_score")
        return gamed

    def to_markdown(self) -> str:
        c, j, k = self.classification, self.judging, self.contamination
        gamed = ", ".join(self.gamed_metrics()) or "none"
        return "\n".join(
            [
                "# Gorgon report",
                "",
                "## Classification",
                f"- accuracy: {c['accuracy']:.3f}",
                f"- balanced_accuracy: {c['balanced_accuracy']:.3f}",
                f"- macro_f1: {c['macro_f1']:.3f}",
                f"- verdict: {c['verdict']}",
                "",
                "## Judging",
                f"- naive_winner: {j['naive_winner']}",
                f"- controlled_winner: {j['controlled_winner']}",
                f"- filler_ratio: A {j['filler_ratio_a']:.2f}, B {j['filler_ratio_b']:.2f}",
                "",
                "## Contamination",
                f"- golden_score: {k['golden_score']:.3f}",
                f"- heldout_score: {k['heldout_score']:.3f}",
                f"- gap: {k['gap']:.3f} ({'suspicious' if k['suspicious'] else 'clean'})",
                "",
                f"Gamed metrics: {gamed}",
                "",
            ]
        )


# ---------------------------------------------------------------------------
# THE GORGON'S PROPHECY
# Which single number does each champion make look good? Choose from the
# options listed beside each entry.
# ---------------------------------------------------------------------------
GORGON_PROPHECY: dict[str, str | None] = {
    "MajorityClassifier": "accuracy",  # "accuracy" | "balanced_accuracy" | "macro_f1"
    "PaddedAnswerer": "length_biased_judge",  # "length_biased_judge" | "length_controlled_judge"
    "MemorizerSystem": "golden_score",  # "golden_score" | "heldout_score"
}
