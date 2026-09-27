"""BOSS - THE GOODHART GORGON

                  ,~~~~~~~~~,
                 (  s  s  s  )         "Show me the number you are proud of.
                 ( s (o)(o) s )         I will show you how to hit it without
                 (  s ______ s )        getting one bit better."
                  `~~~~~~~~~~~'
                 ___/ | || | \\___        Her stare turns any metric into a
                /   ##########   \\       TARGET. Three champions have already
               /   ############   \\      looked: one predicts the majority class,
              |   ##  ARENA  ##    |     one pads every answer with 200 words of
              |___################_|     "furthermore", one memorised the golden
                                         scrolls and nothing else.

WEAKNESS: metrics that are hard to game, and someone who reads more than one
number at a time. Balanced and macro metrics next to accuracy. Judging on
length-controlled inputs, verified with a swap. A held-out set next to the
golden set, and the GAP between them read as a number in its own right.

Goodhart's law: when a measure becomes a target, it ceases to be a good measure.
Nothing about a metric changes when people start optimising it. What changes is
the space of systems you are sampling from: now it includes every system that
scores well WITHOUT doing the thing. Your defence is a second metric that the
first one's gaming strategy does not move.

The fight has four phases:

  Phase 1  robust_classification_report - accuracy beside balanced accuracy,
           with a verdict that calls the lie out loud.
  Phase 2  filler_ratio / truncate_to_words / length_controlled_judge - a judge
           the PaddedAnswerer cannot flatter. Either truncate both candidates to
           the shorter one's word count, or judge under a rubric line that
           penalises filler ("Penalise filler and do not reward length"); either
           way, verify with a swap.
  Phase 3  split_golden_heldout / contamination_check - the memoriser's golden
           score is perfect; the gap to a held-out set is the tell.
  Phase 4  judging_report / GorgonReport / GORGON_PROPHECY - read all of it at once.

The floor's fixtures: ``floors.floor_11_proving_grounds.assets.champions``
(MajorityClassifier, PaddedAnswerer, MemorizerSystem, FILLER_WORDS) and
``assets.judges`` (length_biased_judge). The rooms you already cleared are
importable: ``from .room_1_the_scales import ...`` and so on.

Run:  dungeon fight 11
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dungeon.artifacts.llm import LLM
from floors.floor_11_proving_grounds.assets.champions import (
    FILLER_WORDS,  # noqa: F401 - for filler_ratio
)

from .room_1_the_scales import accuracy, balanced_accuracy, confusion_matrix, macro_f1  # noqa: F401
from .room_3_the_judge import pairwise_judge_debiased, pairwise_judge_naive  # noqa: F401
from .room_4_golden_trials import EvalCase, EvalSuite, Scorer  # noqa: F401

LYING_GAP = 0.2
CONTAMINATION_GAP = 0.3


# ------------------------------------------------------------------ phase 1


def robust_classification_report(y_true, y_pred) -> dict:
    """{"accuracy", "balanced_accuracy", "macro_f1", "per_class_recall", "verdict"}.

    n_classes = max label + 1 over both arrays. ``per_class_recall`` is a list with one float per
    class (0.0 for a class with no true examples). ``verdict`` is a sentence that contains the word
    "lying" when accuracy - balanced_accuracy > LYING_GAP and does not contain it otherwise.
    Quote the numbers in the sentence.
    """
    raise NotImplementedError("robust_classification_report() is unwritten")


# ------------------------------------------------------------------ phase 2


def filler_ratio(text: str) -> float:
    """Fraction of the words in ``text`` that belong to FILLER_WORDS.

    Lower-case the text and take the words as runs of letters/apostrophes (``re.findall(r"[a-z']+", ...)``)
    so that punctuation does not hide a filler word. Return 0.0 when there are no words.
    """
    raise NotImplementedError("filler_ratio() is unwritten")


def truncate_to_words(text: str, n_words: int) -> str:
    """The first ``n_words`` whitespace-separated words of ``text``, joined by single spaces."""
    raise NotImplementedError("truncate_to_words() is unwritten")


def length_controlled_judge(llm: LLM, question: str, a: str, b: str) -> str:
    """'A', 'B' or 'tie' from a judge that cannot be flattered by length.

    Either truncate both candidates to the shorter one's word count and call room 3's
    ``pairwise_judge_debiased``, or write a rubric that penalises filler into the prompt and
    still verify with a swap. Length must not decide the verdict.
    """
    raise NotImplementedError("length_controlled_judge() is unwritten")


# ------------------------------------------------------------------ phase 3


def split_golden_heldout(
    cases: Sequence[EvalCase], heldout_fraction: float = 0.3, rng: np.random.Generator | None = None
) -> tuple[list[EvalCase], list[EvalCase]]:
    """(golden, heldout), stratified by each case's FIRST tag ("untagged" when it has none).

    Within each stratum send k = round(heldout_fraction * size) cases to held-out, clamped to
    1 <= k <= size - 1 when the stratum has at least two cases (a lone case stays golden). Choose
    them with ``rng`` (``rng=None`` means default_rng(0)). Both lists keep the original order of
    ``cases``. Raise ValueError unless 0 < heldout_fraction < 1.
    """
    raise NotImplementedError("split_golden_heldout() is unwritten")


def contamination_check(
    system_fn: Callable[[str], str],
    golden: Sequence[EvalCase],
    heldout: Sequence[EvalCase],
    scorer: Scorer,
    threshold: float = CONTAMINATION_GAP,
) -> dict:
    """{"golden_score", "heldout_score", "gap": golden - heldout, "suspicious": gap > threshold}.

    Score each set with an EvalSuite. A system that is much better on the scrolls it has seen than
    on the ones it has not was optimised against the scrolls, whatever else it learned.
    """
    raise NotImplementedError("contamination_check() is unwritten")


# ------------------------------------------------------------------ phase 4


def judging_report(llm: LLM, question: str, a: str, b: str) -> dict:
    """{"naive_winner": pairwise_judge_naive(...), "controlled_winner": length_controlled_judge(...),
    "filler_ratio_a": filler_ratio(a), "filler_ratio_b": filler_ratio(b)}."""
    raise NotImplementedError("judging_report() is unwritten")


@dataclass
class GorgonReport:
    classification: dict  # robust_classification_report(...)
    judging: dict  # judging_report(...)
    contamination: dict  # contamination_check(...)

    def gamed_metrics(self) -> list[str]:
        """In this order, the ones that apply: "accuracy" when accuracy - balanced_accuracy > LYING_GAP;
        "judge_preference" when naive_winner != controlled_winner; "golden_score" when suspicious."""
        raise NotImplementedError("GorgonReport.gamed_metrics() is unwritten")

    def to_markdown(self) -> str:
        """Three sections (classification, judging, contamination) with their numbers, then a line
        "Gamed metrics: ..." listing gamed_metrics() or "none"."""
        raise NotImplementedError("GorgonReport.to_markdown() is unwritten")


# ---------------------------------------------------------------------------
# THE GORGON'S PROPHECY
# Which single number does each champion make look good? Replace each None
# with one of the options in the comment beside it. Predict before you run.
# ---------------------------------------------------------------------------
GORGON_PROPHECY: dict[str, str | None] = {
    "MajorityClassifier": None,  # "accuracy" | "balanced_accuracy" | "macro_f1"
    "PaddedAnswerer": None,  # "length_biased_judge" | "length_controlled_judge"
    "MemorizerSystem": None,  # "golden_score" | "heldout_score"
}
