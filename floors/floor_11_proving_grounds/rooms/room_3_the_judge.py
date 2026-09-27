"""ROOM 11.3 - THE JUDGE

    A raised bench, a gavel, and a judge who has never once been wrong about a
    contest with an obvious winner. Watch her when the contest is close, though:
    she always points to the champion standing on her left.

When there is no exact answer to compare against (summaries, explanations,
code review comments), you ask a model to grade. That is "LLM-as-judge", and
it is three engineering problems wearing one robe:

1. THE PROMPT. A rubric (criteria and a scale), the question, the answer, and a
   machine-readable output format: JSON with "score" and "rationale". Ask for
   the rationale; it makes disagreements debuggable.
2. THE PARSE. Models decorate JSON with fences and prose. Find the first
   balanced {...}, fall back to a regex, raise ValueError only when there is
   truly no score. Never let a formatting quirk become a silent zero.
3. THE BIAS. Judges prefer the first candidate (position bias), the longer one
   (length bias), and their own writing style. Position bias has a clean fix:
   judge (a, b) AND (b, a). If the verdicts agree, trust them; if the judge
   contradicts itself, the honest answer is "tie".

Then measure the judge like any other system: against humans. Cohen's kappa
corrects raw agreement for the agreement chance alone would produce,

    kappa = (p_o - p_e) / (1 - p_e),   p_e = sum_k p_a(k) p_b(k)

and Spearman's rank correlation is Pearson's correlation computed on ranks
(ties get the average of the ranks they span).

PROMPT CONTRACT. This floor's mock judges (and the trial) read your prompts, so
the format is fixed:
  * ``judge_prompt``: a system message with the criteria, the scale and the JSON
    instruction; a user message containing the question and the answer.
  * ``pairwise_prompt``: a system message with the instructions ("reply with a
    single letter, A or B"); a user message holding the question, then the line
    ``Answer A:`` followed by a, then the line ``Answer B:`` followed by b, and
    NOTHING after b.
The mocks stand in for a real model by reading a hidden ``[quality=N]`` tag in
each answer. Your code must not look at it; the trial would not be fooled anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np  # noqa: F401 - you will want it for the ranks

from dungeon.artifacts.llm import LLM, Message  # noqa: F401 - Message builds prompts


@dataclass
class Rubric:
    criteria: list[str]
    scale: tuple[int, int] = (1, 5)


@dataclass
class Judgement:
    score: int
    rationale: str
    raw: str = field(default="", repr=False)


# --------------------------------------------------------------- scoring


def judge_prompt(question: str, answer: str, rubric: Rubric) -> list[Message]:
    """[system, user]. System: judge role, every criterion, the scale's two ends, and an instruction
    to reply with JSON of the form {"score": <int>, "rationale": "<text>"}. User: the question and the answer."""
    raise NotImplementedError("judge_prompt() is unwritten")


def parse_judgement(text: str) -> tuple[int, str]:
    """(score, rationale) from a judge reply.

    Must survive: bare JSON; JSON inside ``` fences (with or without a language tag);
    prose before and after the JSON; a numeric score given as a string or a float.
    Raise ValueError when no score can be found. Return the score as a Python int.
    """
    raise NotImplementedError("parse_judgement() is unwritten")


def llm_judge(llm: LLM, question: str, answer: str, rubric: Rubric) -> Judgement:
    """Build the prompt, call ``llm.complete(messages)``, parse ``completion.text``.

    Raise ValueError if the score is outside ``rubric.scale``. Keep the raw reply in ``Judgement.raw``.
    """
    raise NotImplementedError("llm_judge() is unwritten")


# -------------------------------------------------------------- pairwise


def pairwise_prompt(question: str, a: str, b: str) -> list[Message]:
    """[system, user] following the PROMPT CONTRACT in the module docstring."""
    raise NotImplementedError("pairwise_prompt() is unwritten")


def parse_choice(text: str) -> str:
    """'A' or 'B' from a judge reply (a standalone letter; ``\\b([AB])\\b``). ValueError if neither is present."""
    raise NotImplementedError("parse_choice() is unwritten")


def pairwise_judge_naive(llm: LLM, question: str, a: str, b: str) -> str:
    """One call with pairwise_prompt(question, a, b); return 'A' or 'B'."""
    raise NotImplementedError("pairwise_judge_naive() is unwritten")


def pairwise_judge_debiased(llm: LLM, question: str, a: str, b: str) -> str:
    """Judge (a, b) and then (b, a). Return 'A' if a wins both, 'B' if b wins both, else 'tie'.

    Careful with the translation: in the second call, a verdict of "A" means b won.
    Exactly two calls to the judge.
    """
    raise NotImplementedError("pairwise_judge_debiased() is unwritten")


# ------------------------------------------------------------- agreement


def cohens_kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """(p_o - p_e) / (1 - p_e). p_o = fraction of positions where the labels agree;
    p_e = sum over categories k of (fraction of a equal to k) * (fraction of b equal to k).

    Perfect agreement is 1.0; agreement at chance level is 0.0; systematic disagreement is negative.
    If p_e == 1 (both raters always give the same single label) return 1.0.
    """
    raise NotImplementedError("cohens_kappa() is unwritten")


def average_ranks(values) -> np.ndarray:
    """1-based ranks of ``values``; tied values all receive the MEAN of the ranks they would occupy.

    average_ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    """
    raise NotImplementedError("average_ranks() is unwritten")


def spearman(x, y) -> float:
    """Spearman's rank correlation: Pearson's r computed on average_ranks(x) and average_ranks(y).

    Return 0.0 if either sequence is constant (its ranks have zero variance).
    """
    raise NotImplementedError("spearman() is unwritten")


def agreement_report(judge_scores: Sequence, human_scores: Sequence) -> dict:
    """{"n": int, "kappa": float, "spearman": float, "exact_agreement": float, "mean_abs_diff": float}.

    exact_agreement is the fraction of items where the scores are equal; mean_abs_diff is
    mean(|judge - human|). Raise ValueError if the sequences do not pair up.
    """
    raise NotImplementedError("agreement_report() is unwritten")
