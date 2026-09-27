"""ROOM 11.3 - THE JUDGE  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

A judge is a model call with three engineering problems around it: the prompt
(rubric, scale, output format), the parse (models decorate JSON), and the bias
(position, length, self-preference). Swapping the candidates and demanding a
consistent verdict removes position bias at the cost of a second call. Kappa
and Spearman tell you whether the judge agrees with humans more than chance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from dungeon.artifacts.llm import LLM, Message


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
    """System message: role, criteria, scale, JSON format. User message: question + answer."""
    low, high = rubric.scale
    criteria = "\n".join(f"- {c}" for c in rubric.criteria)
    system = (
        "You are a strict, fair judge of answers. Score the answer against every criterion below "
        f"on an integer scale from {low} (worst) to {high} (best).\n"
        f"Criteria:\n{criteria}\n"
        'Reply with JSON only, exactly of the form {"score": <integer>, "rationale": "<one sentence>"}.'
    )
    user = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return [Message.system(system), Message.user(user)]


_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


def _first_json_object(text: str) -> dict | None:
    """The first balanced {...} in ``text`` that parses as JSON, or None."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


def parse_judgement(text: str) -> tuple[int, str]:
    """(score, rationale) from a judge reply that may be fenced, prosed or bare. ValueError if no score."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        obj = _first_json_object(chunk)
        if obj is not None and "score" in obj:
            return int(round(float(obj["score"]))), str(obj.get("rationale", "")).strip()
    # Last resort: a bare "score: 4" somewhere in prose.
    m = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        r = re.search(r'"?rationale"?\s*[:=]\s*"?([^"\n}]*)', text, re.IGNORECASE)
        return int(round(float(m.group(1)))), (r.group(1).strip() if r else "")
    raise ValueError(f"No score found in judge reply: {text[:80]!r}")


def llm_judge(llm: LLM, question: str, answer: str, rubric: Rubric) -> Judgement:
    """One call, parsed and range-checked against the rubric's scale."""
    completion = llm.complete(judge_prompt(question, answer, rubric))
    score, rationale = parse_judgement(completion.text)
    low, high = rubric.scale
    if not low <= score <= high:
        raise ValueError(f"judge scored {score}, outside the rubric scale {rubric.scale}")
    return Judgement(score=score, rationale=rationale, raw=completion.text)


# -------------------------------------------------------------- pairwise


def pairwise_prompt(question: str, a: str, b: str) -> list[Message]:
    """Instructions in the system message; question, 'Answer A:' a, 'Answer B:' b in the user message."""
    system = (
        "You compare two candidate answers to the same question and pick the better one. "
        "Reply with a single letter: A or B."
    )
    user = f"Question:\n{question}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"
    return [Message.system(system), Message.user(user)]


def parse_choice(text: str) -> str:
    """'A' or 'B' from a judge reply. ValueError when neither letter stands alone."""
    m = re.search(r"\b([AB])\b", text.strip())
    if not m:
        raise ValueError(f"No A/B verdict in judge reply: {text[:80]!r}")
    return m.group(1)


def pairwise_judge_naive(llm: LLM, question: str, a: str, b: str) -> str:
    """One call, one letter. Position bias included at no extra charge."""
    return parse_choice(llm.complete(pairwise_prompt(question, a, b)).text)


def pairwise_judge_debiased(llm: LLM, question: str, a: str, b: str) -> str:
    """Judge (a, b) and (b, a); return the consistent winner as 'A'/'B' (meaning a/b), else 'tie'."""
    first = pairwise_judge_naive(llm, question, a, b)  # "A" here means a
    second = pairwise_judge_naive(llm, question, b, a)  # "A" here means b
    winner_first = first
    winner_second = "B" if second == "A" else "A"  # translate back to a/b
    return winner_first if winner_first == winner_second else "tie"


# ------------------------------------------------------------- agreement


def cohens_kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """(p_o - p_e) / (1 - p_e): observed agreement corrected for the agreement chance would give."""
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("kappa needs two equally long, non-empty label sequences")
    p_o = float(np.mean(a == b))
    categories = np.union1d(a, b)
    p_e = float(sum(np.mean(a == c) * np.mean(b == c) for c in categories))
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def average_ranks(values) -> np.ndarray:
    """1-based ranks; tied values share the mean of the ranks they would have taken."""
    x = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    while i < x.shape[0]:
        j = i
        while j + 1 < x.shape[0] and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # mean of positions i..j, 1-based
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Pearson correlation of the average ranks of x and y."""
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.shape != ry.shape or rx.size < 2:
        raise ValueError("spearman needs two equally long sequences of at least two values")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return 0.0  # a constant sequence has no rank order to correlate
    return float(np.sum(rx * ry) / denom)


def agreement_report(judge_scores: Sequence, human_scores: Sequence) -> dict:
    """kappa, spearman, exact agreement rate and mean absolute difference between judge and humans."""
    j = np.asarray(judge_scores, dtype=np.float64).ravel()
    h = np.asarray(human_scores, dtype=np.float64).ravel()
    if j.shape != h.shape or j.size == 0:
        raise ValueError("judge and human scores must pair up")
    return {
        "n": int(j.shape[0]),
        "kappa": cohens_kappa(j, h),
        "spearman": spearman(j, h),
        "exact_agreement": float(np.mean(j == h)),
        "mean_abs_diff": float(np.mean(np.abs(j - h))),
    }
