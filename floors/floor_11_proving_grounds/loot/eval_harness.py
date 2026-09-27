"""Eval Harness - loot from Floor 11, The Proving Grounds.

A standalone evaluation toolkit: numpy and the standard library only, no
dungeon imports. Copy this file into a project and use it as is.

    from eval_harness import EvalCase, EvalSuite, contains, compare_to_baseline

    cases = [EvalCase("q1", "What is 2 + 2?", "4", tags=("arithmetic",))]
    report = EvalSuite(cases).run(my_system, contains)
    print(report.to_markdown())
    save_report(report, "baseline.json")
    ...
    for r in compare_to_baseline(new_report, load_report("baseline.json")):
        print(r.message)

Also here: percentile ``bootstrap_ci`` and ``paired_bootstrap`` for error bars,
and ``pairwise_judge_debiased`` for LLM-as-judge comparisons without position
bias. The judge functions take any object with ``complete(messages) -> text``
where messages are ``[{"role": ..., "content": ...}]`` dicts, so they work with
a thin wrapper around any chat API.

Run this file directly for a small demonstration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

# ------------------------------------------------------------------ cases


@dataclass
class EvalCase:
    id: str
    input: str
    expected: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.tags = tuple(self.tags)


Scorer = Callable[[EvalCase, str], float]
"""(case, output) -> score in [0, 1]."""


def exact_match(case: EvalCase, output: str) -> float:
    return 1.0 if output.strip() == case.expected.strip() else 0.0


def contains(case: EvalCase, output: str) -> float:
    return 1.0 if case.expected.strip().lower() in output.lower() else 0.0


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_tolerance(tol: float = 1e-6) -> Scorer:
    """Scorer: the first number in the output must be within ``tol`` of float(expected)."""

    def score(case: EvalCase, output: str) -> float:
        m = _NUMBER.search(output)
        if not m:
            return 0.0
        try:
            return 1.0 if abs(float(m.group(0)) - float(case.expected)) <= tol else 0.0
        except ValueError:
            return 0.0

    return score


# ---------------------------------------------------------------- reports


@dataclass
class CaseResult:
    id: str
    input: str
    expected: str
    output: str
    score: float
    passed: bool
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.tags = tuple(self.tags)


@dataclass
class EvalReport:
    per_case: list[CaseResult]
    score: float
    by_tag: dict[str, float] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.per_case if r.passed)

    def to_markdown(self) -> str:
        lines = [
            "# Eval report",
            "",
            f"Overall score: {self.score:.3f} ({self.n_passed}/{len(self.per_case)} passed)",
            "",
            "| tag | score | cases |",
            "|---|---|---|",
        ]
        for tag in sorted(self.by_tag):
            n = sum(1 for r in self.per_case if tag in r.tags)
            lines.append(f"| {_cell(tag)} | {self.by_tag[tag]:.3f} | {n} |")
        lines += ["", "| id | tags | expected | output | score | passed |", "|---|---|---|---|---|---|"]
        for r in self.per_case:
            lines.append(
                f"| {_cell(r.id)} | {_cell(', '.join(r.tags))} | {_cell(r.expected)} | "
                f"{_cell(r.output)} | {r.score:.2f} | {'yes' if r.passed else 'no'} |"
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {"score": self.score, "by_tag": dict(self.by_tag), "per_case": [asdict(r) for r in self.per_case]}

    @classmethod
    def from_dict(cls, data: dict) -> EvalReport:
        return cls(
            per_case=[CaseResult(**{**r, "tags": tuple(r.get("tags", ()))}) for r in data["per_case"]],
            score=float(data["score"]),
            by_tag={k: float(v) for k, v in data.get("by_tag", {}).items()},
        )


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


class EvalSuite:
    def __init__(self, cases: Iterable[EvalCase]) -> None:
        self.cases = list(cases)
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every EvalCase needs a unique id")

    def run(self, system_fn: Callable[[str], str], scorer: Scorer, pass_threshold: float = 0.5) -> EvalReport:
        """Run every case; a crashing case scores 0 and the run continues."""
        results: list[CaseResult] = []
        for case in self.cases:
            try:
                output = str(system_fn(case.input))
                score = float(scorer(case, output))
            except Exception as exc:  # noqa: BLE001
                output = f"ERROR: {type(exc).__name__}: {exc}"
                score = 0.0
            results.append(
                CaseResult(case.id, case.input, case.expected, output, score, score >= pass_threshold, case.tags)
            )
        overall = sum(r.score for r in results) / len(results) if results else 0.0
        by_tag: dict[str, float] = {}
        for tag in sorted({t for r in results for t in r.tags}):
            tagged = [r.score for r in results if tag in r.tags]
            by_tag[tag] = sum(tagged) / len(tagged)
        return EvalReport(results, overall, by_tag)


def save_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_report(path: str | Path) -> EvalReport:
    return EvalReport.from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------ regressions


@dataclass
class Regression:
    kind: str  # "case" | "overall"
    case_id: str | None
    before: float
    after: float
    message: str


def compare_to_baseline(report: EvalReport, baseline: EvalReport, tolerance: float = 0.02) -> list[Regression]:
    """Cases that passed before and fail now, plus an overall drop beyond ``tolerance``."""
    before = {r.id: r for r in baseline.per_case}
    out: list[Regression] = []
    for r in report.per_case:
        old = before.get(r.id)
        if old is not None and old.passed and not r.passed:
            out.append(
                Regression(
                    "case", r.id, old.score, r.score,
                    f"{r.id} passed before ({old.score:.2f}) and fails now ({r.score:.2f}): "
                    f"expected {r.expected!r}, got {r.output[:60]!r}",
                )
            )
    drop = baseline.score - report.score
    if drop > tolerance:
        out.append(
            Regression(
                "overall", None, baseline.score, report.score,
                f"overall score fell from {baseline.score:.3f} to {report.score:.3f} (drop {drop:.3f} > {tolerance})",
            )
        )
    return out


# ------------------------------------------------------------- error bars


def bootstrap_ci(
    values,
    statistic: Callable = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(values)``."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = np.array([statistic(x[row]) for row in idx], dtype=np.float64)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def paired_bootstrap(
    a_scores, b_scores, n_boot: int = 2000, rng: np.random.Generator | None = None, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """(delta, low, high, p_two_sided) for delta = mean(b - a), resampling paired item indices."""
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(a_scores, dtype=np.float64).ravel()
    b = np.asarray(b_scores, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired scores must line up item by item")
    d = b - a
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    deltas = d[idx].mean(axis=1)
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))))
    return float(d.mean()), float(low), float(high), p


def is_significant(ci: tuple[float, float]) -> bool:
    low, high = ci
    return bool(low > 0.0 or high < 0.0)


def required_sample_size(p: float, margin: float, z: float = 1.96) -> int:
    """Normal approximation: items needed to estimate a proportion near p to +-margin."""
    if not 0.0 < p < 1.0 or margin <= 0.0:
        raise ValueError("need 0 < p < 1 and margin > 0")
    return int(np.ceil(z * z * p * (1.0 - p) / (margin * margin)))


# ------------------------------------------------------------ judge tools


class ChatModel(Protocol):
    """Anything that turns a list of {"role", "content"} dicts into reply text."""

    def complete(self, messages: Sequence[dict]) -> str: ...


def pairwise_prompt(question: str, a: str, b: str) -> list[dict]:
    return [
        {"role": "system", "content": "Compare two candidate answers to the same question and pick the better one. Reply with a single letter: A or B."},
        {"role": "user", "content": f"Question:\n{question}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"},
    ]


def parse_choice(text: str) -> str:
    m = re.search(r"\b([AB])\b", text.strip())
    if not m:
        raise ValueError(f"no A/B verdict in {text[:80]!r}")
    return m.group(1)


def pairwise_judge_debiased(llm: ChatModel, question: str, a: str, b: str) -> str:
    """Judge (a, b) and (b, a). 'A'/'B' when both orders agree on the text, else 'tie'."""
    first = parse_choice(llm.complete(pairwise_prompt(question, a, b)))
    second = parse_choice(llm.complete(pairwise_prompt(question, b, a)))
    second_as_text = "B" if second == "A" else "A"
    return first if first == second_as_text else "tie"


def length_controlled_judge(llm: ChatModel, question: str, a: str, b: str) -> str:
    """Truncate both candidates to the shorter one's word count, then judge with a swap."""
    n = min(len(a.split()), len(b.split()))
    cut = lambda t: " ".join(t.split()[:n])  # noqa: E731
    return pairwise_judge_debiased(llm, question, cut(a), cut(b))


def cohens_kappa(labels_a, labels_b) -> float:
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    p_o = float(np.mean(a == b))
    p_e = float(sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b)))
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ------------------------------------------------------------------- demo

if __name__ == "__main__":
    cases = [
        EvalCase("add", "What is 2 + 2?", "4", ("arithmetic",)),
        EvalCase("mul", "What is 3 * 7?", "21", ("arithmetic",)),
        EvalCase("cap", "What is the capital letter after B?", "C", ("letters",)),
    ]

    def yesterday(prompt: str) -> str:
        return {"What is 2 + 2?": "4", "What is 3 * 7?": "21", "What is the capital letter after B?": "C"}[prompt]

    def today(prompt: str) -> str:
        return "I would rather not say." if "*" in prompt else yesterday(prompt)

    baseline = EvalSuite(cases).run(yesterday, contains)
    current = EvalSuite(cases).run(today, contains)
    print(current.to_markdown())
    for reg in compare_to_baseline(current, baseline):
        print("REGRESSION:", reg.message)
    rng = np.random.default_rng(0)
    a = (rng.random(200) < 0.80).astype(float)
    b = np.clip(a + (rng.random(200) < 0.05), 0, 1)
    print("paired bootstrap (delta, low, high, p):", paired_bootstrap(a, b, rng=rng))
