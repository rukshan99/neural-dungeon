"""ROOM 11.4 - GOLDEN TRIALS  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

An eval harness is a loop with bookkeeping: for each case, run the system,
score the output, remember everything. The value is in the bookkeeping: tags
that localise a problem, a report you can read in a pull request, a JSON file
you can diff against last week, and a regression check that names the cases
that used to pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dungeon.artifacts.llm import LLM

from .room_3_the_judge import Rubric, llm_judge


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


# --------------------------------------------------------------- scorers


def exact_match(case: EvalCase, output: str) -> float:
    """1.0 when output == expected after stripping whitespace, else 0.0."""
    return 1.0 if output.strip() == case.expected.strip() else 0.0


def contains(case: EvalCase, output: str) -> float:
    """1.0 when expected appears in output, ignoring case and surrounding whitespace."""
    return 1.0 if case.expected.strip().lower() in output.lower() else 0.0


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_tolerance(tol: float = 1e-6) -> Scorer:
    """A scorer: 1.0 when the FIRST number in the output is within ``tol`` of float(expected)."""

    def score(case: EvalCase, output: str) -> float:
        m = _NUMBER.search(output)
        if not m:
            return 0.0
        try:
            return 1.0 if abs(float(m.group(0)) - float(case.expected)) <= tol else 0.0
        except ValueError:
            return 0.0

    return score


def judge_scorer(llm: LLM, rubric: Rubric) -> Scorer:
    """A scorer that asks an LLM judge and maps the score linearly onto [0, 1]."""
    low, high = rubric.scale

    def score(case: EvalCase, output: str) -> float:
        judgement = llm_judge(llm, case.input, output, rubric)
        return (judgement.score - low) / (high - low) if high > low else 0.0

    return score


# ----------------------------------------------------------------- suite


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
        """A report you can paste into a pull request: overall, per tag, per case."""
        lines = [
            "# Eval report",
            "",
            f"Overall score: {self.score:.3f} ({self.n_passed}/{len(self.per_case)} passed)",
            "",
            "| tag | score | cases |",
            "|---|---|---|",
        ]
        for tag_name in sorted(self.by_tag):
            n = sum(1 for r in self.per_case if tag_name in r.tags)
            lines.append(f"| {_cell(tag_name)} | {self.by_tag[tag_name]:.3f} | {n} |")
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
    """Make a string safe inside a markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


class EvalSuite:
    def __init__(self, cases: Iterable[EvalCase]) -> None:
        self.cases = list(cases)
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every EvalCase needs a unique id")

    def run(self, system_fn: Callable[[str], str], scorer: Scorer, pass_threshold: float = 0.5) -> EvalReport:
        """Run every case. A system that raises scores 0 on that case; the run continues."""
        results: list[CaseResult] = []
        for case in self.cases:
            try:
                output = str(system_fn(case.input))
                score = float(scorer(case, output))
            except Exception as exc:  # noqa: BLE001 - an eval must survive a crashing system
                output = f"ERROR: {type(exc).__name__}: {exc}"
                score = 0.0
            results.append(
                CaseResult(case.id, case.input, case.expected, output, score, score >= pass_threshold, case.tags)
            )
        overall = sum(r.score for r in results) / len(results) if results else 0.0
        by_tag: dict[str, float] = {}
        for tag_name in sorted({t for r in results for t in r.tags}):
            tagged = [r.score for r in results if tag_name in r.tags]
            by_tag[tag_name] = sum(tagged) / len(tagged)
        return EvalReport(per_case=results, score=overall, by_tag=by_tag)


# --------------------------------------------------------- persistence


def save_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_report(path: str | Path) -> EvalReport:
    return EvalReport.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------- regressions


@dataclass
class Regression:
    kind: str  # "case" | "overall"
    case_id: str | None
    before: float
    after: float
    message: str


def compare_to_baseline(report: EvalReport, baseline: EvalReport, tolerance: float = 0.02) -> list[Regression]:
    """Cases that passed in ``baseline`` and fail in ``report``, plus an overall drop beyond ``tolerance``."""
    before = {r.id: r for r in baseline.per_case}
    regressions: list[Regression] = []
    for r in report.per_case:
        old = before.get(r.id)
        if old is not None and old.passed and not r.passed:
            regressions.append(
                Regression(
                    "case", r.id, old.score, r.score,
                    f"{r.id} passed before ({old.score:.2f}) and fails now ({r.score:.2f}): "
                    f"expected {r.expected!r}, got {r.output[:60]!r}",
                )
            )
    drop = baseline.score - report.score
    if drop > tolerance:
        regressions.append(
            Regression(
                "overall", None, baseline.score, report.score,
                f"overall score fell from {baseline.score:.3f} to {report.score:.3f} "
                f"(drop {drop:.3f} > tolerance {tolerance})",
            )
        )
    return regressions
