"""ROOM 11.4 - GOLDEN TRIALS

    A vault of sealed scrolls, each one a question with the answer written on
    the back. The keeper unrolls them in the same order every day, reads out
    what yesterday's champion said, and keeps every ledger. "Faster than
    arguing," she says. "The scrolls do not care who is talking."

An eval harness is a loop with excellent bookkeeping:

    for case in cases:
        output = system_fn(case.input)
        score  = scorer(case, output)      # 0.0 .. 1.0
        remember everything

The bookkeeping is the product. Tags (``arithmetic``, ``lore``, ``shapes``) let
a per-tag score point at WHICH capability regressed. A markdown report goes
into the pull request. A JSON file is the baseline you compare the next run
against. A regression is a case that passed before and fails now, or an
overall drop bigger than the noise you are willing to tolerate.

Two kinds of case set: a GOLDEN set, curated and versioned, that you run on
every change; and a HELD-OUT set that nobody optimises against. The boss will
show you why you need both.

Scorers are plain functions ``(case, output) -> float``. Some are closures:
``numeric_tolerance(tol)`` and ``judge_scorer(llm, rubric)`` return scorers.
Room 3's ``llm_judge`` is available via ``from .room_3_the_judge import ...``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from dungeon.artifacts.llm import LLM

from .room_3_the_judge import Rubric, llm_judge  # noqa: F401 - for judge_scorer


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
    """1.0 when output.strip() == expected.strip(), else 0.0."""
    raise NotImplementedError("exact_match() is unwritten")


def contains(case: EvalCase, output: str) -> float:
    """1.0 when expected (stripped) appears anywhere in output, case-insensitively; else 0.0."""
    raise NotImplementedError("contains() is unwritten")


def numeric_tolerance(tol: float = 1e-6) -> Scorer:
    """Return a scorer: 1.0 when the FIRST number in the output (regex ``-?\\d+(?:\\.\\d+)?``)
    is within ``tol`` of float(case.expected); 0.0 when there is no number or it is off."""
    raise NotImplementedError("numeric_tolerance() is unwritten")


def judge_scorer(llm: LLM, rubric: Rubric) -> Scorer:
    """Return a scorer that calls llm_judge(llm, case.input, output, rubric) and maps the score
    linearly onto [0, 1]: (score - low) / (high - low)."""
    raise NotImplementedError("judge_scorer() is unwritten")


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
    score: float  # mean of the case scores
    by_tag: dict[str, float] = field(default_factory=dict)  # mean score over the cases carrying each tag

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.per_case if r.passed)

    def to_markdown(self) -> str:
        """A report for a pull request: the overall score (3 decimals) and how many passed,
        a table with one row per tag, and a table with one row per case (id, tags, expected,
        output, score, passed). Escape '|' inside cells."""
        raise NotImplementedError("EvalReport.to_markdown() is unwritten")

    def to_dict(self) -> dict:
        """JSON-ready: {"score", "by_tag", "per_case": [asdict(result), ...]}."""
        raise NotImplementedError("EvalReport.to_dict() is unwritten")

    @classmethod
    def from_dict(cls, data: dict) -> EvalReport:
        """Inverse of to_dict. Remember that JSON turned the tag tuples into lists."""
        raise NotImplementedError("EvalReport.from_dict() is unwritten")


class EvalSuite:
    def __init__(self, cases: Iterable[EvalCase]) -> None:
        self.cases = list(cases)
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every EvalCase needs a unique id")

    def run(self, system_fn: Callable[[str], str], scorer: Scorer, pass_threshold: float = 0.5) -> EvalReport:
        """Run every case in order and build the report.

        passed = score >= pass_threshold. If ``system_fn`` or the scorer raises on a case, record
        output "ERROR: <ExceptionType>: <message>" with score 0.0 and carry on: an eval that
        crashes on the tenth case tells you nothing about the other ninety.
        """
        raise NotImplementedError("EvalSuite.run() is unwritten")


# --------------------------------------------------------- persistence


def save_report(report: EvalReport, path: str | Path) -> Path:
    """Write report.to_dict() as JSON to ``path``. Return the Path."""
    raise NotImplementedError("save_report() is unwritten")


def load_report(path: str | Path) -> EvalReport:
    """Read the JSON at ``path`` back into an EvalReport equal to the one that was saved."""
    raise NotImplementedError("load_report() is unwritten")


# ---------------------------------------------------------- regressions


@dataclass
class Regression:
    kind: str  # "case" | "overall"
    case_id: str | None  # None for the overall entry
    before: float
    after: float
    message: str


def compare_to_baseline(report: EvalReport, baseline: EvalReport, tolerance: float = 0.02) -> list[Regression]:
    """One Regression(kind="case") for every case that PASSED in ``baseline`` and FAILS in ``report``
    (matched by id; cases missing from either side are ignored), in report order, plus one
    Regression(kind="overall") at the end if baseline.score - report.score > tolerance.
    Messages should quote the numbers; someone will read them at 2 a.m."""
    raise NotImplementedError("compare_to_baseline() is unwritten")
