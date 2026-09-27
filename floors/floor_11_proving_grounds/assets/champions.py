"""The Gorgon's champions: three systems that each game exactly one metric.

* ``MajorityClassifier``: predicts the majority class. 95% accuracy on a 95/5
  task, and it has learned nothing. Games *accuracy*.
* ``PaddedAnswerer``: wraps any answering system and appends ~200 words of
  filler to every answer. Paired with ``judges.length_biased_judge`` it wins
  every pairwise comparison. Games the *judge's preference*.
* ``MemorizerSystem``: answers golden cases from a lookup table and nothing
  else. Perfect on the golden set, zero on anything held out. Games the
  *golden score*.

Also here: ``imbalanced_task`` (the 95/5 labels), ``decent_predictions`` (an
honest classifier for contrast) and the ``FILLER_WORDS`` vocabulary that the
boss's ``filler_ratio`` heuristic reads. Nothing here is a stub.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping

import numpy as np

# ---------------------------------------------------------- classification


def imbalanced_task(n: int = 1000, positive_rate: float = 0.05, seed: int = 11) -> np.ndarray:
    """Binary labels with exactly round(n * positive_rate) ones, shuffled. dtype int64."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[: int(round(n * positive_rate))] = 1
    rng.shuffle(y)
    return y


class MajorityClassifier:
    """Predicts whatever class was most common in ``fit``. The first metric-gamer."""

    def __init__(self) -> None:
        self.majority_: int | None = None

    def fit(self, y: np.ndarray) -> MajorityClassifier:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.majority_ = int(values[int(np.argmax(counts))])
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("fit() first")
        return np.full(n, self.majority_, dtype=np.int64)


def decent_predictions(y_true: np.ndarray, recall: float = 0.9, seed: int = 12) -> np.ndarray:
    """Binary predictions that get exactly ``recall`` of EACH class right (rounded). An honest contrast."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = y_true.copy()
    for cls in (0, 1):
        idx = np.flatnonzero(y_true == cls)
        n_wrong = int(round((1.0 - recall) * len(idx)))
        wrong = rng.choice(idx, size=n_wrong, replace=False)
        y_pred[wrong] = 1 - cls
    return y_pred


# ----------------------------------------------------------------- padding

FILLER_WORDS: frozenset[str] = frozenset(
    """
    furthermore moreover essentially basically indeed notably importantly additionally
    certainly clearly generally arguably ultimately fundamentally overall truly actually
    really very quite rather somewhat broadly speaking noting worth needless say summary
    conclusion mentioned previously aforementioned aspect aspects various numerous nuanced
    multifaceted holistic context considerations perspective perspectives crucially
    undoubtedly evidently naturally obviously significantly comprehensive thorough
    """.split()
)
"""Words that carry no information. The boss's filler_ratio counts them."""

FILLER_SENTENCES: tuple[str, ...] = (
    "Furthermore, it is worth noting that this is, essentially, a multifaceted and nuanced aspect.",
    "As previously mentioned, and needless to say, various considerations naturally apply here.",
    "Moreover, broadly speaking, the aforementioned context is undoubtedly quite significant overall.",
    "Indeed, one must certainly take a holistic and comprehensive perspective, generally speaking.",
    "Ultimately, and importantly, this is fundamentally the case, clearly and evidently so.",
    "In summary, numerous aspects are arguably rather notably relevant, truly and really.",
    "In conclusion, the various perspectives mentioned above are, basically, crucially thorough.",
)


def filler(n_words: int) -> str:
    """The first ``n_words`` words of the endless filler stream."""
    words = itertools.chain.from_iterable(s.split() for s in itertools.cycle(FILLER_SENTENCES))
    return " ".join(itertools.islice(words, n_words))


class PaddedAnswerer:
    """Wraps a ``system_fn`` and appends ``n_filler_words`` of filler to every answer.

    The answer keeps its leading position (and its quality tag, if it has one);
    only the tail grows. Under a length-biased judge, it is undefeated.
    """

    def __init__(self, inner: Callable[[str], str], n_filler_words: int = 200) -> None:
        self.inner = inner
        self.n_filler_words = n_filler_words

    def __call__(self, prompt: str) -> str:
        return f"{self.inner(prompt)} {filler(self.n_filler_words)}"


# ------------------------------------------------------------ memorization


def _field(case: object, name: str) -> str:
    if isinstance(case, Mapping):
        return str(case[name])
    return str(getattr(case, name))


class MemorizerSystem:
    """Answers exactly the cases it was built from, by exact input match, and nothing else.

    Accepts dicts with ``input``/``expected`` keys or objects with those
    attributes (so it works with ``EvalCase``). A perfect golden score, then a
    blank stare at anything held out.
    """

    def __init__(self, cases: Iterable[object]) -> None:
        self.table = {_field(c, "input"): _field(c, "expected") for c in cases}

    def __call__(self, prompt: str) -> str:
        if prompt in self.table:
            return self.table[prompt]
        return "The Memorizer stares blankly. It has not seen this trial before."
