"""The deployed dungeon, simulated: a stream of days for the Watchtower to watch.

Nothing here is a stub, and nothing here imports from ``rooms/`` or
``solutions/``. Trials build the learner's ``Watchtower`` around these fixtures.

THE WORLD. Every request carries four numeric features ``x0..x3``. The true
label is a noisy rule on the features with a hinge on ``x2``:

    margin = x0 + 0.5*x1 - 3*max(0, x2 - 1) + 0.3*noise
    label  = 1 if margin > 0 else 0

Before the drift ``x2 ~ N(0, 0.5^2)``, so it exceeds 1 on about 2% of requests
and a rule that ignores the hinge loses about a point of accuracy. From
``drift_day`` on, the mean of ``x2`` climbs by ``drift_rate`` per day and
nothing else changes: the *inputs* move, the rule that makes the labels does
not. That is covariate shift, and it is the kind a drift glass can see.

THE MODELS. ``DEPLOYED`` ignores ``x2`` (it never mattered when the model was
built). ``RETRAINED`` knows the hinge and keeps its accuracy on drifted data.
``WORSE`` is a hasty retrain that is worse than what is deployed. ``SLOW`` is as
accurate as ``RETRAINED`` and three times slower; ``FLAKY`` is as accurate and
fails 8% of its requests. Each exposes
``serve(features, rng) -> (predictions, latency_ms)``; a failed request carries
the prediction ``-1``, and latencies are lognormal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

N_FEATURES = 4
FEATURE_SCALES = np.array([1.0, 1.0, 0.5, 1.0])
LABEL_NOISE = 0.3


# ------------------------------------------------------------------ the truth


def _margin(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return x[:, 0] + 0.5 * x[:, 1] - 3.0 * np.maximum(0.0, x[:, 2] - 1.0)


def true_labels(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The world's verdict on each request: int64 0/1, noisy so no model is perfect."""
    noise = LABEL_NOISE * rng.standard_normal(len(features))
    return (_margin(features) + noise > 0).astype(np.int64)


# ----------------------------------------------------------------- the models


def deployed_rule(features: np.ndarray) -> np.ndarray:
    """Ignores x2. Fine while x2 stays small; wrong once the world moves."""
    x = np.asarray(features, dtype=np.float64)
    return (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)


def retrained_rule(features: np.ndarray) -> np.ndarray:
    """Knows the hinge on x2. Accurate before and after the drift."""
    return (_margin(features) > 0).astype(np.int64)


def worse_rule(features: np.ndarray) -> np.ndarray:
    """A hasty retrain that only looks at x1. About 60% accurate."""
    x = np.asarray(features, dtype=np.float64)
    return (x[:, 1] > 0).astype(np.int64)


@dataclass(frozen=True)
class Model:
    """A served model: a prediction rule plus its latency and failure behaviour."""

    name: str
    rule: Callable[[np.ndarray], np.ndarray]
    latency_median_ms: float
    latency_sigma: float = 0.4  # lognormal spread; p95 = median * exp(1.645 * sigma)
    failure_rate: float = 0.001  # fraction of requests that fail outright (prediction -1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.rule(np.asarray(features, dtype=np.float64))

    def serve(self, features: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(predictions, latency_ms) for a batch. Failed requests predict -1."""
        n = len(features)
        preds = self.predict(features)
        failed = rng.random(n) < self.failure_rate
        preds = np.where(failed, -1, preds).astype(np.int64)
        latency = self.latency_median_ms * np.exp(self.latency_sigma * rng.standard_normal(n))
        return preds, latency


DEPLOYED = Model("scribe-v1", deployed_rule, latency_median_ms=300.0)
RETRAINED = Model("scribe-v2", retrained_rule, latency_median_ms=340.0)
WORSE = Model("scribe-v2-hasty", worse_rule, latency_median_ms=340.0)
SLOW = Model("scribe-v2-ponderous", retrained_rule, latency_median_ms=900.0)  # p95 ~ 1740 ms
FLAKY = Model("scribe-v2-brittle", retrained_rule, latency_median_ms=340.0, failure_rate=0.08)


# ------------------------------------------------------------------- the days


@dataclass
class Day:
    """One day of traffic: features (N, 4) and the labels the world will eventually reveal."""

    day: int
    features: np.ndarray
    labels: np.ndarray

    @property
    def n_requests(self) -> int:
        return int(self.features.shape[0])


def feature_shift(day: int, drift_day: int, drift_rate: float) -> float:
    """How far the mean of x2 has moved on ``day``: 0 before the drift, then drift_rate per day."""
    if day < drift_day:
        return 0.0
    return drift_rate * (day - drift_day + 1)


def sample_features(n: int, rng: np.random.Generator, shift: float = 0.0) -> np.ndarray:
    """(n, 4) features from the world with x2's mean moved by ``shift``."""
    x = rng.standard_normal((n, N_FEATURES)) * FEATURE_SCALES
    x[:, 2] += shift
    return x


def reference_batch(seed: int = 13, n: int = 5000) -> np.ndarray:
    """Pre-drift features: what the deployed model was built on. The glass compares against this."""
    return sample_features(n, np.random.default_rng(seed))


def simulate_days(
    n_days: int,
    drift_day: int,
    seed: int,
    n_per_day: int = 500,
    drift_rate: float = 0.06,
) -> Iterator[Day]:
    """Yield ``n_days`` days of traffic; the mean of x2 starts climbing on ``drift_day``.

    Deterministic for a given seed. ``drift_day >= n_days`` means no drift at all.
    """
    rng = np.random.default_rng(seed)
    for day in range(n_days):
        features = sample_features(n_per_day, rng, feature_shift(day, drift_day, drift_rate))
        yield Day(day=day, features=features, labels=true_labels(features, rng))


def with_label_delay(days: Sequence[Day], delay: int) -> Iterator[tuple[Day, np.ndarray | None]]:
    """Pair each day with the labels that ARRIVE that day: those of ``delay`` days earlier, or None.

    Labels for day t are only known on day t + delay. This is the label-delay
    problem: quality is always measured in the past.
    """
    if delay < 0:
        raise ValueError("delay must be non-negative")
    for index, day in enumerate(days):
        late = days[index - delay].labels if index >= delay else None
        yield day, late
