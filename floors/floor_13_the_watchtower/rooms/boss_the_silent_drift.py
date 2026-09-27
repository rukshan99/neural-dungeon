"""BOSS - THE SILENT DRIFT

                                                      "No day is the day it changed.
      .       .        .       .        .       .      Ask your accuracy. It will tell
         .        .        .       .        .          you nothing happened, and it will
      .     .   .    .    .   .    .    .    .  .      go on saying so for weeks, and it
     . . . . . . . . . . . . . . . . . . . . . . .     will be telling the truth about
    ...............................................    the only days it can see."
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   The Silent Drift moves the world
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  under a model a little each day.

WEAKNESS: watching the INPUTS, not just the accuracy, and deciding with
statistics. The inputs are visible today. Accuracy needs labels, labels arrive
``label_delay`` days late, and the drift has to grow before accuracy suffers at
all. A tower that only watches accuracy is always weeks behind.

The fight is one class, ``Watchtower``, driven a day at a time by the trial:

    tower.observe_day(day, features, labels_delayed)

``features`` is today's (N, 4) traffic; ``labels_delayed`` is the labels of day
``day - label_delay`` (or None while none have arrived). Inside a day, in order:

  1. THE GLASS   (room 2)  ``self.monitor.observe(features)``. While ``state ==
     "watching"`` and the report says drifted: set ``drift_day``, ``state =
     "canary"``, ``canary_start_day``, ``canary_fraction = self.ramp[0]``, and
     log signal "input_drift", decision "start_canary".
  2. SERVE       (room 3)  request ids are ``f"{day}-{i}"``. In "canary" state
     ``assign_arm(id, canary_fraction, self.salt)`` picks the arm; "promoted"
     sends everything to the candidate; otherwise everything to the deployed
     model. Call ``model.serve(features[mask], self.rng)`` per arm and store a
     ``DayRecord`` in ``self.records[day]``: you will need it when labels come.
  3. GUARDRAILS  (rooms 3, 4)  latency and failures of TODAY's canary requests
     are known today. Push them into ``self.canary_latency`` / ``self.canary_failures``
     (1.0 per failed request, 0.0 otherwise), read ``canary_metrics(...)``, run
     ``guardrail_check(metrics, self.slo)``. A failure rolls back immediately:
     ``state = "rolled_back"``, fraction 0, log signal "guardrail", decision
     "rollback", with the violations named. Do not wait for labels you do not need.
  4. LATE LABELS (room 4)  if ``labels_delayed`` is not None and the record for
     ``day - label_delay`` exists: ``score_record`` it. Feed its ``error_rate`` to
     ``self.alerts.observe("error_rate", ...)`` and log each event (signal
     "error_rate", decision "alert" or "resolved"). If that day was a canary
     day, append its per-arm scores to ``self.baseline_scores`` / ``self.canary_scores``.
  5. DECIDE      (room 3)  in "canary" state, once the baseline list is non-empty,
     at least ``MIN_CANARY_SCORES`` canary scores have arrived (fifty labelled
     requests have a standard error of four points; do not decide on them) and
     ``hold_until`` has passed: ``compare_arms`` on everything scored so far,
     ``release_decision(comparison, guardrails, self.min_effect)``, and log it
     under signal "canary". "promote" moves the fraction up the ramp
     (``advance_ramp``), sets ``hold_until = day + 1 + label_delay`` (no second
     decision on the same evidence: tomorrow is the first day at the new
     fraction and its labels take label_delay days) and, at 1.0, ``state =
     "promoted"``. "rollback" sets ``state = "rolled_back"`` and fraction 0.

THE LEDGER. Every entry has ``day, signal, value, decision, reason``; ``self.log``
builds it and accepts extra keyword ``details``. The trial reads these details:
canary decisions carry the comparison (``delta, ci_low, ci_high, n_baseline,
n_canary``), promotes also ``guardrails_passed=True, fraction_before,
fraction_after``; guardrail rollbacks carry ``violations``. Reasons quote numbers.

Fixtures: ``floors.floor_13_the_watchtower.assets.deployment`` (``Model``,
``DEPLOYED``, ``RETRAINED``, ``WORSE``, ``SLOW``, ``FLAKY``, ``simulate_days``,
``reference_batch``, ``with_label_delay``). The rooms you cleared are imported
below. Then fill in ``DRIFT_PROPHECY`` before you run.

Run:  dungeon fight 13
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floors.floor_13_the_watchtower.assets.deployment import Model

from .room_2_the_drift_glass import DriftMonitor
from .room_3_the_canary_cage import (  # noqa: F401 - the tools of the cage
    advance_ramp,
    assign_arm,
    compare_arms,
    guardrail_check,
    ramp_schedule,
    release_decision,
)
from .room_4_the_alarm_bell import AlertManager, RollingWindow, Threshold

DRIFT_THRESHOLDS = {"psi": 0.1, "ks": 0.15}  # PSI "moderate" is enough to start a canary; KS at 0.1 false-alarms at n=500
DEFAULT_SLO = {"p95_latency_ms": 1000.0, "error_rate": 0.02}
ERROR_RATE_FIRE = 0.15
ERROR_RATE_CLEAR = 0.12
DEFAULT_RAMP = (0.1, 0.5, 1.0)
GUARDRAIL_WINDOW = 2000  # canary requests pooled for p95 / error rate; one day's 50 requests is too few to judge
MIN_CANARY_SCORES = 200  # no quality decision on fewer labelled canary requests (SE ~0.04 at 50, ~0.02 at 200)
LOG_FIELDS = ("day", "signal", "value", "decision", "reason")


@dataclass
class DayRecord:
    """What was served on one day: who served each request, what they said, how long it took."""

    day: int
    canary: np.ndarray  # bool, True where the candidate served the request
    predictions: np.ndarray  # int, -1 for a failed request
    latency_ms: np.ndarray


def score_record(record: DayRecord, labels) -> dict:
    """Join late labels onto a day's record.

    {"correct": (N,) float 0/1 (a failed request, prediction -1, is wrong),
     "baseline_scores": correct where not record.canary, "canary_scores": correct where record.canary,
     "error_rate": 1 - mean(correct)}. Raise ValueError if the labels do not line up with the predictions.
    """
    raise NotImplementedError("score_record() is unwritten")


def canary_metrics(latency_window: RollingWindow, failure_window: RollingWindow) -> dict:
    """{"p95_latency_ms": latency_window.percentile(95), "error_rate": fraction of failure_window values >= 1.0}.

    Raise ValueError if either window is empty.
    """
    raise NotImplementedError("canary_metrics() is unwritten")


class Watchtower:
    """Watches a deployed model, runs a canary of a candidate when the inputs drift, decides with statistics."""

    def __init__(
        self,
        reference_features,
        deployed: Model,
        candidate: Model,
        label_delay: int,
        *,
        ramp=DEFAULT_RAMP,
        slo: dict | None = None,
        drift_thresholds: dict | None = None,
        min_effect: float = 0.0,
        n_boot: int = 300,
        alpha: float = 0.05,
        alert_cooldown: int = 3,
        salt: str = "watchtower",
        seed: int = 0,
    ) -> None:
        self.deployed = deployed
        self.candidate = candidate
        self.label_delay = int(label_delay)
        self.ramp = ramp_schedule(ramp)
        self.slo = dict(DEFAULT_SLO if slo is None else slo)
        self.min_effect = float(min_effect)
        self.n_boot = int(n_boot)
        self.alpha = float(alpha)
        self.salt = salt
        self.rng = np.random.default_rng(seed)
        self.monitor = DriftMonitor(reference_features, window=1, thresholds=drift_thresholds or DRIFT_THRESHOLDS)
        self.day = -1
        self.alerts = AlertManager(clock=lambda: self.day, cooldown=alert_cooldown)
        self.alerts.add_rule("error_rate", Threshold(ERROR_RATE_FIRE, ERROR_RATE_CLEAR), severity="page")
        self.state = "watching"  # watching | canary | promoted | rolled_back
        self.canary_fraction = 0.0
        self.drift_day: int | None = None
        self.canary_start_day: int | None = None
        self.hold_until: int | None = None  # after a ramp step, no decision until labels from the new fraction arrive
        self.canary_latency = RollingWindow(GUARDRAIL_WINDOW)
        self.canary_failures = RollingWindow(GUARDRAIL_WINDOW)
        self.records: dict[int, DayRecord] = {}
        self.baseline_scores: list[np.ndarray] = []
        self.canary_scores: list[np.ndarray] = []
        self.decision_log: list[dict] = []

    # ------------------------------------------------------------ the ledger

    def log(self, day: int, signal: str, value: float, decision: str, reason: str, **details) -> dict:
        """Append {"day", "signal", "value", "decision", "reason"} (+ "details" if any) and return it."""
        entry = {"day": int(day), "signal": signal, "value": float(value), "decision": decision, "reason": reason}
        if details:
            entry["details"] = details
        self.decision_log.append(entry)
        return entry

    def first_day(self, signal: str, decision: str | None = None) -> int | None:
        """The day of the first log entry with this signal (and decision, if given), or None."""
        for entry in self.decision_log:
            if entry["signal"] == signal and (decision is None or entry["decision"] == decision):
                return entry["day"]
        return None

    # --------------------------------------------------------------- one day

    def observe_day(self, day: int, features, labels_delayed) -> dict:
        """One day in the tower: glass, serve, guardrails, late labels, decide (see the module docstring).

        Set ``self.day = day`` first (the alert manager's clock reads it). Return a small report dict,
        e.g. {"day", "state", "canary_fraction", "drift": <monitor report>, "decision"}; the trial reads
        ``decision_log``, ``state``, ``drift_day``, ``canary_start_day`` and ``canary_fraction``.
        Split the work into helper methods if you like.
        """
        raise NotImplementedError("Watchtower.observe_day() is unwritten")


def render_decision_log(log) -> str:
    """The audit trail as a markdown table: a header ``| day | signal | value | decision | reason |``,
    a separator, one row per entry (value to 4 decimals), '|' inside cells escaped as '\\|'."""
    raise NotImplementedError("render_decision_log() is unwritten")


# ---------------------------------------------------------------------------
# THE DRIFT PROPHECY
# Replace each None with one of the two options named in the question.
# Reason first (label delay; what a median can see; Floor 11's sample-size
# arithmetic), then let the trial judge you.
# ---------------------------------------------------------------------------
DRIFT_PROPHECY: dict[str, str | None] = {
    "under slow covariate drift, which signal moves first: input_drift or error_rate": None,
    "a slow tail grows on 3% of requests: which percentile moves first, p50 or p99": None,
    "a 0.5% canary at 1000 requests/day sees a 2-point accuracy change within a week: yes or no": None,
}
