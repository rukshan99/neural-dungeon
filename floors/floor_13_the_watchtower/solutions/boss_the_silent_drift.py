"""BOSS - THE SILENT DRIFT  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

The order of operations inside a day is the whole fight:

  1. the glass:      compare today's inputs with the reference (no labels needed)
  2. serve:          assign each request to an arm, remember who served what
  3. guardrails:     latency and errors of the canary arm are known TODAY
  4. late labels:    score the day whose labels just arrived; ring the bell
  5. decide:         compare arms on everything scored so far; promote/extend/rollback

Inputs move first; accuracy follows, ``label_delay`` days late. Everything
that changes the state goes into ``decision_log`` with its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floors.floor_13_the_watchtower.assets.deployment import Model

from .room_2_the_drift_glass import DriftMonitor
from .room_3_the_canary_cage import (
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
    """Join late labels onto a day's record: per-arm 0/1 scores and the day's error rate."""
    labels = np.asarray(labels).ravel()
    if labels.shape[0] != record.predictions.shape[0]:
        raise ValueError("labels do not line up with the day's requests")
    correct = (record.predictions == labels).astype(np.float64)  # a failed request (-1) is wrong
    return {
        "correct": correct,
        "baseline_scores": correct[~record.canary],
        "canary_scores": correct[record.canary],
        "error_rate": float(1.0 - correct.mean()),
    }


def canary_metrics(latency_window: RollingWindow, failure_window: RollingWindow) -> dict:
    """{"p95_latency_ms": nearest-rank p95 of the pooled latencies, "error_rate": fraction of pooled failures}.

    ``failure_window`` holds one 1.0/0.0 per canary request (1.0 = the request failed).
    """
    if len(latency_window) == 0 or len(failure_window) == 0:
        raise ValueError("no canary traffic to measure")
    return {
        "p95_latency_ms": latency_window.percentile(95),
        "error_rate": failure_window.rate(lambda failed: failed >= 1.0),
    }


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
        entry = {"day": int(day), "signal": signal, "value": float(value), "decision": decision, "reason": reason}
        if details:
            entry["details"] = details
        self.decision_log.append(entry)
        return entry

    def first_day(self, signal: str, decision: str | None = None) -> int | None:
        for entry in self.decision_log:
            if entry["signal"] == signal and (decision is None or entry["decision"] == decision):
                return entry["day"]
        return None

    # --------------------------------------------------------------- one day

    def observe_day(self, day: int, features, labels_delayed) -> dict:
        self.day = int(day)
        features = np.asarray(features, dtype=np.float64)
        report = self._watch_the_glass(features)
        record = self._serve(features)
        guardrails = self._check_guardrails(record)
        scored = self._score_late_labels(labels_delayed)
        decision = None
        holding = self.hold_until is not None and self.day < self.hold_until
        enough = self.baseline_scores and sum(len(s) for s in self.canary_scores) >= MIN_CANARY_SCORES
        if self.state == "canary" and scored is not None and not holding and enough:
            decision = self._decide(guardrails)
        return {
            "day": self.day,
            "state": self.state,
            "canary_fraction": self.canary_fraction,
            "drift": report,
            "guardrails": guardrails,
            "scored_day": None if scored is None else scored["day"],
            "decision": decision,
        }

    def _watch_the_glass(self, features: np.ndarray) -> dict:
        report = self.monitor.observe(features)
        if self.state == "watching" and report["drifted"]:
            self.drift_day = self.day
            self.state = "canary"
            self.canary_start_day = self.day
            self.canary_fraction = self.ramp[0]
            feats = report["drifted_features"]
            self.log(
                self.day,
                "input_drift",
                report["max_psi"],
                "start_canary",
                f"PSI {report['max_psi']:.3f} / KS {report['max_ks']:.3f} on feature(s) {feats} exceed "
                f"{self.monitor.thresholds}; starting {self.candidate.name} on {self.canary_fraction:.0%} of traffic",
                psi=report["psi"],
                ks=report["ks"],
                features=feats,
            )
        return report

    def _serve(self, features: np.ndarray) -> DayRecord:
        n = features.shape[0]
        if self.state == "canary":
            canary = np.array(
                [assign_arm(f"{self.day}-{i}", self.canary_fraction, self.salt) == "canary" for i in range(n)]
            )
        elif self.state == "promoted":
            canary = np.ones(n, dtype=bool)
        else:
            canary = np.zeros(n, dtype=bool)
        predictions = np.empty(n, dtype=np.int64)
        latency = np.empty(n, dtype=np.float64)
        if (~canary).any():
            predictions[~canary], latency[~canary] = self.deployed.serve(features[~canary], self.rng)
        if canary.any():
            predictions[canary], latency[canary] = self.candidate.serve(features[canary], self.rng)
        record = DayRecord(self.day, canary, predictions, latency)
        self.records[self.day] = record
        return record

    def _check_guardrails(self, record: DayRecord) -> dict:
        if self.state != "canary" or not record.canary.any():
            return {"passed": True, "violations": [], "details": {}}
        for latency, prediction in zip(record.latency_ms[record.canary], record.predictions[record.canary]):
            self.canary_latency.push(latency)
            self.canary_failures.push(1.0 if prediction == -1 else 0.0)
        metrics = canary_metrics(self.canary_latency, self.canary_failures)
        guardrails = guardrail_check(metrics, self.slo)
        if not guardrails["passed"]:
            worst = guardrails["violations"][0]
            self._rollback(
                "guardrail",
                float(metrics[worst]),
                f"canary {worst} = {metrics[worst]:.3g} breaks the SLO limit {self.slo[worst]:.3g} "
                f"(violations: {guardrails['violations']}); rolling back before any labels arrive",
                metrics=metrics,
                violations=guardrails["violations"],
            )
        return guardrails

    def _score_late_labels(self, labels_delayed) -> dict | None:
        scored_day = self.day - self.label_delay
        if labels_delayed is None or scored_day not in self.records:
            return None
        scored = score_record(self.records[scored_day], labels_delayed)
        scored["day"] = scored_day
        for event in self.alerts.observe("error_rate", scored["error_rate"]):
            self.log(
                self.day,
                "error_rate",
                scored["error_rate"],
                "alert" if event.kind == "fire" else "resolved",
                f"error rate of day {scored_day} is {scored['error_rate']:.3f} "
                f"({'above' if event.kind == 'fire' else 'back below'} the {event.kind} line; labels arrived "
                f"{self.label_delay} days late)",
                scored_day=scored_day,
            )
        if self.canary_start_day is not None and scored_day >= self.canary_start_day and self.records[scored_day].canary.any():
            if scored["baseline_scores"].size:
                self.baseline_scores.append(scored["baseline_scores"])
            if scored["canary_scores"].size:
                self.canary_scores.append(scored["canary_scores"])
        return scored

    def _decide(self, guardrails: dict) -> str:
        comparison = compare_arms(
            np.concatenate(self.baseline_scores),
            np.concatenate(self.canary_scores),
            n_boot=self.n_boot,
            alpha=self.alpha,
            rng=self.rng,
        )
        decision = release_decision(comparison, guardrails, self.min_effect)
        evidence = (
            f"delta {comparison['delta']:+.4f}, {100 * (1 - self.alpha):.0f}% CI "
            f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}], "
            f"n = {comparison['n_baseline']} baseline / {comparison['n_canary']} canary"
        )
        if decision == "rollback":
            self._rollback("canary", comparison["delta"], f"{evidence}: the canary is significantly worse", **comparison)
        elif decision == "promote":
            new_fraction = advance_ramp(self.canary_fraction, "promote", self.ramp)
            complete = new_fraction >= 1.0
            # Tomorrow is the first day served at the new fraction; its labels arrive label_delay days later.
            # Until then there is no fresh evidence, so do not decide again on the same numbers.
            self.hold_until = self.day + 1 + self.label_delay
            self.log(
                self.day,
                "canary",
                comparison["delta"],
                "promote",
                f"{evidence}: significantly better and guardrails hold; "
                + (
                    "release complete at 100%"
                    if complete
                    else f"ramping {self.canary_fraction:.0%} -> {new_fraction:.0%}; next decision on day {self.hold_until}"
                ),
                fraction_before=self.canary_fraction,
                fraction_after=new_fraction,
                guardrails_passed=True,
                **comparison,
            )
            self.canary_fraction = new_fraction
            if complete:
                self.state = "promoted"
        else:
            self.log(
                self.day,
                "canary",
                comparison["delta"],
                "extend",
                f"{evidence}: the interval still covers zero; keep collecting at {self.canary_fraction:.0%}",
                fraction=self.canary_fraction,
                **comparison,
            )
        return decision

    def _rollback(self, signal: str, value: float, reason: str, **details) -> None:
        self.state = "rolled_back"
        self.canary_fraction = advance_ramp(self.canary_fraction, "rollback", self.ramp)
        self.log(self.day, signal, value, "rollback", reason, **details)


def render_decision_log(log) -> str:
    """The audit trail as a markdown table: | day | signal | value | decision | reason |."""

    def cell(text) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = ["| day | signal | value | decision | reason |", "|---|---|---|---|---|"]
    for entry in log:
        lines.append(
            f"| {entry['day']} | {cell(entry['signal'])} | {entry['value']:.4f} | "
            f"{cell(entry['decision'])} | {cell(entry['reason'])} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# THE DRIFT PROPHECY
# ---------------------------------------------------------------------------
DRIFT_PROPHECY: dict[str, str | None] = {
    "under slow covariate drift, which signal moves first: input_drift or error_rate": "input_drift",
    "a slow tail grows on 3% of requests: which percentile moves first, p50 or p99": "p99",
    "a 0.5% canary at 1000 requests/day sees a 2-point accuracy change within a week: yes or no": "no",
}
