"""BOSS FIGHT - THE SILENT DRIFT

Phase 1: the glass sees the inputs move within a few days, and never before they do.
Phase 2: a canary of the retrained scribe is promoted on evidence, a worse one is rolled back
         on the numbers, a slow or brittle one is rolled back on guardrails before any label arrives.
Phase 3: the decision log reads as an audit trail with its evidence.
Phase 4: the prophecy: which signal moves first, which percentile sees a tail, what a tiny canary can see.
"""

import functools
import math

import numpy as np
import pytest

from dungeon.trials import load_room
from floors.floor_13_the_watchtower.assets.deployment import (
    DEPLOYED,
    FLAKY,
    RETRAINED,
    SLOW,
    WORSE,
    reference_batch,
    simulate_days,
    with_label_delay,
)

boss = load_room(__file__, "boss_the_silent_drift")
room4 = load_room(__file__, "room_4_the_alarm_bell")

pytestmark = pytest.mark.boss

N_DAYS, DRIFT_DAY, LABEL_DELAY, N_PER_DAY = 60, 20, 5, 500
K = 7  # the glass must see the drift within K days of its onset (the reference solution takes 2-3)
SEEDS = (0, 1, 2)
CANDIDATES = {"retrained": RETRAINED, "worse": WORSE, "slow": SLOW, "flaky": FLAKY}
LOG_FIELDS = ("day", "signal", "value", "decision", "reason")


@functools.lru_cache(maxsize=None)
def _run(seed, candidate="retrained", n_days=N_DAYS, drift_day=DRIFT_DAY):
    days = list(simulate_days(n_days, drift_day, seed, n_per_day=N_PER_DAY))
    tower = boss.Watchtower(reference_batch(seed=100 + seed), DEPLOYED, CANDIDATES[candidate], label_delay=LABEL_DELAY, seed=seed)
    for day, late_labels in with_label_delay(days, LABEL_DELAY):
        tower.observe_day(day.day, day.features, late_labels)
    return tower


def _entries(tower, signal=None, decision=None):
    return [
        e for e in tower.decision_log
        if (signal is None or e["signal"] == signal) and (decision is None or e["decision"] == decision)
    ]


# --------------------------------------------------------------------- phase 1
@pytest.mark.parametrize("seed", SEEDS)
def test_phase_1_the_glass_sees_the_drift_within_k_days(seed):
    tower = _run(seed)
    assert tower.drift_day is not None, "The mean of x2 climbs from day 20 and the glass never noticed. Compare each day's features with the reference."
    assert DRIFT_DAY <= tower.drift_day <= DRIFT_DAY + K, (
        f"Drift starts on day {DRIFT_DAY}; you detected it on day {tower.drift_day}. It must be within {K} days of onset and never before it."
    )
    starts = _entries(tower, "input_drift", "start_canary")
    assert starts and starts[0]["day"] == tower.drift_day, "Detection must be logged as signal 'input_drift', decision 'start_canary', on the day it happened."
    assert tower.canary_start_day == tower.drift_day, "The canary starts the day the glass speaks."


@pytest.mark.parametrize("seed", SEEDS)
def test_phase_1_the_glass_never_cries_wolf_before_the_onset(seed):
    tower = _run(seed)
    early = [e for e in _entries(tower, "input_drift") if e["day"] < DRIFT_DAY]
    assert not early, f"Input drift was logged on day(s) {[e['day'] for e in early]}, before anything moved. Your thresholds are inside the sampling noise."
    quiet = _run(seed, "retrained", n_days=40, drift_day=10**6)
    assert quiet.drift_day is None and quiet.state == "watching", "Forty days without drift: the glass must stay dark and the tower must keep watching."
    assert quiet.decision_log == [], f"Nothing happened, so nothing should be decided; the log has {len(quiet.decision_log)} entries."


# --------------------------------------------------------------------- phase 2
@pytest.mark.parametrize("seed", SEEDS)
def test_phase_2_the_retrained_scribe_is_promoted_only_on_evidence(seed):
    tower = _run(seed)
    assert tower.state == "promoted", f"The retrained scribe is 2-10 points better on drifted traffic; by day {N_DAYS} it should be promoted. State: {tower.state!r}."
    canary_entries = _entries(tower, "canary")
    assert canary_entries, "Every canary comparison is a decision; log it under signal 'canary'."
    assert {e["decision"] for e in canary_entries} <= {"extend", "promote", "rollback"}
    first = canary_entries[0]["day"]
    assert first >= tower.canary_start_day + LABEL_DELAY, (
        f"The first quality decision came on day {first}, but the canary started on day {tower.canary_start_day} and labels arrive {LABEL_DELAY} days late. "
        "You cannot compare accuracies before the first canary labels exist."
    )
    first_n = canary_entries[0].get("details", {}).get("n_canary", 0)
    assert first_n >= boss.MIN_CANARY_SCORES, (
        f"The first quality decision used {first_n} labelled canary requests. Floor 11's arithmetic: fifty have a standard error of "
        f"four points, which is how better models get rolled back on a coin flip. Wait for MIN_CANARY_SCORES = {boss.MIN_CANARY_SCORES}."
    )
    promotes = _entries(tower, "canary", "promote")
    assert promotes, "Promotion never happened."
    for entry in promotes:
        details = entry.get("details", {})
        assert details.get("ci_low", -1.0) > 0.0, (
            f"Day {entry['day']}: a promotion without significance. The CI must exclude zero on the good side; details: {details}."
        )
        assert details.get("guardrails_passed") is True, "Promotion also needs the guardrails to hold; record that in the details."
    assert promotes[-1].get("details", {}).get("fraction_after") == 1.0, "The final promote takes the canary to 100% of traffic."
    for earlier, later in zip(promotes, promotes[1:]):
        assert later["day"] - earlier["day"] >= LABEL_DELAY + 1, (
            f"Ramp steps on days {earlier['day']} and {later['day']}: the second decision used the same evidence as the first. "
            f"After a step, wait until labels from traffic served at the NEW fraction have arrived ({LABEL_DELAY} days)."
        )


def test_phase_2_a_worse_retrain_is_rolled_back_on_the_numbers():
    tower = _run(0, "worse")
    assert tower.state == "rolled_back", f"The hasty retrain is 25 points worse; the tower must roll it back. State: {tower.state!r}."
    assert not _entries(tower, decision="promote"), "A worse model must never be promoted."
    rollbacks = _entries(tower, decision="rollback")
    assert len(rollbacks) == 1, f"Roll back exactly once; got {len(rollbacks)} rollback entries."
    rollback = rollbacks[0]
    assert rollback["signal"] == "canary", f"This rollback is a quality decision (signal 'canary'), not a guardrail; got {rollback['signal']!r}."
    assert rollback["day"] >= tower.canary_start_day + LABEL_DELAY, "Accuracy cannot be judged before labels arrive."
    details = rollback.get("details", {})
    assert details.get("ci_high", 1.0) < 0.0, f"Rollback on quality needs the CI entirely below zero; details: {details}."
    assert tower.canary_fraction == 0.0, "After a rollback nobody is served by the candidate."
    assert not [e for e in _entries(tower, "canary") if e["day"] > rollback["day"]], "After the rollback the canary is over: no more canary decisions."


@pytest.mark.parametrize("candidate,violated", [("slow", "p95_latency_ms"), ("flaky", "error_rate")])
def test_phase_2_a_guardrail_breach_rolls_back_before_any_label_arrives(candidate, violated):
    tower = _run(0, candidate)
    assert tower.state == "rolled_back", f"The {candidate} scribe breaks the SLO; state is {tower.state!r}."
    rollbacks = _entries(tower, decision="rollback")
    assert len(rollbacks) == 1 and rollbacks[0]["signal"] == "guardrail", f"Expected one rollback with signal 'guardrail'; got {rollbacks}."
    rollback = rollbacks[0]
    assert rollback["day"] < tower.canary_start_day + LABEL_DELAY, (
        f"Latency and errors are known the day they happen. The rollback came on day {rollback['day']}, "
        f"but the canary started on day {tower.canary_start_day}: you waited for labels you did not need."
    )
    assert violated in rollback.get("details", {}).get("violations", []) or violated in rollback["reason"], (
        f"Name the guardrail that broke ({violated}) in the reason or details; got {rollback['reason']!r}."
    )
    assert not _entries(tower, decision="promote")


def test_phase_2_score_record_joins_late_labels_by_arm():
    record = boss.DayRecord(
        day=3,
        canary=np.array([True, False, True, False]),
        predictions=np.array([1, 0, -1, 1]),
        latency_ms=np.array([300.0, 310.0, 320.0, 330.0]),
    )
    scored = boss.score_record(record, np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(scored["correct"], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(scored["baseline_scores"], [1.0, 0.0], err_msg="Baseline scores are the correct flags where canary is False.")
    np.testing.assert_array_equal(scored["canary_scores"], [1.0, 0.0], err_msg="A failed request (-1) is a wrong answer.")
    assert math.isclose(scored["error_rate"], 0.5)
    with pytest.raises(ValueError):
        boss.score_record(record, np.array([1, 0]))


def test_phase_2_canary_metrics_read_the_pooled_windows():
    latency = room4.RollingWindow(200)
    failures = room4.RollingWindow(200)
    for value in range(1, 101):
        latency.push(float(value))
        failures.push(1.0 if value <= 3 else 0.0)
    metrics = boss.canary_metrics(latency, failures)
    assert metrics["p95_latency_ms"] == 95.0, "Nearest-rank p95 of 1..100 is 95."
    assert math.isclose(metrics["error_rate"], 0.03), "Three failures in a hundred requests."


# --------------------------------------------------------------------- phase 3
def test_phase_3_the_ledger_has_five_columns_and_quotes_its_evidence():
    tower = _run(0)
    log = tower.decision_log
    assert log, "The decision log is empty."
    for entry in log:
        for key in LOG_FIELDS:
            assert key in entry, f"Every entry needs {LOG_FIELDS}; this one has {sorted(entry)}."
        assert isinstance(entry["day"], int) and isinstance(entry["value"], float)
        assert entry["reason"] and any(ch.isdigit() for ch in entry["reason"]), f"Quote the numbers in the reason; got {entry['reason']!r}."
    days = [e["day"] for e in log]
    assert days == sorted(days), "The ledger is written in order."
    assert log[0]["signal"] == "input_drift" and log[0]["decision"] == "start_canary", "The first entry is the glass speaking."
    assert {"input_drift", "canary"} <= {e["signal"] for e in log}


def test_phase_3_the_bell_rings_late_because_labels_do():
    tower = _run(0, "worse")  # rolled back, so the deployed scribe keeps decaying and the bell must ring
    alerts = _entries(tower, "error_rate", "alert")
    assert alerts, "The deployed scribe's error rate passes 15% weeks after the drift began; the bell (an error_rate alert) must ring."
    alert = alerts[0]
    assert alert["value"] > boss.ERROR_RATE_FIRE
    assert alert["day"] >= tower.drift_day + LABEL_DELAY, "The bell cannot ring about a day whose labels have not arrived."
    assert alert["day"] > tower.drift_day, "The glass spoke first; the bell rang later. That is the whole point of watching the inputs."


def test_phase_3_render_decision_log_is_a_markdown_table():
    tower = _run(0)
    text = boss.render_decision_log(tower.decision_log)
    lines = [line for line in text.splitlines() if line.startswith("|")]
    assert len(lines) == len(tower.decision_log) + 2, "A header row, a separator, and one row per entry."
    header = lines[0].lower()
    for column in LOG_FIELDS:
        assert column in header, f"The header must name the column {column!r}."
    weird = boss.render_decision_log([{"day": 1, "signal": "a|b", "value": 0.5, "decision": "extend", "reason": "x|y"}])
    assert "a\\|b" in weird and "x\\|y" in weird, "Escape '|' inside cells or the table breaks."


# --------------------------------------------------------------------- phase 4
PROPHECY_KEYS = {
    "under slow covariate drift, which signal moves first: input_drift or error_rate": ("input_drift", "error_rate"),
    "a slow tail grows on 3% of requests: which percentile moves first, p50 or p99": ("p50", "p99"),
    "a 0.5% canary at 1000 requests/day sees a 2-point accuracy change within a week: yes or no": ("yes", "no"),
}


def test_phase_4_the_prophecy_is_complete():
    assert set(boss.DRIFT_PROPHECY) == set(PROPHECY_KEYS), "Do not rename or remove the prophecy's questions; answer them."


def _prophecy(key):
    prediction = boss.DRIFT_PROPHECY.get(key)
    if prediction is None:
        raise NotImplementedError(f"You have not prophesied {key!r}. Fill in DRIFT_PROPHECY.")
    assert prediction in PROPHECY_KEYS[key], f"Answer with one of {PROPHECY_KEYS[key]}, not {prediction!r}."
    return prediction


def test_phase_4_which_signal_moves_first():
    key = "under slow covariate drift, which signal moves first: input_drift or error_rate"
    prediction = _prophecy(key)
    tower = _run(0, "worse")
    glass_day, bell_day = tower.first_day("input_drift"), tower.first_day("error_rate")
    assert glass_day is not None and bell_day is not None, "This question needs both signals in the log of the rolled-back run."
    truth = "input_drift" if glass_day < bell_day else "error_rate"
    assert prediction == truth, (
        f"The glass spoke on day {glass_day} and the bell rang on day {bell_day}: {truth!r} moves first, not {prediction!r}. "
        "Accuracy needs labels, labels arrive late, and the drift has to grow before accuracy suffers at all."
    )


def test_phase_4_which_percentile_sees_a_slow_tail():
    key = "a slow tail grows on 3% of requests: which percentile moves first, p50 or p99"
    prediction = _prophecy(key)
    rng = np.random.default_rng(7)
    before = 300.0 * np.exp(0.4 * rng.standard_normal(20_000))
    after = before.copy()
    after[rng.random(20_000) < 0.03] *= 6.0
    change = {p: np.percentile(after, p) / np.percentile(before, p) - 1.0 for p in (50, 99)}
    truth = "p99" if change[99] > change[50] else "p50"
    assert prediction == truth, (
        f"p50 moved {change[50]:+.1%} and p99 moved {change[99]:+.1%}: {truth} sees a 3% tail, {prediction} does not. "
        "The median is blind to anything that happens to fewer than half the requests."
    )


def test_phase_4_what_a_tiny_canary_can_see_in_a_week():
    key = "a 0.5% canary at 1000 requests/day sees a 2-point accuracy change within a week: yes or no"
    prediction = _prophecy(key)
    n_canary = int(0.005 * 1000 * 7)
    se = math.sqrt(0.9 * 0.1 / n_canary)
    needed = math.ceil(1.96**2 * 0.9 * 0.1 / 0.02**2)
    truth = "yes" if 0.02 >= 1.96 * se else "no"
    assert prediction == truth, (
        f"A week of a 0.5% canary is {n_canary} requests. The standard error of an accuracy near 0.9 is {se:.3f}, so two points is "
        f"{0.02 / se:.1f} standard errors; Floor 11's formula says you need {needed} requests to pin down +-2 points. The answer is {truth!r}."
    )
