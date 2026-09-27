"""TRIAL 13.4 - THE ALARM BELL

A rolling window with nearest-rank percentiles, a threshold with hysteresis
that rings once instead of six times, an alert manager that deduplicates
inside its cooldown and says when things are better, and error-budget
arithmetic that is exact.
"""

import math

import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_4_the_alarm_bell")


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------- window
def test_the_window_keeps_only_the_last_size_values():
    window = room.RollingWindow(3)
    for value in (1.0, 2.0, 3.0, 4.0):
        window.push(value)
    assert len(window) == 3 and window.values == [2.0, 3.0, 4.0], f"size=3 keeps the last three pushes; got {window.values}."


def test_percentiles_use_nearest_rank_as_on_floor_12():
    window = room.RollingWindow(10)
    for value in (50.0, 10.0, 40.0, 20.0, 30.0):
        window.push(value)
    assert window.percentile(50) == 30.0, "Sorted [10, 20, 30, 40, 50]: rank ceil(0.5 * 5) = 3 -> 30."
    assert window.percentile(95) == 50.0, "rank ceil(0.95 * 5) = 5 -> 50. Nearest rank never interpolates."
    assert window.percentile(0) == 10.0 and window.percentile(100) == 50.0, "p0 is the minimum, p100 the maximum."
    big = room.RollingWindow(100)
    for value in range(1, 101):
        big.push(float(value))
    assert big.percentile(99) == 99.0, "1..100: rank ceil(0.99 * 100) = 99 -> 99."
    with pytest.raises(ValueError):
        window.percentile(101)
    with pytest.raises(ValueError):
        room.RollingWindow(5).percentile(50)


def test_mean_and_rate():
    window = room.RollingWindow(10)
    for value in (100.0, 200.0, 300.0, 1500.0):
        window.push(value)
    assert math.isclose(window.mean(), 525.0)
    assert math.isclose(window.rate(lambda v: v > 1000.0), 0.25), "One of four values is over 1000: rate 0.25."
    with pytest.raises(ValueError):
        room.RollingWindow(5).mean()


# ----------------------------------------------------------------- hysteresis
def test_a_flapping_metric_rings_once_and_resolves_once():
    threshold = room.Threshold(fire_above=0.20, clear_below=0.15)
    edges = [threshold.update(v) for v in (0.10, 0.21, 0.19, 0.22, 0.18, 0.21, 0.10, 0.10)]
    assert edges == [None, "fired", None, None, None, None, "resolved", None], (
        f"With hysteresis the metric hovering around 0.2 fires ONCE (at 0.21) and resolves ONCE (at 0.10, below 0.15); got {edges}. "
        "A plain threshold would have flipped six times."
    )
    assert threshold.active is False


def test_the_threshold_stays_active_between_the_lines():
    threshold = room.Threshold(0.20, 0.15)
    assert threshold.update(0.25) == "fired" and threshold.active is True
    assert threshold.update(0.17) is None and threshold.active is True, "0.17 is below the fire line but above the clear line: still active."
    assert threshold.update(0.14) == "resolved" and threshold.active is False
    with pytest.raises(ValueError):
        room.Threshold(fire_above=0.1, clear_below=0.2)


# --------------------------------------------------------------------- alerts
def test_the_manager_emits_fire_and_resolved_events_with_the_clock_time():
    clock = FakeClock(100.0)
    manager = room.AlertManager(clock, cooldown=0.0)
    manager.add_rule("p99_latency_s", room.Threshold(1.0, 0.8), severity="page")
    assert manager.observe("p99_latency_s", 0.5) == []
    clock.t = 101.0
    fired = manager.observe("p99_latency_s", 1.3)
    assert len(fired) == 1 and isinstance(fired[0], room.Alert)
    alert = fired[0]
    assert (alert.name, alert.severity, alert.value, alert.fired_at, alert.kind) == ("p99_latency_s", "page", 1.3, 101.0, "fire"), f"Got {alert!r}."
    assert manager.active == {"p99_latency_s"}
    clock.t = 105.0
    resolved = manager.observe("p99_latency_s", 0.6)
    assert len(resolved) == 1 and resolved[0].kind == "resolved" and resolved[0].fired_at == 105.0, "Say when it is better, too: a resolved event."
    assert manager.active == set()
    assert [e.kind for e in manager.events] == ["fire", "resolved"]
    with pytest.raises(KeyError):
        manager.observe("no_such_rule", 1.0)


def test_repeats_inside_the_cooldown_are_deduplicated():
    clock = FakeClock(0.0)
    manager = room.AlertManager(clock, cooldown=5.0)
    manager.add_rule("error_rate", room.Threshold(0.2, 0.1))
    kinds = []
    for t, value in [(0.0, 0.3), (1.0, 0.05), (2.0, 0.3), (3.0, 0.05), (10.0, 0.3)]:
        clock.t = t
        kinds += [e.kind for e in manager.observe("error_rate", value)]
    assert kinds == ["fire", "resolved", "fire"], (
        f"fire at t=0, resolve at t=1, the re-fire at t=2 is inside the 5 s cooldown (suppressed, and so is its quiet resolution), "
        f"the re-fire at t=10 is a new page. Expected ['fire', 'resolved', 'fire'], got {kinds}."
    )
    assert manager.suppressed == 1


def test_hysteresis_through_the_manager_pages_once_for_a_flapping_metric():
    clock = FakeClock(0.0)
    manager = room.AlertManager(clock, cooldown=0.0)
    manager.add_rule("error_rate", room.Threshold(0.20, 0.15))
    kinds = []
    for t, value in enumerate((0.10, 0.21, 0.19, 0.22, 0.18, 0.21, 0.10)):
        clock.t = float(t)
        kinds += [e.kind for e in manager.observe("error_rate", value)]
    assert kinds == ["fire", "resolved"], f"Exactly one fire and one resolve for a metric that hovers at the line; got {kinds}."


# --------------------------------------------------------------- error budget
def test_error_budget_arithmetic_is_exact():
    budget = room.error_budget(0.999, 100_000, failures=30)
    assert math.isclose(budget["allowed_failures"], 100.0), "99.9% over 100,000 requests allows 100 failures."
    assert math.isclose(budget["remaining_fraction"], 0.7), f"30 of 100 spent leaves 70%; got {budget['remaining_fraction']}."
    assert math.isclose(budget["burn_rate"], 0.3), f"Observed failure rate 0.0003 / allowed 0.001 = burn rate 0.3; got {budget['burn_rate']}."
    spent = room.error_budget(0.999, 100_000, failures=100)
    assert math.isclose(spent["remaining_fraction"], 0.0, abs_tol=1e-9) and math.isclose(spent["burn_rate"], 1.0), "Exactly at budget: nothing left, burning at 1x."
    over = room.error_budget(0.999, 100_000, failures=250)
    assert math.isclose(over["remaining_fraction"], -1.5) and math.isclose(over["burn_rate"], 2.5), "Over budget goes negative; burn rate 2.5x means the window's budget lasts 40% of the window."


def test_error_budget_rejects_nonsense():
    with pytest.raises(ValueError):
        room.error_budget(1.0, 1000, 0)
    with pytest.raises(ValueError):
        room.error_budget(0.99, 0, 0)
    with pytest.raises(ValueError):
        room.error_budget(0.99, 1000, -1)
