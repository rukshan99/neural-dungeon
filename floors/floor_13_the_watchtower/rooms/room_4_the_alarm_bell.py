"""ROOM 13.4 - THE ALARM BELL

    A bronze bell hangs in the middle of the room, and beside it a ledger of
    every time it has rung. Most of the entries are crossed out. "Rang at
    dawn, stopped, rang again, stopped, rang again," reads one. "Nobody came
    the fourth time." The bell is not the problem. The line it was told to
    watch is.

Alerting is the part of monitoring that touches people, so its failure modes
are human ones: the bell that rings six times for one incident gets muted,
and the muted bell misses the real one. Four tools:

ROLLING WINDOW. The last ``size`` values of a metric (a deque). From it you read
NEAREST-RANK percentiles, exactly as on Floor 12: sort, take the element at
1-based rank ceil(p / 100 * n) (rank 0 becomes 1). p50 is the typical request;
p99 is the one in the contract. Also a mean and a ``rate(predicate)``: the
fraction of values for which the predicate holds, e.g. the error rate.

HYSTERESIS. One line to fire and a LOWER line to clear:

    inactive -> active   when value > fire_above     ("fired")
    active   -> inactive when value < clear_below    ("resolved")
    otherwise unchanged                              (None)

A metric hovering at 0.19, 0.21, 0.19, 0.22 around a fire line of 0.20 with a
clear line of 0.15 rings ONCE and resolves once. A plain threshold flaps.

ALERT MANAGER. Named rules (each a Threshold plus a severity), an injected
clock, and a COOLDOWN: a fire for the same rule within ``cooldown`` of the last
one is suppressed (counted, not emitted), and so is its quiet resolution.
Every emitted event is an ``Alert`` with ``kind`` "fire" or "resolved" and the
clock reading it happened at. Say when it is better, too: an incident that is
never resolved is an incident somebody is still awake for.

ERROR BUDGET. An SLO of 99.9% over 100,000 requests ALLOWS 100 failures.
``remaining_fraction`` is what is left of them; ``burn_rate`` is how fast you
are spending them, as observed failure rate / allowed failure rate. Burn rate
1.0 spends the budget exactly at the end of the window; 2.5 spends it in 40% of
the window. Alert on burn rate, not on individual failures.
"""

from __future__ import annotations

import math  # noqa: F401 - ceil for the nearest rank
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

# ------------------------------------------------------------------- window


class RollingWindow:
    """The last ``size`` values pushed. Nearest-rank percentiles, as on Floor 12."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        self.size = size
        self._values: deque[float] = deque(maxlen=size)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> list[float]:
        return list(self._values)

    def push(self, value: float) -> None:
        """Append ``float(value)``; the deque drops the oldest value once it is full."""
        raise NotImplementedError("RollingWindow.push() is unwritten")

    def percentile(self, p: float) -> float:
        """Nearest rank: sort, take the element at 1-based rank ceil(p / 100 * n), rank 0 becoming 1.

        Raise ValueError for an empty window or p outside [0, 100]. p0 is the minimum, p100 the maximum.
        """
        raise NotImplementedError("RollingWindow.percentile() is unwritten")

    def mean(self) -> float:
        """Arithmetic mean of the values; ValueError when empty."""
        raise NotImplementedError("RollingWindow.mean() is unwritten")

    def rate(self, predicate: Callable[[float], bool]) -> float:
        """Fraction of the values for which ``predicate(value)`` is true; ValueError when empty."""
        raise NotImplementedError("RollingWindow.rate() is unwritten")


# --------------------------------------------------------------- hysteresis


class Threshold:
    """Fires when value > fire_above; clears only when value < clear_below."""

    def __init__(self, fire_above: float, clear_below: float) -> None:
        if clear_below > fire_above:
            raise ValueError("clear_below must not exceed fire_above")
        self.fire_above = float(fire_above)
        self.clear_below = float(clear_below)
        self.active = False

    def update(self, value: float) -> str | None:
        """Apply one value. Return "fired" on the inactive->active edge, "resolved" on active->inactive, else None."""
        raise NotImplementedError("Threshold.update() is unwritten")


# ------------------------------------------------------------------- alerts


@dataclass(frozen=True)
class Alert:
    name: str
    severity: str
    value: float
    fired_at: float  # the clock reading when this event was emitted
    kind: str = "fire"  # "fire" | "resolved"


class AlertManager:
    """Named hysteresis rules, a cooldown between fire notifications, and an event log."""

    def __init__(self, clock: Callable[[], float], cooldown: float = 0.0) -> None:
        self.clock = clock
        self.cooldown = float(cooldown)
        self.rules: dict[str, tuple[Threshold, str]] = {}
        self.events: list[Alert] = []
        self.last_fired: dict[str, float] = {}
        self.suppressed = 0
        self._announced: set[str] = set()  # rules whose current firing was actually emitted

    def add_rule(self, name: str, threshold: Threshold, severity: str = "page") -> None:
        self.rules[name] = (threshold, severity)

    @property
    def active(self) -> set[str]:
        return {name for name, (threshold, _severity) in self.rules.items() if threshold.active}

    def observe(self, name: str, value: float) -> list[Alert]:
        """Feed one value to rule ``name``; return the events emitted (an empty list or one Alert).

        KeyError for an unknown rule. Read ``now = self.clock()`` and ``edge = threshold.update(value)``.
        "fired": if ``name`` fired within the last ``cooldown`` (``now - last_fired[name] < cooldown``),
        increment ``suppressed`` and emit nothing; otherwise emit ``Alert(name, severity, value, now, "fire")``,
        record ``last_fired[name] = now`` and add ``name`` to ``_announced``.
        "resolved": emit ``Alert(..., "resolved")`` only if ``name`` is in ``_announced`` (a suppressed
        fire resolves silently), then discard it. Append every emitted event to ``self.events``.
        """
        raise NotImplementedError("AlertManager.observe() is unwritten")


# ------------------------------------------------------------- error budget


def error_budget(slo_target: float, window_requests: int, failures: int) -> dict:
    """{"allowed_failures": (1 - slo_target) * window_requests,
        "remaining_fraction": (allowed - failures) / allowed        (negative when over budget),
        "burn_rate": (failures / window_requests) / (1 - slo_target)}

    Raise ValueError unless 0 < slo_target < 1, window_requests > 0 and failures >= 0.
    """
    raise NotImplementedError("error_budget() is unwritten")
