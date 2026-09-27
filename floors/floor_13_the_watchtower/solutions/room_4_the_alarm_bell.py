"""ROOM 13.4 - THE ALARM BELL  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Hysteresis is two lines and a boolean: fire when the value crosses the high
line, clear only when it drops below the low line, do nothing in between. The
manager adds a cooldown on top so that a metric which does cross back and
forth cannot page you twice in five minutes. The error budget is arithmetic;
write it down once and stop redoing it in your head during incidents.
"""

from __future__ import annotations

import math
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

    def push(self, value: float) -> None:
        self._values.append(float(value))

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> list[float]:
        return list(self._values)

    def percentile(self, p: float) -> float:
        """Sort; take the element at 1-based rank ceil(p / 100 * n), rank 0 becoming 1."""
        if not self._values:
            raise ValueError("the window is empty")
        if not 0.0 <= p <= 100.0:
            raise ValueError("p must be in [0, 100]")
        ordered = sorted(self._values)
        rank = max(1, math.ceil(p / 100.0 * len(ordered)))
        return ordered[rank - 1]

    def mean(self) -> float:
        if not self._values:
            raise ValueError("the window is empty")
        return sum(self._values) / len(self._values)

    def rate(self, predicate: Callable[[float], bool]) -> float:
        """Fraction of the values for which ``predicate`` is true."""
        if not self._values:
            raise ValueError("the window is empty")
        return sum(1 for v in self._values if predicate(v)) / len(self._values)


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
        """"fired" on the inactive->active edge, "resolved" on active->inactive, else None."""
        if not self.active and value > self.fire_above:
            self.active = True
            return "fired"
        if self.active and value < self.clear_below:
            self.active = False
            return "resolved"
        return None


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
        self._announced: set[str] = set()

    def add_rule(self, name: str, threshold: Threshold, severity: str = "page") -> None:
        self.rules[name] = (threshold, severity)

    @property
    def active(self) -> set[str]:
        return {name for name, (threshold, _severity) in self.rules.items() if threshold.active}

    def observe(self, name: str, value: float) -> list[Alert]:
        """Feed one value to rule ``name``; return the events emitted (zero or one)."""
        if name not in self.rules:
            raise KeyError(f"no alert rule named {name!r}")
        threshold, severity = self.rules[name]
        now = float(self.clock())
        edge = threshold.update(value)
        emitted: list[Alert] = []
        if edge == "fired":
            last = self.last_fired.get(name)
            if last is not None and now - last < self.cooldown:
                self.suppressed += 1  # a repeat inside the cooldown: the pager stays quiet
            else:
                emitted.append(Alert(name, severity, float(value), now, "fire"))
                self.last_fired[name] = now
                self._announced.add(name)
        elif edge == "resolved" and name in self._announced:
            emitted.append(Alert(name, severity, float(value), now, "resolved"))
            self._announced.discard(name)
        self.events.extend(emitted)
        return emitted


# ------------------------------------------------------------- error budget


def error_budget(slo_target: float, window_requests: int, failures: int) -> dict:
    """allowed = (1 - target) * requests; remaining = (allowed - failures) / allowed;
    burn_rate = observed failure rate / allowed failure rate."""
    if not 0.0 < slo_target < 1.0:
        raise ValueError("slo_target must be strictly between 0 and 1")
    if window_requests <= 0:
        raise ValueError("window_requests must be positive")
    if failures < 0:
        raise ValueError("failures cannot be negative")
    allowed = (1.0 - slo_target) * window_requests
    return {
        "allowed_failures": allowed,
        "remaining_fraction": (allowed - failures) / allowed,
        "burn_rate": (failures / window_requests) / (1.0 - slo_target),
    }
