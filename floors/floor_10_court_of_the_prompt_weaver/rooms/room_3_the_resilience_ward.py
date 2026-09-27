"""ROOM 10.3 - THE RESILIENCE WARD

    The heralds are unreliable. Some are slow, some are asleep, one has a
    rate limit and a clipboard. The court does not stop for them. It waits a
    little, then a little more, and if a herald is truly gone it stops
    knocking on that door for a while.

Four wards, each a small piece of production engineering:

* **Backoff with full jitter.** Retry k waits uniform(0, min(cap, base * 2**k)).
  The doubling spreads load; the *randomness* is what stops a thousand
  clients that failed together from retrying together (the thundering herd).
  Without jitter, synchronised retries hit the recovering server in waves.
* **A retry wrapper** that retries only what can succeed on retry (rate limits,
  timeouts, 5xx, connection errors), gives up after ``max_attempts`` with the
  last error chained, honours ``retry_after`` when the server states one, and
  never sleeps after the final failure (nobody is waiting for that sleep).
* **Idempotency.** A retry can duplicate a side effect: the request reached the
  server, the reply was lost, you sent it again. Cache completed results by a
  request key so a duplicate returns the remembered answer without running.
* **A circuit breaker.** After N consecutive failures the door is *open*: calls
  are rejected immediately instead of each waiting for a timeout. After
  ``recovery_time`` the door is *half-open*: one probe call goes through. If it
  succeeds the breaker closes; if it fails the breaker opens again.

The trials never sleep. They inject a recording ``sleep`` and a seeded ``rng``
and check the numbers. Write your code so that they can.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from dungeon.artifacts.llm import RETRYABLE_ERRORS


class RetriesExhausted(Exception):
    """Every attempt failed with a retryable error. ``__cause__`` is the last one."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"gave up after {attempts} attempt(s): {type(last_error).__name__}: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class CircuitOpen(Exception):
    """The breaker is open: the call was rejected without being attempted."""


def backoff_delay(
    attempt: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: bool = True,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (0 = the first retry).

    ceiling = min(cap, base * 2**attempt). With ``jitter`` draw
    ``uniform(0.0, ceiling)`` from ``rng`` (or the ``random`` module when rng
    is None); without it return the ceiling itself.
    """
    raise NotImplementedError("backoff_delay() is unwritten")


def with_retries(
    fn: Callable[[], Any],
    *,
    max_attempts: int,
    base: float = 0.5,
    cap: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE_ERRORS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Any:
    """Call ``fn()`` until it returns, up to ``max_attempts`` times.

    * An error not in ``retry_on`` propagates immediately, untouched.
    * After a retryable failure that is not the last attempt: compute
      ``backoff_delay(attempt_index, base, cap, True, rng)``; if the error has a
      non-None ``retry_after`` attribute, the delay is at least that; call
      ``on_retry(attempt_number, error, delay)`` if given (attempt_number starts
      at 1); then ``sleep(delay)``.
    * After the last failure: no sleep. Raise ``RetriesExhausted(max_attempts, error)``
      *from* that error so ``__cause__`` is set.
    """
    raise NotImplementedError("with_retries() is unwritten")


class Idempotent:
    """Wrap ``fn`` so a repeated request (same key) returns the remembered result without running.

    The key is ``key(*args, **kwargs)`` if a key function is given, otherwise
    something derived from the arguments themselves (``repr`` of the args and
    sorted kwargs is fine). Only *successful* results are cached: a failed
    attempt did not complete the request, so a retry must run ``fn`` again.
    ``executions`` counts how many times ``fn`` actually ran.
    """

    def __init__(self, fn: Callable[..., Any], key: Callable[..., Any] | None = None):
        self.fn = fn
        self.key_fn = key
        self.cache: dict[Any, Any] = {}
        self.executions = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Idempotent.__call__() is unwritten")


class CircuitBreaker:
    """closed (normal) -> open (reject fast) -> half_open (one probe) -> closed, or open again.

    ``state`` is one of ``"closed"``, ``"open"``, ``"half_open"``. Use the
    injected ``clock()`` (seconds) for every time comparison so the trial can
    move time by hand.
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_time: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.clock = clock
        self.failures = 0  # consecutive failures
        self._state = self.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        """Current state. An open breaker whose recovery_time has passed reports half_open."""
        raise NotImplementedError("CircuitBreaker.state is unwritten")

    def allow(self) -> bool:
        """May a call be attempted right now? (Everything except open.)"""
        raise NotImplementedError("CircuitBreaker.allow() is unwritten")

    def record_success(self) -> None:
        """Reset the failure count and close the breaker."""
        raise NotImplementedError("CircuitBreaker.record_success() is unwritten")

    def record_failure(self) -> None:
        """Count it. Open the breaker (stamping the clock) if the threshold is reached or a half-open probe failed."""
        raise NotImplementedError("CircuitBreaker.record_failure() is unwritten")

    def call(self, fn: Callable[[], Any]) -> Any:
        """Raise CircuitOpen without calling ``fn`` when open; otherwise call, record, return or re-raise."""
        raise NotImplementedError("CircuitBreaker.call() is unwritten")
