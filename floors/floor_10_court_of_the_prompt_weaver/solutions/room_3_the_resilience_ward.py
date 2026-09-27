"""ROOM 10.3 - THE RESILIENCE WARD  (reference solution)

Spoilers below. The stub you are meant to edit is in ../rooms/.

Four wards against a flaky world:

* exponential backoff with FULL jitter: delay ~ uniform(0, min(cap, base * 2**attempt))
* a retry wrapper that honours Retry-After, never retries a 4xx, never sleeps
  after the final failure, and chains the last error when it gives up
* idempotency: a retried duplicate must not run the side effect twice
* a circuit breaker: closed -> open after N failures -> half-open after a rest
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


# ------------------------------------------------------------------- backoff


def backoff_delay(
    attempt: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: bool = True,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (0 for the first retry).

    ceiling = min(cap, base * 2**attempt); with jitter the delay is drawn
    uniformly from [0, ceiling] ("full jitter"), otherwise it is the ceiling.
    """
    ceiling = min(cap, base * (2**attempt))
    if not jitter:
        return ceiling
    return (rng or random).uniform(0.0, ceiling)


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
    """Call ``fn()`` up to ``max_attempts`` times, backing off between attempts.

    Errors outside ``retry_on`` propagate at once (nothing about a bad request
    improves with time). A ``retry_after`` attribute on the error, when present,
    is a floor on the delay. ``on_retry(attempt_number, error, delay)`` is
    called before each sleep. There is no sleep after the last failure.
    """
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = backoff_delay(attempt, base, cap, True, rng)
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = max(delay, float(retry_after))
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
    assert last is not None  # max_attempts >= 1, so the loop ran and failed
    raise RetriesExhausted(max_attempts, last) from last


# --------------------------------------------------------------- idempotency


class Idempotent:
    """Remember the result of each request key; a duplicate returns it without running ``fn``.

    Only *successful* results are cached. A failed attempt is not a completed
    request, so retrying it must run ``fn`` again.
    """

    def __init__(self, fn: Callable[..., Any], key: Callable[..., Any] | None = None):
        self.fn = fn
        self.key_fn = key
        self.cache: dict[Any, Any] = {}
        self.executions = 0

    def _key(self, args: tuple, kwargs: dict) -> Any:
        if self.key_fn is not None:
            return self.key_fn(*args, **kwargs)
        return repr((args, sorted(kwargs.items())))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._key(args, kwargs)
        if key in self.cache:
            return self.cache[key]
        self.executions += 1
        result = self.fn(*args, **kwargs)
        self.cache[key] = result
        return result


# ------------------------------------------------------------ circuit breaker


class CircuitBreaker:
    """closed (normal) -> open (reject fast) -> half_open (one probe) -> closed or open again."""

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
        self.failures = 0
        self._state = self.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._state == self.OPEN and self._opened_at is not None:
            if self.clock() - self._opened_at >= self.recovery_time:
                self._state = self.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        return self.state != self.OPEN

    def record_success(self) -> None:
        self.failures = 0
        self._state = self.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == self.HALF_OPEN or self.failures >= self.failure_threshold:
            self._state = self.OPEN
            self._opened_at = self.clock()

    def call(self, fn: Callable[[], Any]) -> Any:
        if not self.allow():
            raise CircuitOpen(
                f"circuit open after {self.failures} failure(s); retry in "
                f"{self.recovery_time - (self.clock() - (self._opened_at or 0.0)):.1f}s"
            )
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
