"""TRIAL 10.3 - THE RESILIENCE WARD

Nothing here sleeps. A recording sleep and a seeded rng stand in for time and
chance, and the numbers they record are judged.
"""

import random

import pytest

from dungeon.artifacts.llm import (
    FlakyLLM,
    InvalidRequestError,
    LLMTimeoutError,
    Message,
    RateLimitError,
    ScriptedLLM,
    ServerError,
)
from dungeon.trials import load_room

room = load_room(__file__, "room_3_the_resilience_ward")


class Recorder:
    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def flaky_court(schedule):
    inner = ScriptedLLM(["The court is in session."] * 20)
    flaky = FlakyLLM(inner, schedule)
    return flaky, (lambda: flaky.complete([Message.user("Is the court open?")]))


# -------------------------------------------------------------------- backoff
def test_backoff_without_jitter_doubles_until_the_cap():
    delays = [room.backoff_delay(a, base=0.5, cap=8.0, jitter=False) for a in range(6)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0], (
        f"Without jitter the delay is min(cap, base * 2**attempt): expected [0.5, 1, 2, 4, 8, 8], got {delays}"
    )


def test_full_jitter_stays_under_the_ceiling_and_actually_varies():
    rng = random.Random(7)
    for attempt in range(8):
        ceiling = min(8.0, 0.5 * 2**attempt)
        for _ in range(40):
            d = room.backoff_delay(attempt, base=0.5, cap=8.0, jitter=True, rng=rng)
            assert 0.0 <= d <= ceiling, f"attempt {attempt}: delay {d} is outside [0, {ceiling}]. Full jitter is uniform(0, ceiling)."
    draws = {round(room.backoff_delay(3, rng=rng), 6) for _ in range(20)}
    assert len(draws) > 1, "Twenty jittered delays came out identical. Jitter that does not vary is not jitter."


def test_full_jitter_draws_from_the_injected_rng():
    expected = random.Random(11).uniform(0.0, 2.0)
    got = room.backoff_delay(2, base=0.5, cap=8.0, jitter=True, rng=random.Random(11))
    assert got == pytest.approx(expected), (
        f"With rng=Random(11), attempt 2 (ceiling 2.0) should be rng.uniform(0, 2.0) = {expected}, got {got}. "
        "Use the rng you were given, not the global random module."
    )


# --------------------------------------------------------------- with_retries
def test_a_flaky_herald_eventually_answers():
    flaky, ask = flaky_court([ServerError("503"), LLMTimeoutError("slow"), None])
    rec = Recorder()
    result = room.with_retries(ask, max_attempts=5, base=0.5, cap=8.0, sleep=rec, rng=random.Random(0))
    assert result.text == "The court is in session."
    assert flaky.attempts == 3, f"Two failures then a success is three attempts; the herald was called {flaky.attempts} times."
    assert len(rec.sleeps) == 2, f"Two failures means two sleeps; you slept {len(rec.sleeps)} times: {rec.sleeps}"
    assert 0.0 <= rec.sleeps[0] <= 0.5, f"First retry ceiling is base=0.5; slept {rec.sleeps[0]}"
    assert 0.0 <= rec.sleeps[1] <= 1.0, f"Second retry ceiling is 1.0; slept {rec.sleeps[1]}"


def test_retry_after_is_honoured_as_a_floor():
    flaky, ask = flaky_court([RateLimitError("slow down", retry_after=3.0), None])
    rec = Recorder()
    room.with_retries(ask, max_attempts=3, base=0.5, cap=8.0, sleep=rec, rng=random.Random(0))
    assert len(rec.sleeps) == 1
    assert rec.sleeps[0] >= 3.0, (
        f"The server said retry_after=3.0 but you slept {rec.sleeps[0]}. When the server names a delay, wait at least that long."
    )


def test_a_bad_request_is_never_retried():
    flaky, ask = flaky_court([InvalidRequestError("malformed petition"), None])
    rec = Recorder()
    with pytest.raises(InvalidRequestError):
        room.with_retries(ask, max_attempts=5, sleep=rec, rng=random.Random(0))
    assert flaky.attempts == 1, f"A 4xx does not improve with time. You called the herald {flaky.attempts} times."
    assert rec.sleeps == [], f"No retry, no sleep. You slept {rec.sleeps}."


def test_exhaustion_raises_with_the_last_error_chained():
    flaky, ask = flaky_court([ServerError(f"boom {i}") for i in range(10)])
    rec = Recorder()
    with pytest.raises(room.RetriesExhausted) as info:
        room.with_retries(ask, max_attempts=3, base=0.5, cap=8.0, sleep=rec, rng=random.Random(0))
    assert flaky.attempts == 3, f"max_attempts=3 means three calls, you made {flaky.attempts}."
    assert len(rec.sleeps) == 2, (
        f"Three attempts have two gaps. You slept {len(rec.sleeps)} times. Never sleep after the final failure: nobody is waiting for it."
    )
    cause = info.value.__cause__
    assert isinstance(cause, ServerError), f"Chain the last error with 'raise ... from err' so __cause__ is set; got {cause!r}"
    assert "boom 2" in str(cause), f"__cause__ should be the LAST error (boom 2), got {cause}"


def test_delays_never_exceed_the_cap():
    flaky, ask = flaky_court([ServerError("down")] * 8 + [None])
    rec = Recorder()
    room.with_retries(ask, max_attempts=10, base=1.0, cap=2.0, sleep=rec, rng=random.Random(3))
    assert len(rec.sleeps) == 8
    assert all(0.0 <= s <= 2.0 for s in rec.sleeps), f"cap=2.0 but you slept {rec.sleeps}. Exponential growth must stop at the cap."


def test_on_retry_is_told_what_happened():
    flaky, ask = flaky_court([ServerError("one"), RateLimitError("two", retry_after=1.5), None])
    events = []
    rec = Recorder()
    room.with_retries(
        ask, max_attempts=5, sleep=rec, rng=random.Random(0),
        on_retry=lambda attempt, exc, delay: events.append((attempt, type(exc).__name__, delay)),
    )
    assert [e[0] for e in events] == [1, 2], f"on_retry gets the attempt number (starting at 1) for each retry: {events}"
    assert [e[1] for e in events] == ["ServerError", "RateLimitError"], f"on_retry gets the error that caused the retry: {events}"
    assert [e[2] for e in events] == rec.sleeps, f"on_retry's delay must be the delay you then sleep: events {events}, sleeps {rec.sleeps}"


def test_retry_on_can_be_narrowed():
    calls = {"n": 0}

    def sometimes():
        calls["n"] += 1
        if calls["n"] < 3:
            raise KeyError("missing scroll")
        return "found"

    rec = Recorder()
    assert room.with_retries(sometimes, max_attempts=5, retry_on=(KeyError,), sleep=rec, rng=random.Random(0)) == "found"
    assert calls["n"] == 3
    with pytest.raises(ServerError):
        room.with_retries(lambda: (_ for _ in ()).throw(ServerError("x")), max_attempts=5, retry_on=(KeyError,), sleep=rec)


# ---------------------------------------------------------------- idempotency
def test_a_duplicate_petition_is_not_paid_twice():
    paid = []

    def pay(order_id):
        paid.append(order_id)
        return f"paid {order_id}"

    idem = room.Idempotent(pay)
    first = idem("order-1")
    again = idem("order-1")
    idem("order-2")
    assert paid == ["order-1", "order-2"], f"The same key must not run the side effect twice. Payments made: {paid}"
    assert first == again == "paid order-1", "A duplicate returns the remembered result."
    assert idem.executions == 2


def test_a_failed_attempt_is_not_remembered_as_done():
    calls = {"n": 0}

    def pay(order_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerError("the treasury is asleep")
        return f"paid {order_id}"

    idem = room.Idempotent(pay)
    rec = Recorder()
    result = room.with_retries(lambda: idem("order-9"), max_attempts=3, sleep=rec, rng=random.Random(0))
    assert result == "paid order-9"
    assert calls["n"] == 2, f"The first attempt failed, so the retry must run pay() again: pay ran {calls['n']} time(s)."
    idem("order-9")
    assert calls["n"] == 2, "Once the request has succeeded, a duplicate must not run it a third time."


def test_a_custom_key_function_decides_what_counts_as_the_same_request():
    runs = []
    idem = room.Idempotent(lambda **kw: runs.append(kw) or "ok", key=lambda **kw: kw["order_id"])
    idem(order_id="x", note="first")
    idem(order_id="x", note="second")
    idem(order_id="y", note="third")
    assert len(runs) == 2, f"Same order_id is the same request even if the note differs: ran {len(runs)} times."


# ------------------------------------------------------------ circuit breaker
def test_a_closed_breaker_lets_calls_through():
    cb = room.CircuitBreaker(failure_threshold=3, recovery_time=30.0, clock=FakeClock())
    assert cb.state == "closed"
    assert cb.allow() is True
    assert cb.call(lambda: "hello") == "hello"
    assert cb.state == "closed"


def test_the_breaker_opens_after_the_threshold_and_rejects_fast():
    clock = FakeClock()
    cb = room.CircuitBreaker(failure_threshold=3, recovery_time=30.0, clock=clock)

    def failing():
        raise ServerError("the herald is down")

    for i in range(3):
        with pytest.raises(ServerError):
            cb.call(failing)
        if i < 2:
            assert cb.state == "closed", f"After {i + 1} failure(s) of a threshold of 3 the breaker must still be closed."
    assert cb.state == "open", f"Three consecutive failures should open the breaker; state is {cb.state!r}"
    assert cb.allow() is False

    probes = {"n": 0}

    def probe():
        probes["n"] += 1
        return "ok"

    with pytest.raises(room.CircuitOpen):
        cb.call(probe)
    assert probes["n"] == 0, "An open breaker rejects WITHOUT calling the function. That is the whole point: fail fast."


def test_the_breaker_half_opens_after_the_recovery_time_and_closes_on_success():
    clock = FakeClock()
    cb = room.CircuitBreaker(failure_threshold=2, recovery_time=30.0, clock=clock)
    for _ in range(2):
        with pytest.raises(ServerError):
            cb.call(lambda: (_ for _ in ()).throw(ServerError("down")))
    assert cb.state == "open"
    clock.advance(29.9)
    assert cb.state == "open", "29.9s into a 30s recovery time the breaker is still open."
    clock.advance(0.1)
    assert cb.state == "half_open", f"Once recovery_time has passed the breaker is half_open (one probe allowed); state is {cb.state!r}"
    assert cb.allow() is True
    assert cb.call(lambda: "back") == "back"
    assert cb.state == "closed", f"A successful probe closes the breaker; state is {cb.state!r}"
    assert cb.failures == 0, "Closing resets the consecutive failure count."


def test_a_failed_probe_reopens_the_breaker_for_another_rest():
    clock = FakeClock()
    cb = room.CircuitBreaker(failure_threshold=2, recovery_time=30.0, clock=clock)

    def failing():
        raise ServerError("still down")

    for _ in range(2):
        with pytest.raises(ServerError):
            cb.call(failing)
    clock.advance(30.0)
    assert cb.state == "half_open"
    with pytest.raises(ServerError):
        cb.call(failing)
    assert cb.state == "open", f"A failed probe in half_open reopens the breaker; state is {cb.state!r}"
    clock.advance(15.0)
    assert cb.state == "open", "The rest restarts from the failed probe, so 15s later it is still open."
    clock.advance(15.0)
    assert cb.state == "half_open"


def test_a_success_resets_the_failure_count():
    cb = room.CircuitBreaker(failure_threshold=3, recovery_time=30.0, clock=FakeClock())

    def failing():
        raise ServerError("hiccup")

    for _ in range(2):
        with pytest.raises(ServerError):
            cb.call(failing)
    cb.call(lambda: "fine")
    for _ in range(2):
        with pytest.raises(ServerError):
            cb.call(failing)
    assert cb.state == "closed", (
        "Two failures, a success, two failures: never three in a row, so the breaker stays closed. "
        "The threshold counts CONSECUTIVE failures."
    )
