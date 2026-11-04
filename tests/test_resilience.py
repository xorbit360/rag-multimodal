import pytest

from app.infrastructure.resilience import (
    CircuitBreaker, CircuitOpenError, retry_with_backoff,
)


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("red caida")
        return "ok"

    assert retry_with_backoff(flaky, max_attempts=3, base_delay=0.01, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts():
    with pytest.raises(ConnectionError):
        retry_with_backoff(lambda: (_ for _ in ()).throw(ConnectionError("x")),
                           max_attempts=2, base_delay=0.01, sleep=lambda _: None)


def test_circuit_breaker_opens_after_threshold_and_blocks():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=60)

    def fail():
        raise RuntimeError("llm caido")

    with pytest.raises(RuntimeError):
        breaker.call(fail)
    with pytest.raises(RuntimeError):
        breaker.call(fail)
    with pytest.raises(CircuitOpenError):
        breaker.call(fail)


def test_circuit_breaker_resets_after_success():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60)
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state.value == "closed"
