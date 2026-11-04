"""Retry con exponential backoff + Circuit Breaker, sin dependencias externas.

Uso: envolver SOLO llamadas a proveedores (LLM, embeddings, vectorial).
Los errores del dominio (PDF corrupto, pregunta sin contexto) NO se
reintentan: no son fallos transitorios de red.
"""
from __future__ import annotations

import logging
import random
import time
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    pass


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.reset_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker abierto: proveedor en cooldown")
        try:
            result = fn()
        except Exception:
            self._failures += 1
            log.warning("circuit_breaker fallo %s/%s", self._failures, self.failure_threshold)
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            raise
        self._failures = 0
        self._state = CircuitState.CLOSED
        return result


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Reintenta errores transitorios. El ultimo error se propaga tal cual."""
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - se vuelve a lanzar abajo
            last = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
            log.warning("retry intento %s/%s tras %.2fs: %s", attempt, max_attempts, delay, exc)
            sleep(delay)
    assert last is not None
    raise last
