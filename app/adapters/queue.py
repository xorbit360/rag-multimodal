"""Colas de trabajos: en memoria (hilo) + Redis (escala).

InMemoryJobQueue ejecuta en un hilo daemon y actualiza el JobStore: la
API responde al instante y la ingesta pesada no bloquea el hilo principal.
RedisJobQueue solo encola el job_id en Redis; un worker aparte (app/worker.py)
lo consume. Asi la misma interfaz pasa de 1 manual a 100 manuales
anadiendo workers, sin cambiar los casos de uso.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.domain.models import Job

log = logging.getLogger(__name__)


class InMemoryJobQueue:
    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def enqueue(self, job: Job, fn: Callable[[], None]) -> None:
        self._pool.submit(fn)


class RedisJobQueue:
    def __init__(self, redis_url: str, queue_name: str = "rag_ingest_jobs"):
        import redis

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._queue_name = queue_name

    def enqueue(self, job: Job, fn: Callable[[], None]) -> None:
        # En modo Redis, fn NO se ejecuta en la API: el worker lo recupera
        # por job_id desde el JobStore compartido (base de datos en prod).
        # Aqui solo se encola el identificador; el worker re-ejecuta la
        # ingesta desde el archivo subido, guardado en storage/uploads.
        self._redis.lpush(self._queue_name, job.id)
        log.info("job encolado en redis job_id=%s", job.id)


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())
