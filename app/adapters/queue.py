"""Colas y almacenes de trabajos: en memoria (hilo) + Redis (escala).

InMemoryJobQueue ejecuta en un hilo daemon y actualiza el JobStore: la
API responde al instante y la ingesta pesada no bloquea el hilo principal.
RedisJobQueue solo encola el job_id en Redis; un worker aparte (app/worker.py)
lo consume y re-ejecuta la ingesta desde el archivo guardado.
RedisJobStore guarda cada Job como JSON en Redis (hash rag_jobs) y el
indice file_hash -> job_id en rag_hash_index: asi API y workers comparten
el mismo estado, los jobs sobreviven a un reinicio de la API (Redis con
appendonly en docker-compose) y varios workers pueden consumir en
paralelo sin duplicar trabajo. La misma interfaz pasa de 1 manual a
100 manuales anadiendo workers, sin cambiar los casos de uso.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.domain.models import Job, JobStatus

log = logging.getLogger(__name__)


def _job_to_dict(job: Job) -> dict:
    return {
        "id": job.id, "document_name": job.document_name,
        "status": job.status.value, "progress": job.progress,
        "message": job.message, "document_id": job.document_id,
        "chunks_count": job.chunks_count, "images_count": job.images_count,
        "error": job.error, "file_hash": job.file_hash,
        "duplicate": job.duplicate, "pdf_path": job.pdf_path,
    }


def _job_from_dict(data: dict) -> Job:
    return Job(
        id=data["id"], document_name=data["document_name"],
        status=JobStatus(data.get("status", "pending")),
        progress=float(data.get("progress", 0.0)),
        message=data.get("message", ""), document_id=data.get("document_id"),
        chunks_count=int(data.get("chunks_count", 0)),
        images_count=int(data.get("images_count", 0)),
        error=data.get("error"), file_hash=data.get("file_hash"),
        duplicate=bool(data.get("duplicate", False)),
        pdf_path=data.get("pdf_path"),
    )


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
        self._hash_index: dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            if job.file_hash and not job.duplicate:
                self._hash_index[job.file_hash] = job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_by_hash(self, file_hash: str) -> Job | None:
        with self._lock:
            job_id = self._hash_index.get(file_hash)
            return self._jobs.get(job_id) if job_id else None

    def clear_hash(self, file_hash: str, job_id: str) -> None:
        with self._lock:
            if self._hash_index.get(file_hash) == job_id:
                del self._hash_index[file_hash]

    def delete_by_document(self, document_id: str) -> list[Job]:
        with self._lock:
            removed = [j for j in self._jobs.values()
                       if j.document_id == document_id]
            for job in removed:
                del self._jobs[job.id]
                if job.file_hash and self._hash_index.get(job.file_hash) == job.id:
                    del self._hash_index[job.file_hash]
            return removed


class RedisJobStore:
    """JobStore compartido y persistente en Redis.

    Cada save() escribe el Job completo (incluida pdf_path) y mantiene
    el indice de hash. Un worker en otro proceso / contenedor lee el
    mismo estado: por eso la cola Redis ya no depende de la memoria
    de la API. Redis corre con appendonly en docker-compose, asi que
    los jobs sobreviven a un reinicio del contenedor de Redis.
    """

    def __init__(self, redis_url: str):
        import redis

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._jobs_key = "rag_jobs"
        self._hash_key = "rag_hash_index"

    def save(self, job: Job) -> None:
        self._redis.hset(self._jobs_key, job.id, json.dumps(_job_to_dict(job)))
        if job.file_hash and not job.duplicate:
            self._redis.hset(self._hash_key, job.file_hash, job.id)

    def get(self, job_id: str) -> Job | None:
        raw = self._redis.hget(self._jobs_key, job_id)
        return _job_from_dict(json.loads(raw)) if raw else None

    def all(self) -> list[Job]:
        return [_job_from_dict(json.loads(v))
                for v in self._redis.hvals(self._jobs_key)]

    def get_by_hash(self, file_hash: str) -> Job | None:
        job_id = self._redis.hget(self._hash_key, file_hash)
        return self.get(job_id) if job_id else None

    def clear_hash(self, file_hash: str, job_id: str) -> None:
        # Borrar solo si sigue apuntando a este job (evita borrar un
        # reintento posterior que reutilizo el mismo hash).
        script = (
            "if redis.call('hget', KEYS[1], ARGV[1]) == ARGV[2] then "
            "return redis.call('hdel', KEYS[1], ARGV[1]) else return 0 end"
        )
        self._redis.eval(script, 1, self._hash_key, file_hash, job_id)

    def delete_by_document(self, document_id: str) -> list[Job]:
        removed = [j for j in self.all() if j.document_id == document_id]
        if not removed:
            return []
        removed_ids = {j.id for j in removed}
        self._redis.hdel(self._jobs_key, *removed_ids)
        hash_index = self._redis.hgetall(self._hash_key)
        stale = [h for h, jid in hash_index.items() if jid in removed_ids]
        if stale:
            self._redis.hdel(self._hash_key, *stale)
        return removed
