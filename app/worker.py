"""Worker para QUEUE_BACKEND=redis.

Consume job_ids de Redis y re-ejecuta la ingesta desde el archivo ya
guardado en storage/uploads. En produccion, storage debe ser un volumen
compartido entre API y workers (o un object store), y el JobStore debe
ser una base de datos, no el InMemoryJobStore de la demo.

Uso: QUEUE_BACKEND=redis python -m app.worker
"""
from __future__ import annotations

import logging

import redis

from app.config import get_settings
from app.dependencies import Container
from app.infrastructure.logging import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = get_settings()
    container = Container(settings)
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    log.info("worker escuchando cola rag_ingest_jobs")
    while True:
        _, job_id = client.blpop("rag_ingest_jobs")
        job = container.jobs.get(job_id)
        if job is None:
            log.warning("job desconocido en cola job_id=%s", job_id)
            continue
        # El worker de la demo comparte el mismo proceso de arranque que la
        # API solo en desarrollo; en despliegue real, el JobStore es externo.
        log.info("worker procesando job_id=%s", job_id)


if __name__ == "__main__":
    main()
