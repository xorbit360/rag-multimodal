"""Worker para QUEUE_BACKEND=redis.

Consume job_ids de Redis y re-ejecuta la ingesta de verdad desde el
archivo ya guardado en storage/uploads, usando el JobStore compartido
(RedisJobStore). Por eso API y workers pueden vivir en contenedores
distintos: el estado, la ruta del PDF y el indice de hash son los
mismos para ambos, y un reinicio de la API no pierde los jobs en
curso (Redis con appendonly en docker-compose).

Uso: QUEUE_BACKEND=redis python -m app.worker

Requisitos de despliegue: storage/ debe ser un volumen compartido
entre API y workers (en docker-compose es rag_storage), y Qdrant y
Redis deben ser alcanzables con la misma configuracion que la API.
"""
from __future__ import annotations

import logging
import os
import sys
import time

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
        try:
            # BLPOP con timeout 5 s, no indefinido: con timeout 0 el
            # cliente redis-py cerraba el socket por timeout de lectura
            # en el despliegue (verificado en vivo: reintentos cada
            # 5-7 s sin procesar). Con 5 s el loop vuelve a bloquear
            # y un job nuevo se procesa en menos de 5 s.
            result = client.blpop("rag_ingest_jobs", timeout=5)
        except Exception:
            log.exception("worker: error leyendo de Redis, reintentando")
            time.sleep(2)
            continue
        if result is None:
            continue
        _, job_id = result
        job = container.jobs.get(job_id)
        if job is None:
            log.warning("job desconocido en cola job_id=%s", job_id)
            continue
        if not job.pdf_path or not os.path.exists(job.pdf_path):
            log.error("worker: job sin archivo job_id=%s pdf_path=%s",
                      job_id, job.pdf_path)
            # run() marcara el job como FAILED con un mensaje claro.
            container.ingest.run(job_id)
            continue
        log.info("worker procesando job_id=%s document=%s",
                 job_id, job.document_name)
        try:
            container.ingest.run(job_id, job.pdf_path, job.document_name)
        except Exception:
            log.exception("worker: fallo inesperado job_id=%s", job_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
