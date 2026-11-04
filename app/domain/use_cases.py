"""Casos de uso: el core. No importan FastAPI, Streamlit ni proveedores."""
from __future__ import annotations

import logging
import os
import uuid

from app.adapters.chunking import chunk_document
from app.domain.models import (
    INSUFFICIENT_ANSWER, Answer, Job, JobStatus, Source,
)
from app.domain.ports import (
    EmbeddingPort, JobQueuePort, JobStorePort, LLMPort, PDFParserPort,
    VectorStorePort,
)
from app.domain.retrieval import hybrid_retrieve

log = logging.getLogger(__name__)

MIN_RELEVANCE_SCORE = 0.15

SYSTEM_PROMPT = """Eres un asistente tecnico que responde SOLO con el contexto recuperado
de los documentos. Reglas obligatorias:
- Si el contexto no contiene la respuesta, responde EXACTAMENTE:
  "No tengo información suficiente en los documentos para responder eso."
- Nunca inventes datos, cifras, paginas ni fuentes.
- Cita solo documentos y paginas que aparecen en el contexto.
- Responde en espanol, de forma concisa."""


class IngestUseCase:
    def __init__(self, parser: PDFParserPort, embedding: EmbeddingPort,
                 store: VectorStorePort, queue: JobQueuePort, jobs: JobStorePort):
        self.parser = parser
        self.embedding = embedding
        self.store = store
        self.queue = queue
        self.jobs = jobs

    def start(self, pdf_path: str, document_name: str) -> Job:
        job = Job(id=str(uuid.uuid4()), document_name=document_name)
        self.jobs.save(job)

        def _run() -> None:
            self.run(job.id, pdf_path, document_name)

        self.queue.enqueue(job, _run)
        log.info("ingesta iniciada job_id=%s document=%s", job.id, document_name)
        return job

    def run(self, job_id: str, pdf_path: str, document_name: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        try:
            job.status = JobStatus.PROCESSING
            job.progress = 0.1
            job.message = "Extrayendo texto, tablas e imagenes"
            self.jobs.save(job)
            document_id = str(uuid.uuid4())
            parsed = self.parser.parse(pdf_path, document_id, document_name)
            job.progress = 0.5
            job.message = "Aplicando chunking por layout"
            self.jobs.save(job)
            chunks = chunk_document(parsed)
            job.progress = 0.7
            job.message = "Generando embeddings e indexando"
            self.jobs.save(job)
            vectors = self.embedding.embed([c.text for c in chunks]) if chunks else []
            self.store.upsert(chunks, vectors)
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.message = "Completado"
            job.document_id = document_id
            job.chunks_count = len(chunks)
            job.images_count = len(parsed.images)
            log.info("ingesta completada job_id=%s chunks=%s imagenes=%s",
                     job_id, len(chunks), len(parsed.images))
        except Exception as exc:  # noqa: BLE001 - el job debe reflejar el fallo
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.message = "Fallo en la ingesta"
            log.exception("ingesta fallida job_id=%s error=%s", job_id, exc)
        finally:
            self.jobs.save(job)


class AskUseCase:
    def __init__(self, llm: LLMPort, embedding: EmbeddingPort, store: VectorStorePort,
                 images_by_id: dict[str, str] | None = None):
        self.llm = llm
        self.embedding = embedding
        self.store = store
        self.images_by_id = images_by_id or {}

    def ask(self, question: str, top_k: int = 5) -> Answer:
        question_id = str(uuid.uuid4())[:8]
        log.info("pregunta recibida question_id=%s question=%s", question_id, question)
        retrieved = hybrid_retrieve(question, self.embedding, self.store, top_k=top_k)
        if not retrieved or retrieved[0].score < MIN_RELEVANCE_SCORE:
            log.info("sin contexto suficiente question_id=%s top_score=%s",
                     question_id, retrieved[0].score if retrieved else 0)
            return Answer(text=INSUFFICIENT_ANSWER, sources=[])
        context_lines = []
        for r in retrieved:
            context_lines.append(
                f"[Fuente: {r.chunk.document_name}, Pagina {r.chunk.page}] {r.chunk.text}")
        context = "\n".join(context_lines)
        user_prompt = (
            f"CONTEXTO:\n{context}\n"
            f"PREGUNTA: {question}\n"
            "Responde solo con el contexto anterior."
        )
        text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        sources = [
            Source(document_name=r.chunk.document_name, page=r.chunk.page,
                   kind=r.chunk.kind, chunk_id=r.chunk.id, image_id=r.chunk.image_id)
            for r in retrieved
        ]
        # Imagen mas cercana: la del chunk mejor puntuado que tenga imagen,
        # o, en su defecto, la primera imagen de la misma pagina.
        image_id = next((s.image_id for s in sources if s.image_id), None)
        image_page = next((s.page for s in sources if s.image_id), None)
        return Answer(text=text, sources=sources, image_id=image_id, image_page=image_page)
