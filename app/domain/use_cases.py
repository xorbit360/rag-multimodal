"""Casos de uso: el core. No importan FastAPI, Streamlit ni proveedores."""
from __future__ import annotations

import logging
import os
import threading
import uuid

from app.adapters.chunking import chunk_document
from app.domain.models import (
    INSUFFICIENT_ANSWER, Answer, AudioAnswer, ImageAnswer, Job, JobStatus, Source,
)
from app.domain.ports import (
    EmbeddingPort, JobQueuePort, JobStorePort, LLMPort, PDFParserPort,
    TranscriptionPort, VectorStorePort, VisionPort,
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
        # Indice hash SHA-256 del PDF -> job canonico. Cuando el JobStore
        # lo soporta (RedisJobStore / InMemoryJobStore), el indice vive
        # en el propio store y sobrevive al proceso; si no, se usa este
        # diccionario local solo como respaldo. Nunca se reconstruye
        # perdiendo deduplicacion: el vector store persiste aparte.
        self._hash_index: dict[str, str] = {}
        self._hash_lock = threading.Lock()

    def _get_by_hash(self, file_hash: str) -> Job | None:
        get_by_hash = getattr(self.jobs, "get_by_hash", None)
        if callable(get_by_hash):
            return get_by_hash(file_hash)
        with self._hash_lock:
            canonical = self._hash_index.get(file_hash)
        return self.jobs.get(canonical) if canonical else None

    def _set_hash(self, file_hash: str, job_id: str) -> None:
        # RedisJobStore / InMemoryJobStore ya indexan en save(); el
        # diccionario local solo se mantiene para stores antiguos.
        if not hasattr(self.jobs, "get_by_hash"):
            with self._hash_lock:
                self._hash_index[file_hash] = job_id

    def _clear_hash(self, file_hash: str, job_id: str) -> None:
        clear_hash = getattr(self.jobs, "clear_hash", None)
        if callable(clear_hash):
            clear_hash(file_hash, job_id)
            return
        with self._hash_lock:
            if self._hash_index.get(file_hash) == job_id:
                del self._hash_index[file_hash]

    def start(self, pdf_path: str, document_name: str,
              file_hash: str | None = None) -> Job:
        if file_hash:
            existing = self._get_by_hash(file_hash)
            if existing is not None and existing.status != JobStatus.FAILED:
                    if existing.status == JobStatus.COMPLETED:
                        # Guardado inteligente: no se duplica nada. Se devuelve
                        # un job nuevo ya completado que apunta al documento
                        # original, con el aviso en el mensaje.
                        reuse = Job(
                            id=str(uuid.uuid4()), document_name=document_name,
                            status=JobStatus.COMPLETED, progress=1.0,
                            message="Documento ya indexado: se reutilizo la "
                                    "ingesta anterior, sin duplicar chunks.",
                            document_id=existing.document_id,
                            chunks_count=existing.chunks_count,
                            images_count=existing.images_count,
                            file_hash=file_hash, duplicate=True)
                        self.jobs.save(reuse)
                        log.info("ingesta duplicada evitada hash=%s job_id=%s",
                                 file_hash[:12], reuse.id)
                        return reuse
                    # En curso: se devuelve el mismo job para que el cliente
                    # lo siga por polling; no se lanza un segundo trabajo.
                    log.info("ingesta en curso reutilizada hash=%s job_id=%s",
                             file_hash[:12], existing.id)
                    return existing
        job = Job(id=str(uuid.uuid4()), document_name=document_name,
                  file_hash=file_hash, pdf_path=pdf_path)
        self.jobs.save(job)
        if file_hash:
            self._set_hash(file_hash, job.id)

        def _run() -> None:
            self.run(job.id, pdf_path, document_name)

        self.queue.enqueue(job, _run)
        log.info("ingesta iniciada job_id=%s document=%s", job.id, document_name)
        return job

    def run_sync(self, pdf_path: str, document_name: str,
                 file_hash: str | None = None) -> Job:
        """Ingesta sincrona, sin pasar por la cola: crea el job, registra
        el hash (para que la subida posterior del mismo PDF por UI lo
        detecte como duplicado) y ejecuta el mismo pipeline de run().
        La usa el seed del manual de ejemplo en el arranque; la subida
        por UI sigue usando start() (asincrona)."""
        job = Job(id=str(uuid.uuid4()), document_name=document_name,
                  file_hash=file_hash, pdf_path=pdf_path)
        self.jobs.save(job)
        if file_hash:
            self._set_hash(file_hash, job.id)
        self.run(job.id, pdf_path, document_name)
        saved = self.jobs.get(job.id)
        return saved if saved is not None else job

    def run(self, job_id: str, pdf_path: str | None = None,
            document_name: str | None = None) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        # El worker Redis llega solo con el job_id: la ruta del PDF
        # guardada en el propio Job es la fuente de verdad.
        pdf_path = pdf_path or job.pdf_path
        document_name = document_name or job.document_name
        if not pdf_path:
            job.status = JobStatus.FAILED
            job.error = "Job sin pdf_path: no se puede re-ejecutar"
            job.message = "Fallo en la ingesta"
            self.jobs.save(job)
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
            # Un fallo libera el hash: un reintento posterior debe poder
            # volver a ingerir el mismo archivo.
            if job.file_hash:
                self._clear_hash(job.file_hash, job.id)
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


class AudioAskUseCase:
    """Pregunta por audio: transcribe y luego ejecuta el flujo normal de
    ask con el texto transcrito. La transcripcion se devuelve tal cual:
    nunca se inventa una cuando falta el token (eso lo impide el endpoint
    exigiendo el override antes de llamar aqui)."""

    def __init__(self, transcriber: TranscriptionPort, ask: AskUseCase):
        self.transcriber = transcriber
        self.ask = ask

    def ask_audio(self, audio: bytes, filename: str, model: str,
                  question_text: str = "", top_k: int = 5) -> AudioAnswer:
        transcription = self.transcriber.transcribe(audio, filename, model)
        parts = [p for p in (question_text.strip(), transcription) if p]
        question = "\n".join(parts)
        answer = self.ask.ask(question, top_k=top_k)
        return AudioAnswer(transcription=transcription, answer=answer)


class ImageAskUseCase:
    """Pregunta con imagen: el modelo de vision describe la imagen y con
    esa descripcion se ejecuta la recuperacion hibrida normal contra el
    RAG. La imagen enviada NO se indexa: es insumo de pregunta, y la
    busqueda responde a que hace referencia dentro de lo ya indexado."""

    def __init__(self, vision: VisionPort, ask: AskUseCase):
        self.vision = vision
        self.ask = ask

    def ask_image(self, image: bytes, mime: str, question_text: str = "",
                  top_k: int = 5) -> ImageAnswer:
        description = self.vision.describe(image, mime)
        if question_text.strip():
            query = (f"{question_text.strip()}\n"
                     f"Descripcion de la imagen adjunta: {description}")
        else:
            query = ("A que hace referencia esta imagen en los documentos? "
                     f"Descripcion de la imagen: {description}")
        answer = self.ask.ask(query, top_k=top_k)
        return ImageAnswer(description=description, answer=answer)
