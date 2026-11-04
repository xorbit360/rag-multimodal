"""API FastAPI: transporte. Toda la logica vive en app/domain/."""
from __future__ import annotations

import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.llm import PROVIDERS, build_override_llm, detect_provider
from app.api.schemas import (
    AskRequest, AskResponse, DetectRequest, DetectResponse, JobOut, SourceOut,
)
from app.config import get_settings
from app.dependencies import Container
from app.domain.use_cases import AskUseCase
from app.infrastructure.logging import setup_logging

setup_logging()
settings = get_settings()
container = Container(settings)

app = FastAPI(title="RAG Multimodal", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": settings.use_demo_llm,
            "vector_store": settings.vector_store, "queue": settings.queue_backend}


@app.post("/ingest", response_model=JobOut)
def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(container.uploads_dir, f"{uuid.uuid4()}_{safe_name}")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    job = container.ingest.start(dest, safe_name)
    return _job_out(job)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    job = container.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return _job_out(job)


@app.get("/documents")
def list_documents():
    docs = {}
    for job in container.jobs.all():
        if job.document_id:
            docs[job.document_id] = {"document_id": job.document_id,
                                     "document_name": job.document_name,
                                     "chunks": job.chunks_count,
                                     "images": job.images_count,
                                     "status": job.status.value}
    return list(docs.values())


@app.post("/llm/detect", response_model=DetectResponse)
def detect_llm(req: DetectRequest):
    # Nunca se registra ni se devuelve la clave: solo proveedor y modelos.
    provider, models = detect_provider(req.api_key, req.base_url)
    if provider is None:
        return DetectResponse(
            detected=False,
            message="No se reconoció el token en OpenAI ni en OpenRouter. "
                    "Puedes configurarlo manual con proveedor y base_url.")
    base = req.base_url if provider == "custom" else PROVIDERS[provider]
    return DetectResponse(detected=True, provider=provider, base_url=base,
                          models=models[:300],
                          message=f"Proveedor detectado: {provider} "
                                  f"({len(models)} modelos)")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.llm is not None:
        try:
            llm = build_override_llm(req.llm.provider, req.llm.model,
                                     req.llm.api_key, req.llm.base_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        use_case = AskUseCase(llm, container.ask.embedding, container.ask.store)
    else:
        use_case = container.ask
    answer = use_case.ask(req.question, top_k=req.top_k)
    image_url = f"/images/{answer.image_id}" if answer.image_id else None
    return AskResponse(
        answer=answer.text,
        sources=[SourceOut(document_name=s.document_name, page=s.page,
                           kind=s.kind, image_id=s.image_id) for s in answer.sources],
        image_id=answer.image_id, image_page=answer.image_page, image_url=image_url)


@app.get("/images/{image_id}")
def get_image(image_id: str):
    path = os.path.join(container.images_dir, f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(path, media_type="image/png")


def _job_out(job):
    return JobOut(job_id=job.id, document_name=job.document_name,
                  status=job.status.value, progress=job.progress,
                  message=job.message, chunks_count=job.chunks_count,
                  images_count=job.images_count, error=job.error)
