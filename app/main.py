"""API FastAPI: transporte. Toda la logica vive en app/domain/."""
from __future__ import annotations

import hashlib
import json
import os
import uuid

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.llm import PROVIDERS, build_override_llm, detect_provider_info
from app.adapters.multimodal import (
    OpenAICompatibleTranscription, OpenAICompatibleVision,
)
from app.api.schemas import (
    AskRequest, AskResponse, AudioAskResponse, DetectRequest, DetectResponse,
    ImageAskResponse, JobOut, ModelInfoOut, SourceOut,
)
from app.config import get_settings
from app.dependencies import Container
from app.domain.model_selection import (
    ModelCandidate, select_model, select_transcription_model,
)
from app.domain.ports import ProviderResponseError, VisionNotSupportedError
from app.domain.use_cases import AskUseCase, AudioAskUseCase, ImageAskUseCase
from app.infrastructure.logging import setup_logging
from app.infrastructure.resilience import CircuitOpenError

setup_logging()
settings = get_settings()
container = Container(settings)

app = FastAPI(title="RAG Multimodal", version="0.3.0")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mp4"}


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
    # Guardado inteligente: se calcula el SHA-256 mientras se escribe, sin
    # cargar el archivo completo en memoria. El mismo hash no duplica chunks.
    digest = hashlib.sha256()
    with open(dest, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    job = container.ingest.start(dest, safe_name, file_hash=digest.hexdigest())
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
    # Nunca se registra ni se devuelve la clave: solo proveedor, modelos,
    # precios y modalidad (cuando el proveedor los da).
    provider, info = detect_provider_info(req.api_key, req.base_url)
    if provider is None:
        return DetectResponse(
            detected=False,
            message="No se reconoció el token en OpenAI ni en OpenRouter. "
                    "Puedes configurarlo manual con proveedor y base_url.")
    base = req.base_url if provider == "custom" else PROVIDERS[provider]
    models_info = [ModelInfoOut(**m) for m in info]
    return DetectResponse(detected=True, provider=provider, base_url=base,
                          models=[m.id for m in models_info],
                          models_info=models_info,
                          message=f"Proveedor detectado: {provider} "
                                  f"({len(models_info)} modelos)")


def _candidates(models_info_json: str | None) -> list[ModelCandidate]:
    """Lista de modelos que el frontend paso (de /llm/detect). Si viene
    vacia o invalida, se devuelve lista vacia: la seleccion usa entonces
    la tabla de costo conocido."""
    if not models_info_json:
        return []
    try:
        raw = json.loads(models_info_json)
    except (ValueError, TypeError):
        return []
    candidates = []
    for m in raw if isinstance(raw, list) else []:
        if isinstance(m, dict) and m.get("id"):
            candidates.append(ModelCandidate(id=m["id"],
                                             prompt_price=m.get("prompt_price"),
                                             modality=m.get("modality")))
    return candidates


def _resolve_llm(provider: str, model: str | None, api_key: str,
                 base_url: str | None, auto_model: bool,
                 models_info_json: str | None, need_vision: bool):
    """Resuelve el adaptador LLM: manual (se respeta el modelo elegido) o
    automatico (el mas economico segun lo que requiera la pregunta).
    Devuelve (llm, modelo_usado, es_automatico)."""
    if auto_model:
        chosen = select_model(_candidates(models_info_json),
                              need_vision=need_vision)
        if chosen is None:
            if need_vision:
                raise HTTPException(
                    status_code=400,
                    detail="Ningun modelo disponible soporta vision. Elige un "
                           "modelo manual con vision o pregunta solo con texto.")
            raise HTTPException(
                status_code=400,
                detail="No se pudo elegir un modelo automatico: el proveedor "
                       "no devolvio modelos. Elige un modelo manual.")
        try:
            return (build_override_llm(provider, chosen, api_key, base_url),
                    chosen, True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if not model:
        raise HTTPException(
            status_code=400,
            detail="Elige un modelo o el modo automatico.")
    try:
        return build_override_llm(provider, model, api_key, base_url), model, False
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _answer_out(answer):
    image_url = f"/images/{answer.image_id}" if answer.image_id else None
    sources = [SourceOut(document_name=s.document_name, page=s.page,
                          kind=s.kind, image_id=s.image_id)
               for s in answer.sources]
    return sources, image_url


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.llm is not None:
        llm, model_used, model_auto = _resolve_llm(
            req.llm.provider, req.llm.model, req.llm.api_key, req.llm.base_url,
            req.llm.auto_model,
            json.dumps([m.model_dump() for m in req.llm.models_info])
            if req.llm.models_info else None,
            need_vision=False)
        use_case = AskUseCase(llm, container.ask.embedding, container.ask.store)
    else:
        use_case = container.ask
        model_used, model_auto = "demo", False
    # Antes, cualquier fallo del proveedor (401, 404, timeout, conexion,
    # respuesta con forma inesperada) escapaba sin mapear y FastAPI
    # devolvia un 500 "Internal Server Error" en el chat. Ahora se
    # traduce a 502 con un mensaje limpio, igual que en audio e imagen.
    try:
        answer = use_case.ask(req.question, top_k=req.top_k)
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor del LLM devolvio un error. Revisa que el "
                   "token sea valido y que el modelo exista.")
    except ProviderResponseError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor devolvio una respuesta inesperada. Revisa "
                   "el modelo elegido y que el token sea valido.")
    except CircuitOpenError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor esta temporalmente no disponible "
                   "(cooldown). Intentalo de nuevo en unos segundos.")
    sources, image_url = _answer_out(answer)
    return AskResponse(
        answer=answer.text, sources=sources,
        image_id=answer.image_id, image_page=answer.image_page,
        image_url=image_url, model_used=model_used, model_auto=model_auto)


@app.post("/ask-audio", response_model=AudioAskResponse)
def ask_audio(audio: UploadFile = File(...),
              provider: str = Form("openai"),
              model: str | None = Form(None),
              api_key: str | None = Form(None),
              base_url: str | None = Form(None),
              auto_model: bool = Form(False),
              models_info_json: str | None = Form(None),
              transcription_model: str = Form("whisper-1"),
              question: str = Form(""),
              top_k: int = Form(5)):
    """Pregunta por audio: transcribe y luego pregunta al RAG con el texto.
    Exige token: sin el, no hay transcripcion posible y no se inventa una.
    El audio no se guarda como documento."""
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="El audio necesita un token: pegalo en la seccion LLM.")
    name = (audio.filename or "audio").lower()
    if not any(name.endswith(ext) for ext in AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Formato de audio no soportado (mp3, wav, m4a, ogg, webm).")
    llm, model_used, model_auto = _resolve_llm(
        provider, model, api_key, base_url, auto_model, models_info_json,
        need_vision=False)
    if auto_model:
        trans_model = select_transcription_model(
            _candidates(models_info_json))
    else:
        trans_model = transcription_model
    transcriber = OpenAICompatibleTranscription(api_key=api_key,
                                                base_url=_provider_url(provider, base_url))
    use_case = AudioAskUseCase(
        transcriber, AskUseCase(llm, container.ask.embedding, container.ask.store))
    try:
        result = use_case.ask_audio(audio.file.read(), audio.filename or "audio",
                                    trans_model, question_text=question,
                                    top_k=top_k)
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor no pudo transcribir el audio. Revisa que el "
                   "proveedor ofrezca transcripcion y que el token sea valido.")
    except ProviderResponseError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor devolvio una respuesta inesperada. Revisa "
                   "el modelo elegido y que el token sea valido.")
    except CircuitOpenError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor esta temporalmente no disponible "
                   "(cooldown). Intentalo de nuevo en unos segundos.")
    answer = result.answer
    sources, image_url = _answer_out(answer)
    return AudioAskResponse(
        transcription=result.transcription, answer=answer.text, sources=sources,
        image_id=answer.image_id, image_page=answer.image_page,
        image_url=image_url, model_used=model_used, model_auto=model_auto,
        transcription_model=trans_model)


@app.post("/ask-image", response_model=ImageAskResponse)
def ask_image(image: UploadFile = File(...),
              provider: str = Form("openai"),
              model: str | None = Form(None),
              api_key: str | None = Form(None),
              base_url: str | None = Form(None),
              auto_model: bool = Form(False),
              models_info_json: str | None = Form(None),
              question: str = Form(""),
              top_k: int = Form(5)):
    """Pregunta con imagen: el modelo de vision la describe y con esa
    descripcion se busca en el RAG a que hace referencia. La imagen
    enviada NO se indexa como documento. Si el modelo no soporta vision,
    error claro, sin descripcion inventada."""
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="La pregunta con imagen necesita un token: pegalo en la "
                   "seccion LLM.")
    name = (image.filename or "").lower()
    if not any(name.endswith(ext) for ext in IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Formato de imagen no soportado (png, jpg, jpeg, webp).")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}[name.rsplit(".", 1)[-1]]
    llm, model_used, model_auto = _resolve_llm(
        provider, model, api_key, base_url, auto_model, models_info_json,
        need_vision=True)
    vision = OpenAICompatibleVision(api_key=api_key,
                                    base_url=_provider_url(provider, base_url),
                                    model=model_used)
    use_case = ImageAskUseCase(
        vision, AskUseCase(llm, container.ask.embedding, container.ask.store))
    try:
        result = use_case.ask_image(image.file.read(), mime,
                                    question_text=question, top_k=top_k)
    except VisionNotSupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor de vision devolvio un error. Revisa el "
                   "modelo elegido y que el token sea valido.")
    except ProviderResponseError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor devolvio una respuesta inesperada. Revisa "
                   "el modelo elegido y que el token sea valido.")
    except CircuitOpenError:
        raise HTTPException(
            status_code=502,
            detail="El proveedor esta temporalmente no disponible "
                   "(cooldown). Intentalo de nuevo en unos segundos.")
    answer = result.answer
    sources, image_url = _answer_out(answer)
    return ImageAskResponse(
        description=result.description, answer=answer.text, sources=sources,
        image_id=answer.image_id, image_page=answer.image_page,
        image_url=image_url, model_used=model_used, model_auto=model_auto)


def _provider_url(provider: str, base_url: str | None) -> str:
    if provider == "custom":
        if not base_url:
            raise HTTPException(
                status_code=400, detail="proveedor custom requiere base_url")
        return base_url
    if provider in PROVIDERS:
        return PROVIDERS[provider]
    raise HTTPException(status_code=400, detail=f"proveedor desconocido: {provider}")


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
                  images_count=job.images_count, error=job.error,
                  duplicate=job.duplicate, document_id=job.document_id)
