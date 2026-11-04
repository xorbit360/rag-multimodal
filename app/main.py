"""API FastAPI: transporte. Toda la logica vive en app/domain/."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.llm import PROVIDERS, build_override_llm, detect_provider_info
from app.adapters.multimodal import (
    OpenAICompatibleTranscription, OpenAICompatibleVision,
)
from app.adapters.token_vault import (
    TokenVaultUnavailable, validate_browser_id,
)
from app.api.schemas import (
    AskRequest, AskResponse, AudioAskResponse, DataOverviewOut, DetectRequest,
    DetectResponse, ImageAskResponse, JobOut, ModelInfoOut, SamplePointOut,
    SourceOut, TokenSaveRequest, TokenStatus,
)
from app.config import get_settings
from app.dependencies import Container
from app.domain.model_selection import (
    ModelCandidate, select_model, select_transcription_model,
)
from app.domain.models import JobStatus
from app.domain.ports import (
    ProviderResponseError, StoreUnavailableError, VisionNotSupportedError,
)
from app.domain.use_cases import AskUseCase, AudioAskUseCase, ImageAskUseCase
from app.infrastructure.logging import setup_logging
from app.infrastructure.resilience import CircuitOpenError

setup_logging()
log = logging.getLogger(__name__)
settings = get_settings()
container = Container(settings)

EXAMPLE_MANUAL = Path(__file__).resolve().parents[1] / "examples" / "Manual_Motor.pdf"


def seed_example_manual(container, settings):
    """Ingesta del manual de ejemplo al arrancar, solo en modo demo y
    solo si el vector store esta vacio. Idempotente: si ya hay chunks,
    no duplica nada (y el hash queda registrado, asi que una subida
    posterior del mismo PDF por UI se detecta como duplicada). Corre el
    mismo pipeline de la subida por UI, de forma sincrona (sin pasar
    por la cola), para no depender del worker en el arranque. Si algo
    falla, solo registra un warning: el seed nunca tumba el arranque.
    Devuelve el job completado o None si no se sembro nada."""
    if not settings.demo_mode:
        return None
    if not EXAMPLE_MANUAL.exists():
        log.warning("seed omitido: no existe el manual de ejemplo %s", EXAMPLE_MANUAL)
        return None
    try:
        existing = container.store.all_chunks()
    except Exception as exc:  # noqa: BLE001 - el seed nunca tumba el arranque
        log.warning("seed omitido: no se pudo revisar el store: %s", exc)
        return None
    if existing:
        log.info("seed omitido: el store ya tiene %s chunks", len(existing))
        return None
    try:
        digest = hashlib.sha256(EXAMPLE_MANUAL.read_bytes()).hexdigest()
        job = container.ingest.run_sync(
            str(EXAMPLE_MANUAL), "Manual_Motor.pdf", file_hash=digest)
    except Exception as exc:  # noqa: BLE001 - el seed nunca tumba el arranque
        log.warning("seed del manual fallido: %s", exc)
        return None
    if job.status != JobStatus.COMPLETED:
        log.warning("seed del manual fallido: job %s estado %s error %s",
                    job.id, job.status, job.error)
        return None
    log.info("seed completado: Manual_Motor.pdf chunks=%s imagenes=%s",
             job.chunks_count, job.images_count)
    return job


@asynccontextmanager
async def lifespan(_app):
    seed_example_manual(container, settings)
    yield


app = FastAPI(title="RAG Multimodal", version="0.5.0", lifespan=lifespan)

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
    # El registro de jobs vive en memoria y se pierde al reiniciar el
    # contenedor, pero los chunks persisten en el vector store: sin
    # esto, tras un reinicio la lista salia vacia aunque Qdrant tuviera
    # los 18 puntos. Se completa con el contenido real del store (los
    # jobs mandan cuando existen, para no duplicar ni perder el estado).
    try:
        chunks = container.store.all_chunks()
    except StoreUnavailableError:
        chunks = []
    by_doc: dict[str, list] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)
    for doc_id, doc_chunks in by_doc.items():
        if doc_id in docs:
            continue
        docs[doc_id] = {"document_id": doc_id,
                        "document_name": doc_chunks[0].document_name,
                        "chunks": len(doc_chunks),
                        "images": len({c.image_id for c in doc_chunks
                                       if c.image_id}),
                        "status": "completed"}
    return list(docs.values())


@app.get("/data/overview", response_model=DataOverviewOut)
def data_overview():
    """Panel de datos solo lectura, SOLO en modo demo: que hay guardado
    en la base de vectores y como se ve, para el evaluador. Es un GET
    que nunca modifica nada ni devuelve secretos: solo metadatos y
    textos que ya son visibles en las respuestas del chat. Fuera del
    modo demo no existe (404), para no exponer datos de un despliegue
    real."""
    if not settings.demo_mode:
        raise HTTPException(
            status_code=404, detail="No disponible fuera del modo demo")
    try:
        stats = container.store.stats()
        samples = container.store.sample_points(limit=5)
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if settings.embedding_provider == "openai":
        embedding_label = f"openai ({settings.embedding_model})"
    else:
        embedding_label = (
            f"hash ({settings.embedding_dimension} dims, solo demo)")
    return DataOverviewOut(
        vector_store=settings.vector_store,
        collection=stats.get("collection", ""),
        points=stats.get("points", 0),
        dimension=stats.get("dimension", 0),
        embedding=embedding_label,
        samples=[SamplePointOut(
            document_name=c.document_name, page=c.page, kind=c.kind,
            section=c.section or "", text=c.text[:300]) for c in samples])


@app.post("/llm/detect", response_model=DetectResponse)
def detect_llm(req: DetectRequest):
    # Nunca se registra ni se devuelve la clave: solo proveedor, modelos,
    # precios y modalidad (cuando el proveedor los da). Con browser_id y
    # sin api_key, se detecta con el token guardado (visitante o
    # predeterminado) sin revelar su valor.
    api_key = req.api_key
    base_url = req.base_url
    if not api_key and req.browser_id:
        try:
            config, _source = container.token_vault.resolve(req.browser_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except TokenVaultUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo usar el token guardado: {exc}")
        if config is None:
            return DetectResponse(
                detected=False,
                message="No hay token guardado para este navegador ni "
                        "predeterminado. Pega un token.")
        api_key = config["api_key"]
        # Solo custom usa base_url propia; openai/openrouter se
        # detectan por su orden habitual (pasar su URL forzaria custom).
        base_url = config.get("base_url") if config.get("provider") == "custom" else None
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Pega un token o usa un token guardado.")
    provider, info = detect_provider_info(api_key, base_url)
    if provider is None:
        return DetectResponse(
            detected=False,
            message="No se reconoció el token en OpenAI ni en OpenRouter. "
                    "Puedes configurarlo manual con proveedor y base_url.")
    base = base_url if provider == "custom" else PROVIDERS[provider]
    models_info = [ModelInfoOut(**m) for m in info]
    return DetectResponse(detected=True, provider=provider, base_url=base,
                          models=[m.id for m in models_info],
                          models_info=models_info,
                          message=f"Proveedor detectado: {provider} "
                                  f"({len(models_info)} modelos)")


def _token_status(browser_id: str | None) -> TokenStatus:
    """Estado del token para un navegador, sin el valor. Nunca lanza
    por problemas de la boveda: los refleja en vault_error para que la
    UI explique que el guardado no esta usable."""
    vault = container.token_vault
    status = TokenStatus(vault_configured=vault.configured,
                         has_default=vault.has_default())
    if browser_id:
        try:
            validate_browser_id(browser_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    try:
        visitor = vault.load_visitor(browser_id) if browser_id else None
        default = vault.load_default()
    except TokenVaultUnavailable as exc:
        status.vault_error = str(exc)
        return status
    active = visitor if visitor is not None else default
    status.has_visitor_token = visitor is not None
    if visitor is not None:
        status.active = "visitor"
    elif default is not None:
        status.active = "default"
    if active is not None:
        status.provider = active.get("provider", "openai")
        status.base_url = active.get("base_url")
        status.model = active.get("model")
        status.auto_model = bool(active.get("auto_model", False))
        info = active.get("models_info") or []
        status.models_info = [ModelInfoOut(**m) for m in info
                              if isinstance(m, dict) and m.get("id")]
    return status


@app.post("/llm/token", response_model=TokenStatus)
def save_token(req: TokenSaveRequest):
    """Guarda el token del visitante, cifrado en el servidor. La
    predeterminada NO se puede poner ni borrar desde aqui: solo el
    dueno la establece con el script en el servidor."""
    vault = container.token_vault
    if not vault.configured:
        raise HTTPException(
            status_code=503,
            detail="El guardado de tokens no esta configurado en el "
                   "servidor: falta TOKEN_ENCRYPTION_KEY. Puedes pegar el "
                   "token sin guardarlo, o pedir al dueno que configure "
                   "la clave.")
    config = {"api_key": req.api_key, "provider": req.provider,
              "base_url": req.base_url, "model": req.model,
              "auto_model": req.auto_model,
              "models_info": ([m.model_dump() for m in req.models_info]
                              if req.models_info else None)}
    try:
        vault.save_visitor(req.browser_id, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TokenVaultUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _token_status(req.browser_id)


@app.get("/llm/token/status", response_model=TokenStatus)
def token_status(browser_id: str | None = None):
    return _token_status(browser_id)


@app.delete("/llm/token", response_model=TokenStatus)
def delete_token(browser_id: str):
    """Borra el token del visitante y vuelve al predeterminado (si lo
    hay) o al modo demo. La predeterminada no se toca."""
    try:
        container.token_vault.delete_visitor(browser_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _token_status(browser_id)


def _resolve_key_and_selection(provider: str, base_url: str | None,
                               model: str | None, auto_model: bool,
                               models_info_json: str | None, api_key: str,
                               browser_id: str | None):
    """Resuelve la clave efectiva y la seleccion de modelo.

    Orden de la clave: token explicito de la peticion > token guardado
    del visitante (por browser_id) > predeterminado compartido > nada
    (demo). La seleccion (proveedor, modelo, automatico) la pone la
    peticion cuando trae una; si no trae ninguna y la clave viene de
    la boveda, se usa la guardada. El modo automatico exige lista de
    modelos (como siempre): sin lista, _resolve_llm devuelve el 400
    habitual. Devuelve (api_key, provider, base_url, model, auto_model,
    models_info_json, origen) con origen en request, visitor, default o
    demo. Nunca devuelve el valor de la clave en errores: los mensajes
    solo hablan de configuracion."""
    if api_key:
        return (api_key, provider, base_url, model, auto_model,
                models_info_json, "request")
    try:
        config, source = container.token_vault.resolve(browser_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TokenVaultUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo usar el token guardado: {exc}")
    if config is None:
        return (None, provider, base_url, model, auto_model,
                models_info_json, "demo")
    if model or auto_model:
        return (config["api_key"], provider, base_url, model, auto_model,
                models_info_json, source)
    sel_info = (json.dumps(config.get("models_info"))
                if config.get("models_info") else None)
    return (config["api_key"], config.get("provider", "openai"),
            config.get("base_url"), config.get("model"),
            bool(config.get("auto_model", False)), sel_info, source)


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
    llm_in = req.llm
    api_key, provider, base_url, model, auto_model, models_info_json, source = (
        _resolve_key_and_selection(
            llm_in.provider if llm_in else "openai",
            llm_in.base_url if llm_in else None,
            llm_in.model if llm_in else None,
            llm_in.auto_model if llm_in else False,
            json.dumps([m.model_dump() for m in llm_in.models_info])
            if llm_in and llm_in.models_info else None,
            llm_in.api_key if llm_in else "",
            req.browser_id))
    if api_key:
        llm, model_used, model_auto = _resolve_llm(
            provider, model, api_key, base_url, auto_model,
            models_info_json, need_vision=False)
        use_case = AskUseCase(llm, container.ask.embedding, container.ask.store)
        token_source = source
    else:
        use_case = container.ask
        model_used, model_auto, token_source = "demo", False, "demo"
    # Antes, cualquier fallo del proveedor (401, 404, timeout, conexion,
    # respuesta con forma inesperada) escapaba sin mapear y FastAPI
    # devolvia un 500 "Internal Server Error" en el chat. Ahora se
    # traduce a 502 con un mensaje limpio, igual que en audio e imagen.
    try:
        answer = use_case.ask(req.question, top_k=req.top_k)
    except StoreUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="El servicio de busqueda no esta disponible en este "
                   "momento. Intentalo de nuevo en unos minutos.")
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
        image_url=image_url, model_used=model_used, model_auto=model_auto,
        token_source=token_source)


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
              top_k: int = Form(5),
              browser_id: str | None = Form(None)):
    """Pregunta por audio: transcribe y luego pregunta al RAG con el texto.
    La clave puede venir pegada en la peticion o resuelta en el servidor
    (token guardado del visitante o predeterminado). Sin ninguna, no se
    transcribe nada y no se inventa una. El audio no se guarda como
    documento."""
    api_key, provider, base_url, model, auto_model, models_info_json, source = (
        _resolve_key_and_selection(
            provider, base_url, model, auto_model, models_info_json,
            api_key or "", browser_id))
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
    except StoreUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="El servicio de busqueda no esta disponible en este "
                   "momento. Intentalo de nuevo en unos minutos.")
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
        transcription_model=trans_model, token_source=source)


@app.post("/ask-image", response_model=ImageAskResponse)
def ask_image(image: UploadFile = File(...),
              provider: str = Form("openai"),
              model: str | None = Form(None),
              api_key: str | None = Form(None),
              base_url: str | None = Form(None),
              auto_model: bool = Form(False),
              models_info_json: str | None = Form(None),
              question: str = Form(""),
              top_k: int = Form(5),
              browser_id: str | None = Form(None)):
    """Pregunta con imagen: el modelo de vision la describe y con esa
    descripcion se busca en el RAG a que hace referencia. La imagen
    enviada NO se indexa como documento. La clave puede venir pegada
    en la peticion o resuelta en el servidor (token guardado del
    visitante o predeterminado). Si el modelo no soporta vision,
    error claro, sin descripcion inventada."""
    api_key, provider, base_url, model, auto_model, models_info_json, source = (
        _resolve_key_and_selection(
            provider, base_url, model, auto_model, models_info_json,
            api_key or "", browser_id))
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
    except StoreUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="El servicio de busqueda no esta disponible en este "
                   "momento. Intentalo de nuevo en unos minutos.")
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
        image_url=image_url, model_used=model_used, model_auto=model_auto,
        token_source=source)


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
