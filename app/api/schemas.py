from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfoOut(BaseModel):
    """Un modelo tal como lo lista el proveedor: id, precio de prompt por
    token (solo si el proveedor lo devuelve, como OpenRouter) y modalidad
    (solo si la devuelve, p. ej. 'text+image->text'). Sin precios ni
    modalidad, la seleccion automatica usa la tabla de costo conocido."""
    id: str
    prompt_price: float | None = None
    modality: str | None = None


class LlmOverride(BaseModel):
    """Token y modelo elegidos en la web. Con api_key no vacio, el
    token vive solo en la peticion: nunca se guarda en disco ni en
    logs. Con api_key vacio, la clave se resuelve en el servidor: el
    token guardado del visitante (por browser_id) o el predeterminado
    compartido; sin ninguno, modo demo.

    Con auto_model=True el backend elige el modelo mas economico segun
    lo que requiera la pregunta (vision si hay imagen, transcripcion si
    hay audio, chat si es solo texto) usando models_info cuando viene, y
    la tabla de costo conocido cuando no. Con auto_model=False se respeta
    el modelo manual. El campo model es opcional en modo automatico."""
    provider: str = "openai"  # openai | openrouter | custom
    base_url: str | None = None
    model: str | None = None
    api_key: str = ""
    auto_model: bool = False
    models_info: list[ModelInfoOut] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 5
    llm: LlmOverride | None = None
    # Identificador aleatorio del navegador: solo sirve para buscar
    # el token guardado de ese visitante. Nunca es el token.
    browser_id: str | None = None


class SourceOut(BaseModel):
    document_name: str
    page: int
    kind: str
    image_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    image_id: str | None = None
    image_page: int | None = None
    image_url: str | None = None
    # Modelo que realmente genero la respuesta, para mostrarlo en la UI.
    model_used: str | None = None
    model_auto: bool = False
    # Origen del token usado: request (pegado en esta sesion), visitor
    # (guardado del visitante), default (predeterminado compartido) o
    # demo (sin token). Nunca incluye el valor del token.
    token_source: str | None = None


class AudioAskResponse(BaseModel):
    transcription: str
    answer: str
    sources: list[SourceOut]
    image_id: str | None = None
    image_page: int | None = None
    image_url: str | None = None
    model_used: str | None = None
    model_auto: bool = False
    transcription_model: str | None = None
    token_source: str | None = None


class ImageAskResponse(BaseModel):
    description: str
    answer: str
    sources: list[SourceOut]
    image_id: str | None = None
    image_page: int | None = None
    image_url: str | None = None
    model_used: str | None = None
    model_auto: bool = False
    token_source: str | None = None


class JobOut(BaseModel):
    job_id: str
    document_name: str
    status: str
    progress: float
    message: str
    chunks_count: int = 0
    images_count: int = 0
    error: str | None = None
    # True cuando el job reutilizo una ingesta anterior (mismo SHA-256).
    duplicate: bool = False
    document_id: str | None = None


class DetectRequest(BaseModel):
    # Token pegado ahora (detect clasico) o identificador del navegador
    # para detectar con el token guardado (visitante o predeterminado).
    # Al menos uno debe venir; el valor del token nunca se devuelve.
    api_key: str = ""
    base_url: str | None = None
    browser_id: str | None = None


class TokenSaveRequest(BaseModel):
    """Guardar el token del visitante, cifrado en el servidor. El
    resto de campos son la configuracion elegida en la web (proveedor,
    modelo, seleccion) para reconstruirla al volver sin volver a pegar
    el token. Nada de esto devuelve jamas el valor del token."""
    browser_id: str
    api_key: str = Field(min_length=1)
    provider: str = "openai"
    base_url: str | None = None
    model: str | None = None
    auto_model: bool = False
    models_info: list[ModelInfoOut] | None = None


class TokenStatus(BaseModel):
    """Estado del token para un navegador, sin el valor: que hay
    guardado, cual esta activo y con que configuracion (sin secreto).
    vault_error explica por que el guardado no esta usable (por
    ejemplo, falta la clave de cifrado) sin exponer nada secreto."""
    has_visitor_token: bool = False
    has_default: bool = False
    # visitor (token propio guardado) | default (predeterminado
    # compartido) | demo (sin token).
    active: str = "demo"
    vault_configured: bool = False
    vault_error: str | None = None
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    auto_model: bool = False
    models_info: list[ModelInfoOut] = []


class DetectResponse(BaseModel):
    detected: bool
    provider: str | None = None
    base_url: str | None = None
    models: list[str] = []
    models_info: list[ModelInfoOut] = []
    message: str
