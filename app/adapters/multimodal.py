"""Adaptadores multimodales OpenAI-compatibles: transcripcion y vision.

Ambos hablan endpoints que exponen OpenAI y cualquier proveedor compatible
con su API (misma base_url del token pegado en la web). Las llamadas van
con retry + circuit breaker, igual que el LLM de chat.

Transcripcion: POST {base}/audio/transcriptions (multipart, modelo
whisper-1 por defecto). Vision: POST {base}/chat/completions con un mensaje
de usuario que lleva texto + image_url en base64.

Ninguno de los dos guarda la clave: solo vive en la instancia, construida
por peticion desde el override, y nunca se registra ni se devuelve.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.domain.ports import ProviderResponseError, VisionNotSupportedError
from app.infrastructure.resilience import CircuitBreaker, retry_with_backoff

log = logging.getLogger(__name__)

_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".mp4": "audio/mp4",
}

VISION_PROMPT = (
    "Describe esta imagen con detalle tecnico: objetos, texto visible, "
    "diagramas y partes. Se preciso y no inventes elementos que no se vean."
)


def audio_mime_for(filename: str) -> str:
    lower = filename.lower()
    for ext, mime in _AUDIO_MIME.items():
        if lower.endswith(ext):
            return mime
    return "audio/mpeg"


class OpenAICompatibleTranscription:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._breaker = CircuitBreaker()

    def transcribe(self, audio: bytes, filename: str, model: str) -> str:
        def _call() -> str:
            resp = httpx.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, audio, audio_mime_for(filename))},
                data={"model": model},
                timeout=300.0,
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if not text:
                raise ProviderResponseError(
                    "el proveedor devolvio una transcripcion vacia")
            return text
        return self._breaker.call(lambda: retry_with_backoff(_call))


class OpenAICompatibleVision:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._breaker = CircuitBreaker()

    def describe(self, image: bytes, mime: str) -> str:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "temperature": 0.0,
        }

        def _call() -> str:
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # 400/404/422 ante una imagen = el modelo no soporta vision.
                # No se inventa ninguna descripcion: se informa el limite.
                if exc.response.status_code in (400, 404, 422):
                    raise VisionNotSupportedError(
                        "El modelo elegido no acepta imagenes. Elige un modelo "
                        "con vision (por ejemplo, uno de la familia GPT-4o) o "
                        "pregunta solo con texto.") from exc
                raise
            try:
                data = resp.json()
                choices = data.get("choices") or []
                content = choices[0].get("message", {}).get("content") if choices else None
            except (ValueError, AttributeError, IndexError, KeyError, TypeError) as exc:
                raise ProviderResponseError(
                    "el proveedor devolvio una respuesta inesperada") from exc
            text = (content or "").strip()
            if not text:
                raise ProviderResponseError(
                    "el proveedor devolvio una descripcion vacia")
            return text
        return self._breaker.call(lambda: retry_with_backoff(_call))
