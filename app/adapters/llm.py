"""LLM: OpenAI-compatible + FakeLLM extractivo para demo/tests.

OpenAICompatibleLLM habla el endpoint /chat/completions, que exponen
OpenAI, OpenRouter y Ollama (con /v1). Cambiar de proveedor es cambiar
base_url y modelo en .env. Las llamadas van con retry + circuit breaker.

FakeLLM NO llama a ninguna API de pago: responde extractivamente desde el
contexto que le pasa el caso de uso. Si el contexto esta vacio, devuelve
la negativa explicita. Sirve para demo sin costo y para tests offline.
"""
from __future__ import annotations

import logging
import re

import httpx

from app.domain.models import INSUFFICIENT_ANSWER
from app.infrastructure.resilience import CircuitBreaker, retry_with_backoff

log = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._breaker = CircuitBreaker()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        def _call() -> str:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        return self._breaker.call(lambda: retry_with_backoff(_call))


class FakeLLM:
    """Respuesta extractiva: devuelve las frases del contexto mas afines a la
    pregunta. Nunca inventa: si no hay contexto, negativa explicita."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "SIN CONTEXTO RECUPERADO" in user_prompt:
            return INSUFFICIENT_ANSWER
        context_match = re.search(r"CONTEXTO:\n(.*?)\nPREGUNTA:", user_prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        if not context:
            return INSUFFICIENT_ANSWER
        question_match = re.search(r"PREGUNTA:\s*(.*)$", user_prompt, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""
        q_words = {w for w in re.findall(r"[a-z0-9áéíóúñü]+", question.lower()) if len(w) > 3}
        sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for w in q_words if w in s.lower()),
            reverse=True,
        )
        best = [s for s in scored[:3] if s.strip()]
        if not best or (q_words and sum(1 for w in q_words if w in " ".join(best).lower()) == 0):
            # Contexto existe pero no responde: mejor la primera frase util
            # que una invencion. El caso de uso ya filtro por umbral antes.
            best = [sentences[0]] if sentences else []
        if not best:
            return INSUFFICIENT_ANSWER
        return "Segun los documentos recuperados: " + " ".join(best).strip()


PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def build_override_llm(provider: str, model: str, api_key: str,
                       base_url: str | None = None) -> OpenAICompatibleLLM:
    """Construye el adaptador LLM para un token pegado en la web."""
    if provider == "custom":
        if not base_url:
            raise ValueError("proveedor custom requiere base_url")
        url = base_url
    elif provider in PROVIDERS:
        url = PROVIDERS[provider]
    else:
        raise ValueError(f"proveedor desconocido: {provider}")
    return OpenAICompatibleLLM(api_key=api_key, base_url=url, model=model)


def _fetch_models(base_url: str, api_key: str):
    """GET /models autenticado. Devuelve la lista cruda o None."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return None


def _parse_models_info(raw_models) -> list[dict]:
    """Normaliza la lista del proveedor: id, precio de prompt (si lo trae,
    como OpenRouter en pricing.prompt) y modalidad (si la trae, como
    OpenRouter en architecture.modality, p. ej. 'text+image->text').
    OpenAI y custom no dan precios ni modalidad: quedan en None."""
    info = []
    for m in raw_models or []:
        model_id = m.get("id")
        if not model_id:
            continue
        prompt_price = None
        pricing = m.get("pricing") or {}
        try:
            if pricing.get("prompt") is not None:
                prompt_price = float(pricing["prompt"])
        except (TypeError, ValueError):
            prompt_price = None
        modality = (m.get("architecture") or {}).get("modality")
        info.append({"id": model_id, "prompt_price": prompt_price,
                     "modality": modality})
    return info


def detect_provider_info(api_key: str, base_url: str | None = None):
    """Igual que detect_provider, pero devuelve tambien precios y
    modalidad por modelo, para la seleccion automatica del mas economico.
    Devuelve (proveedor, models_info) o (None, []) si ninguno lo acepta."""
    if base_url:
        data = _fetch_models(base_url, api_key)
        if data is not None:
            return "custom", _parse_models_info(data)
        return None, []
    order = (["openrouter", "openai"] if api_key.startswith("sk-or-")
             else ["openai", "openrouter"])
    for name in order:
        try:
            if name == "openrouter":
                auth = httpx.get(f"{PROVIDERS[name]}/auth/key",
                                 headers={"Authorization": f"Bearer {api_key}"},
                                 timeout=10.0)
                if auth.status_code != 200:
                    continue
            data = _fetch_models(PROVIDERS[name], api_key)
            if data is not None:
                info = _parse_models_info(data)
                log.info("token detectado proveedor=%s modelos=%s", name, len(info))
                return name, info
        except Exception:
            continue
    return None, []


def detect_provider(api_key: str, base_url: str | None = None):
    """Detecta a que proveedor pertenece un token, probandolo contra
    endpoints autenticados. Nunca se registra ni se devuelve la clave.

    OpenAI: GET /v1/models exige clave valida.
    OpenRouter: /v1/models es publico, asi que la validez se prueba con
    GET /v1/auth/key y el catalogo se lista despues.
    Con base_url propio se prueba solo ese endpoint (custom).
    Devuelve (proveedor, modelos) o (None, []) si ninguno lo acepta.
    El modelo NO viene en el token: se lista desde el proveedor y el
    usuario lo elige en la web."""
    provider, info = detect_provider_info(api_key, base_url)
    return provider, [m["id"] for m in info]
