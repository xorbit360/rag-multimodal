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
