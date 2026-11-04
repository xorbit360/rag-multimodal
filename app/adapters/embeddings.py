"""Embeddings: OpenAI-compatible (OpenAI / OpenRouter / Ollama) + hash local.

El adaptador OpenAI-compatible solo cambia base_url y modelo: por eso el
mismo codigo sirve para OpenAI, OpenRouter y Ollama local. El adaptador
hash es determinista, offline y gratuito: sirve para tests y demo, NO es
la recomendacion para produccion (ver README).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re

import httpx

from app.adapters.chunking import tokenize
from app.infrastructure.resilience import CircuitBreaker, retry_with_backoff

log = logging.getLogger(__name__)


class HashEmbedding:
    """Embedding determinista por hashing de tokens. Sin red, sin costo."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            for token in tokenize(text):
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dimension
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAICompatibleEmbedding:
    def __init__(self, api_key: str, base_url: str, model: str, dimension: int = 384):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension
        self._breaker = CircuitBreaker()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        def _call() -> list[list[float]]:
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
                timeout=60.0,
            )
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]
        raw = self._breaker.call(lambda: retry_with_backoff(_call))
        # Normalizar a la dimension configurada: un modelo real puede
        # devolver mas dimensiones que la coleccion (por ejemplo
        # text-embedding-3-small devuelve 1536 y la coleccion se crea
        # a 384). Sin esto, la ingesta fallaria en Qdrant por tamano
        # de vector. Los modelos de la familia 3 son Matryoshka, asi
        # que truncar conserva la mayor parte de la senal; si un
        # vector viniera mas corto, se rellena con ceros. Luego se
        # renormaliza para que el coseno siga siendo comparable.
        out: list[list[float]] = []
        for vec in raw:
            v = list(vec[:self._dimension])
            v += [0.0] * (self._dimension - len(v))
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out
