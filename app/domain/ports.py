"""Ports (interfaces) de la arquitectura hexagonal.

El core solo conoce estos Protocols. Cambiar de OpenAI a Ollama o a
OpenRouter es cambiar OPENAI_BASE_URL / OPENAI_MODEL, no el core; cambiar
de Qdrant a memoria es cambiar el adaptador, no los casos de uso.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .models import Chunk, Job, ParsedDocument, RetrievedChunk


class PDFParserPort(Protocol):
    def parse(self, path: str, document_id: str, document_name: str) -> ParsedDocument:
        ...


class EmbeddingPort(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStorePort(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        ...

    def all_chunks(self) -> list[Chunk]:
        """Necesario para la parte de palabras clave del retrieval hibrido."""
        ...


class LLMPort(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class JobQueuePort(Protocol):
    def enqueue(self, job: Job, fn: Callable[[], None]) -> None:
        """Encola el trabajo. La implementacion decide como se ejecuta."""
        ...


class JobStorePort(Protocol):
    def save(self, job: Job) -> None:
        ...

    def get(self, job_id: str) -> Job | None:
        ...

    def all(self) -> list[Job]:
        ...
