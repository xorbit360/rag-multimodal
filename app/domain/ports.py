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

    def stats(self) -> dict:
        """Solo lectura, para el panel de datos: coleccion, puntos y
        dimension. Nunca modifica nada."""
        ...

    def sample_points(self, limit: int = 5) -> list[Chunk]:
        """Solo lectura, para el panel de datos: primeros chunks
        guardados, de ejemplo. Nunca modifica nada."""
        ...


class LLMPort(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class VisionNotSupportedError(Exception):
    """El proveedor o el modelo no acepta imagenes (vision)."""


class StoreUnavailableError(Exception):
    """El vector store no responde (conexion caida, coleccion danada u
    otro error del motor). No es un fallo del core: el endpoint lo
    traduce a un 503 limpio, nunca a un 500. Una coleccion inexistente
    NO es este error: significa 'sin documentos' y se devuelve vacio."""


class ProviderResponseError(Exception):
    """El proveedor devolvio una respuesta con forma inesperada (sin
    opciones, contenido vacio o cuerpo no JSON). No es un fallo del
    core: el endpoint lo traduce a un error limpio, nunca a un 500."""


class TranscriptionPort(Protocol):
    def transcribe(self, audio: bytes, filename: str, model: str) -> str:
        """Transcribe un audio con el endpoint OpenAI-compatible
        /audio/transcriptions. Devuelve solo el texto."""
        ...


class VisionPort(Protocol):
    def describe(self, image: bytes, mime: str) -> str:
        """Describe una imagen con un modelo de vision (chat completions
        con image_url). Si el modelo no soporta vision, lanza
        VisionNotSupportedError: nunca se inventa una descripcion."""
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
