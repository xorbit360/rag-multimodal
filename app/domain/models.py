"""Entidades del dominio. Sin dependencias de FastAPI, PyMuPDF ni proveedores.

Un Chunk es la unidad recuperable: texto + posicion espacial (pagina y
bounding box) + posible imagen asociada. Nunca se inventa una fuente: cada
chunk guarda document_name y page reales, y la respuesta solo cita chunks
que existen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int  # 1-based, como la ve el usuario en el PDF

    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def distance_to(self, other: "BoundingBox") -> float:
        cx1, cy1 = self.center()
        cx2, cy2 = other.center()
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


@dataclass
class ExtractedImage:
    id: str
    document_id: str
    page: int
    bbox: BoundingBox
    path: str
    filename: str


@dataclass
class ParsedBlock:
    """Bloque crudo que entrega el parser, antes del chunking."""
    text: str
    page: int
    bbox: BoundingBox
    kind: str = "text"  # text | table | heading
    is_heading: bool = False


@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    page: int
    text: str
    bbox: Optional[BoundingBox]
    kind: str = "text"  # text | table | image
    image_id: Optional[str] = None
    section: str = ""


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    document_name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    document_id: Optional[str] = None
    chunks_count: int = 0
    images_count: int = 0
    error: Optional[str] = None
    # Hash SHA-256 del PDF original: deduplica re-ingestas del mismo archivo.
    file_hash: Optional[str] = None
    # True cuando este job reutilizo una ingesta anterior (mismo hash).
    duplicate: bool = False


@dataclass
class RetrievedChunk:
    chunk: Chunk
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0


@dataclass
class Source:
    document_name: str
    page: int
    kind: str
    chunk_id: str
    image_id: Optional[str] = None


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    image_id: Optional[str] = None
    image_page: Optional[int] = None


@dataclass
class AudioAnswer:
    """Pregunta por audio: transcripcion + respuesta normal del RAG."""
    transcription: str
    answer: Answer


@dataclass
class ImageAnswer:
    """Pregunta con imagen: descripcion del modelo de vision + respuesta
    normal del RAG. La imagen enviada NO se indexa: es insumo de pregunta."""
    description: str
    answer: Answer


INSUFFICIENT_ANSWER = (
    "No tengo información suficiente en los documentos para responder eso."
)
