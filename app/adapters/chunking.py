"""Chunking por layout/semantica, NO solo por caracteres.

Reglas:
1. Se corta por seccion: un encabezado abre una seccion nueva.
2. Las tablas son un chunk propio (no se parten por la mitad).
3. Cada chunk guarda pagina, bbox del primer bloque y seccion.
4. Dentro de una seccion larga, se parte por parrafos con solapamiento,
   para no perder el contexto del borde.
5. Cada imagen se asocia al chunk de texto mas cercano en la misma pagina
   (distancia entre centros de bbox); si no hay texto en la pagina, la
   imagen queda como chunk propio tipo 'image'.
"""
from __future__ import annotations

import re
import uuid

from app.domain.models import BoundingBox, Chunk, ExtractedImage, ParsedBlock, ParsedDocument

MAX_CHARS = 900
OVERLAP_CHARS = 120


def _split_with_overlap(text: str, max_chars: int = MAX_CHARS,
                        overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # Solapamiento: arrastra la cola del chunk anterior
            current = (current[-overlap:] + "\n" + para).strip() if current else para
            if len(current) > max_chars:  # parrafo gigante: corte duro solo aqui
                for i in range(0, len(current), max_chars - overlap):
                    chunks.append(current[i:i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [text]


def _nearest_image(block_bbox: BoundingBox, images: list[ExtractedImage]) -> str | None:
    same_page = [img for img in images if img.page == block_bbox.page]
    if not same_page:
        return None
    nearest = min(same_page, key=lambda img: block_bbox.distance_to(img.bbox))
    return nearest.id


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    section_blocks: list[ParsedBlock] = []

    def flush() -> None:
        nonlocal section_blocks
        if not section_blocks:
            return
        text = "\n".join(b.text for b in section_blocks)
        first = section_blocks[0]
        for part in _split_with_overlap(text):
            chunks.append(Chunk(
                id=str(uuid.uuid4()), document_id=doc.document_id,
                document_name=doc.document_name, page=first.page, text=part,
                bbox=first.bbox, kind="text",
                image_id=_nearest_image(first.bbox, doc.images), section=section))
        section_blocks = []

    # Las tablas van como chunk propio, fuera del flujo de la seccion.
    for block in sorted(doc.blocks, key=lambda b: (b.page, b.bbox.y0, b.bbox.x0)):
        if block.kind == "table":
            chunks.append(Chunk(
                id=str(uuid.uuid4()), document_id=doc.document_id,
                document_name=doc.document_name, page=block.page, text=block.text,
                bbox=block.bbox, kind="table",
                image_id=_nearest_image(block.bbox, doc.images), section=section))
            continue
        if block.is_heading:
            flush()
            section = block.text.split("\n")[0].strip()
            section_blocks = [block]
        else:
            section_blocks.append(block)
    flush()

    # Imagenes sin chunk de texto cercano en su pagina -> chunk propio.
    covered_images = {c.image_id for c in chunks if c.image_id}
    for img in doc.images:
        if img.id not in covered_images:
            chunks.append(Chunk(
                id=str(uuid.uuid4()), document_id=doc.document_id,
                document_name=doc.document_name, page=img.page,
                text=f"[Imagen en la pagina {img.page} del documento {doc.document_name}]",
                bbox=img.bbox, kind="image", image_id=img.id, section=section))
    return chunks


STOPWORDS = {
    "el", "la", "los", "las", "de", "en", "y", "a", "con", "por", "para",
    "es", "son", "un", "una", "unos", "unas", "que", "se", "su", "sus",
    "al", "del", "lo", "como", "más", "mas", "pero", "o", "si", "no",
    "cual", "cuál", "que", "este", "esta", "esto", "ese", "esa",
}


def tokenize(text: str) -> list[str]:
    """Tokens sin palabras vacias: sin esto, 'de' y 'es' hacen que
    cualquier pregunta parezca relevante para cualquier chunk."""
    return [t for t in re.findall(r"[a-z0-9áéíóúñü]+", text.lower())
            if t not in STOPWORDS]
