"""Parser multimodal con PyMuPDF (fitz) + pdfplumber como respaldo de tablas.

Extrae, por pagina:
- bloques de texto con su bounding box real (no texto plano sin posicion),
- tablas con find_tables; si PyMuPDF no detecta ninguna, pdfplumber,
- imagenes raster embebidas, guardadas en disco con pagina y bbox.

Las imagenes vectoriales (dibujos de reportlab sin raster) no aparecen en
get_images: por eso el PDF de ejemplo incrusta el diagrama como PNG.
"""
from __future__ import annotations

import logging
import os
import uuid

import fitz  # PyMuPDF

from app.domain.models import BoundingBox, ExtractedImage, ParsedBlock, ParsedDocument

log = logging.getLogger(__name__)


def _bbox(rect: fitz.Rect, page: int) -> BoundingBox:
    return BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1, page=page)


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    clean = [[(c or "").strip() for c in row] for row in rows if any(c for c in row)]
    if not clean:
        return ""
    lines = [" | ".join(clean[0]), " | ".join("---" for _ in clean[0])]
    lines += [" | ".join(row) for row in clean[1:]]
    return "\n".join(lines)


class PyMuPDFParser:
    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)

    def parse(self, path: str, document_id: str, document_name: str) -> ParsedDocument:
        doc = fitz.open(path)
        blocks: list[ParsedBlock] = []
        images: list[ExtractedImage] = []
        page_count = len(doc)
        try:
            for page_no, page in enumerate(doc, start=1):
                self._parse_text_blocks(page, page_no, blocks)
                self._parse_tables_pymupdf(page, page_no, blocks)
                self._parse_images(page, page_no, document_id, images)
            if not any(b.kind == "table" for b in blocks):
                self._parse_tables_pdfplumber(path, blocks)
        finally:
            doc.close()
        log.info("pdf parseado document=%s paginas=%s bloques=%s imagenes=%s",
                 document_name, page_count, len(blocks), len(images))
        return ParsedDocument(document_id=document_id, document_name=document_name,
                              blocks=blocks, images=images)

    def _parse_text_blocks(self, page: fitz.Page, page_no: int, blocks: list[ParsedBlock]) -> None:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            for line in block.get("lines", []):
                spans = [s.get("text", "") for s in line.get("spans", [])]
                text = "".join(spans).strip()
                if text:
                    lines.append(text)
            text = "\n".join(lines).strip()
            if not text:
                continue
            # Encabezado: linea corta, sin punto final largo, o numerada "1. Titulo"
            first = lines[0] if lines else ""
            is_heading = (
                len(first) <= 80
                and (first[:2].rstrip(".").isdigit() or first.isupper() or len(lines) == 1 and len(text) <= 60)
                and not text.endswith(".")
            )
            blocks.append(ParsedBlock(text=text, page=page_no,
                                      bbox=_bbox(fitz.Rect(block["bbox"]), page_no),
                                      kind="heading" if is_heading else "text",
                                      is_heading=is_heading))

    def _parse_tables_pymupdf(self, page: fitz.Page, page_no: int, blocks: list[ParsedBlock]) -> None:
        try:
            finder = page.find_tables()
            for table in finder:
                md = _table_to_markdown(table.extract())
                if md:
                    blocks.append(ParsedBlock(text=md, page=page_no,
                                              bbox=_bbox(table.bbox, page_no), kind="table"))
        except Exception as exc:  # tablas opcionales, no rompen la ingesta
            log.warning("find_tables fallo pagina=%s error=%s", page_no, exc)

    def _parse_tables_pdfplumber(self, path: str, blocks: list[ParsedBlock]) -> None:
        try:
            import pdfplumber
        except ImportError:
            return
        try:
            with pdfplumber.open(path) as pdf:
                for page_no, page in enumerate(pdf.pages, start=1):
                    for table in page.extract_tables() or []:
                        md = _table_to_markdown(table)
                        if md:
                            rect = BoundingBox(0, 0, 0, 0, page_no)
                            blocks.append(ParsedBlock(text=md, page=page_no, bbox=rect, kind="table"))
        except Exception as exc:
            log.warning("pdfplumber fallback fallo error=%s", exc)

    def _parse_images(self, page: fitz.Page, page_no: int, document_id: str,
                      images: list[ExtractedImage]) -> None:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
                rect = rects[0] if rects else page.rect
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_id = str(uuid.uuid4())
                filename = f"{image_id}.png"
                out_path = os.path.join(self.images_dir, filename)
                pix.save(out_path)
                images.append(ExtractedImage(id=image_id, document_id=document_id,
                                             page=page_no, bbox=_bbox(rect, page_no),
                                             path=out_path, filename=filename))
            except Exception as exc:
                log.warning("imagen no extraida pagina=%s xref=%s error=%s", page_no, xref, exc)
