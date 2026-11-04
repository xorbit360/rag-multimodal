from app.adapters.chunking import chunk_document
from app.domain.models import BoundingBox, ExtractedImage, ParsedBlock, ParsedDocument


def _doc():
    def b(text, page, y, kind="text", heading=False):
        return ParsedBlock(text=text, page=page,
                           bbox=BoundingBox(0, y, 100, y + 20, page),
                           kind=kind, is_heading=heading)
    blocks = [
        b("1. Mantenimiento", 1, 0, kind="heading", heading=True),
        b("Cambie el aceite cada 250 horas.", 1, 30),
        b("2. Pares de apriete", 1, 60, kind="heading", heading=True),
        b("Tornillos de culata | 24 Nm", 1, 90, kind="table"),
    ]
    images = [ExtractedImage(id="img1", document_id="d1", page=1,
                             bbox=BoundingBox(0, 85, 100, 120, 1),
                             path="/tmp/x.png", filename="x.png")]
    return ParsedDocument(document_id="d1", document_name="Manual.pdf",
                          blocks=blocks, images=images)


def test_chunking_splits_by_heading_and_keeps_table_whole():
    chunks = chunk_document(_doc())
    sections = {c.section for c in chunks}
    assert "1. Mantenimiento" in sections
    assert "2. Pares de apriete" in sections
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert "24 Nm" in tables[0].text


def test_chunking_associates_nearest_image_same_page():
    chunks = chunk_document(_doc())
    table = [c for c in chunks if c.kind == "table"][0]
    assert table.image_id == "img1"


def test_chunking_long_section_splits_with_overlap():
    long_text = " ".join(f"frase numero {i} sobre mantenimiento." for i in range(200))
    doc = ParsedDocument(document_id="d1", document_name="M.pdf", blocks=[
        ParsedBlock(text="1. Seccion", page=1, bbox=BoundingBox(0, 0, 10, 10, 1),
                    kind="heading", is_heading=True),
        ParsedBlock(text=long_text, page=1, bbox=BoundingBox(0, 20, 10, 30, 1)),
    ])
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 950 for c in chunks)
