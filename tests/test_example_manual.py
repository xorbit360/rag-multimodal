"""El manual de ejemplo real: paginas, chunks, imagenes y datos clave.

Estos valores son la verdad medida al ingerir examples/Manual_Motor.pdf
con el pipeline real (parser + chunking + embeddings hash + store en
memoria). Si el manual cambia, hay que regenerar y actualizar aqui los
numeros, no inventarlos: el torque de culata sigue siendo 24 Nm, pero
su pagina cambio respecto al manual viejo (antes pagina 3, ahora 4).
"""
import hashlib
import tempfile
from pathlib import Path

import fitz

from app.adapters.chunking import chunk_document
from app.adapters.embeddings import HashEmbedding
from app.adapters.pdf_parser import PyMuPDFParser
from app.adapters.queue import InMemoryJobQueue, InMemoryJobStore
from app.adapters.vector_store import InMemoryVectorStore
from app.domain.models import JobStatus
from app.domain.retrieval import hybrid_retrieve
from app.domain.use_cases import IngestUseCase
from app.main import EXAMPLE_MANUAL, seed_example_manual

MANUAL = str(EXAMPLE_MANUAL)


def _ingest(tmp_path):
    parser = PyMuPDFParser(images_dir=str(tmp_path / "images"))
    parsed = parser.parse(MANUAL, "d-manual", "Manual_Motor.pdf")
    chunks = chunk_document(parsed)
    return parsed, chunks


def test_manual_has_12_pages_and_4_images():
    doc = fitz.open(MANUAL)
    try:
        assert len(doc) == 12
        pages_with_images = [i + 1 for i, page in enumerate(doc)
                             if page.get_images()]
        assert pages_with_images == [3, 6, 7, 8]
    finally:
        doc.close()


def test_manual_ingests_to_18_chunks_5_tables():
    with tempfile.TemporaryDirectory() as tmp:
        parsed, chunks = _ingest(Path(tmp))
    assert len(parsed.images) == 4
    assert len(chunks) == 18
    kinds = [c.kind for c in chunks]
    assert kinds.count("table") == 5
    assert sum(1 for c in chunks if c.image_id) == 4


def test_manual_key_facts_on_exact_pages():
    with tempfile.TemporaryDirectory() as tmp:
        _, chunks = _ingest(Path(tmp))

    def pages_with(text):
        return sorted({c.page for c in chunks if text in c.text})

    assert pages_with("24 Nm") == [4]
    table_culata = [c for c in chunks
                    if c.kind == "table" and "24 Nm" in c.text]
    assert len(table_culata) == 1 and table_culata[0].page == 4
    assert pages_with("12 kW") == [2]
    assert pages_with("termostato") == [6, 10]
    assert pages_with("4,0 bar") == [7]
    assert pages_with("250 horas") == [1, 5]


def test_manual_retrieval_finds_torque_table():
    with tempfile.TemporaryDirectory() as tmp:
        _, chunks = _ingest(Path(tmp))
    store = InMemoryVectorStore()
    embedding = HashEmbedding(dimension=384)
    store.upsert(chunks, embedding.embed([c.text for c in chunks]))
    retrieved = hybrid_retrieve("par de apriete de los tornillos de culata",
                                embedding, store, top_k=5)
    assert retrieved
    # El mejor chunk es el parrafo de la pagina 4 (frase literal); la
    # tabla de pares tambien debe estar entre los recuperados.
    assert retrieved[0].chunk.page == 4
    assert "24 Nm" in retrieved[0].chunk.text
    assert any(c.chunk.kind == "table" and "24 Nm" in c.chunk.text
               for c in retrieved)


class _FakeSettings:
    demo_mode = True


def test_seed_ingests_real_manual_once(tmp_path):
    """El seed con el pipeline real: siembra 18 chunks y la segunda
    llamada no duplica nada; ademas, subir el mismo PDF por UI despues
    del seed se detecta como duplicado por el hash."""
    parser = PyMuPDFParser(images_dir=str(tmp_path / "images"))
    embedding = HashEmbedding(dimension=384)
    store = InMemoryVectorStore()
    jobs = InMemoryJobStore()
    ingest = IngestUseCase(parser, embedding, store,
                            InMemoryJobQueue(), jobs)

    class Container:
        pass

    container = Container()
    container.store = store
    container.ingest = ingest

    job = seed_example_manual(container, _FakeSettings())
    assert job is not None and job.status == JobStatus.COMPLETED
    assert job.chunks_count == 18
    assert job.images_count == 4
    assert len(store.all_chunks()) == 18

    assert seed_example_manual(container, _FakeSettings()) is None
    assert len(store.all_chunks()) == 18

    digest = hashlib.sha256(EXAMPLE_MANUAL.read_bytes()).hexdigest()
    reuse = ingest.start(str(EXAMPLE_MANUAL), "Manual_Motor.pdf",
                         file_hash=digest)
    assert reuse.duplicate is True
    assert len(store.all_chunks()) == 18
