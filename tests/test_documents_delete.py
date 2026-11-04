"""DELETE /documents/{document_id}: borra puntos, trabajos, hash y
archivos del documento, por document_id (no por nombre: dos ingestas
del mismo nombre se borran una por una). Tras borrar, el mismo PDF
se puede volver a ingerir sin aparecer como duplicado."""

from fastapi.testclient import TestClient

import app.main as main_module
from app.adapters.queue import InMemoryJobStore
from app.adapters.vector_store import InMemoryVectorStore
from app.domain.models import Chunk, Job, JobStatus
from app.main import app

client = TestClient(app)


def _setup(monkeypatch):
    store = InMemoryVectorStore()
    store.upsert(
        [Chunk(id="c1", document_id="d1", document_name="Manual_Motor.pdf",
               page=1, text="t1", bbox=None),
         Chunk(id="c2", document_id="d1", document_name="Manual_Motor.pdf",
               page=2, text="t2", bbox=None, image_id="img1"),
         Chunk(id="c3", document_id="d2", document_name="Otro.pdf",
               page=1, text="t3", bbox=None)],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    jobs = InMemoryJobStore()
    jobs.save(Job(id="j1", document_name="Manual_Motor.pdf",
                  document_id="d1", chunks_count=2, images_count=1,
                  status=JobStatus.COMPLETED, file_hash="abc"))
    jobs.save(Job(id="j2", document_name="Otro.pdf",
                  document_id="d2", chunks_count=1, images_count=0,
                  status=JobStatus.COMPLETED, file_hash="def"))
    monkeypatch.setattr(main_module.container, "store", store)
    monkeypatch.setattr(main_module.container, "jobs", jobs)
    return store, jobs


def test_delete_document_removes_only_that_document(monkeypatch):
    store, jobs = _setup(monkeypatch)
    resp = client.delete("/documents/d1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_name"] == "Manual_Motor.pdf"
    assert data["chunks_removed"] == 2
    assert data["jobs_removed"] == 1
    assert [c.document_id for c in store.all_chunks()] == ["d2"]
    assert jobs.get("j1") is None
    assert jobs.get("j2") is not None
    assert jobs.get_by_hash("abc") is None  # se puede volver a ingerir
    docs = client.get("/documents").json()
    assert [d["document_id"] for d in docs] == ["d2"]


def test_delete_unknown_document_is_404(monkeypatch):
    _setup(monkeypatch)
    resp = client.delete("/documents/no-existe")
    assert resp.status_code == 404
    # Nada se borro por el intento fallido.
    assert len(client.get("/documents").json()) == 2
