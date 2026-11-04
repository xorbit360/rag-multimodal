"""GET /documents: union del registro de jobs y el contenido real del
vector store. El registro vive en memoria y se pierde al reiniciar el
contenedor; los chunks persisten en Qdrant. Sin esta union, tras un
reinicio la lista salia vacia aunque el store tuviera los puntos."""

from fastapi.testclient import TestClient

import app.main as main_module
from app.domain.models import Chunk, Job, JobStatus
from app.domain.ports import StoreUnavailableError
from app.main import app

client = TestClient(app)


class FakeJobs:
    def __init__(self, jobs=()):
        self._jobs = list(jobs)

    def all(self):
        return list(self._jobs)


class FakeStore:
    def __init__(self, chunks=(), fail=False):
        self._chunks = list(chunks)
        self._fail = fail

    def all_chunks(self):
        if self._fail:
            raise StoreUnavailableError("qdrant caido")
        return list(self._chunks)


def _chunk(doc_id, name, image_id=None):
    return Chunk(id="c", document_id=doc_id, document_name=name, page=1,
                 text="t", bbox=None, kind="text", image_id=image_id,
                 section="")


def test_documents_from_store_after_restart(monkeypatch):
    # Sin jobs (registro perdido al reiniciar) pero con chunks en el
    # store: la lista muestra el documento real, no vacia.
    monkeypatch.setattr(main_module.container, "jobs", FakeJobs([]))
    monkeypatch.setattr(main_module.container, "store", FakeStore([
        _chunk("d1", "Manual_Motor.pdf", image_id="img1"),
        _chunk("d1", "Manual_Motor.pdf"),
        _chunk("d1", "Manual_Motor.pdf", image_id="img2")]))
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["document_name"] == "Manual_Motor.pdf"
    assert data[0]["chunks"] == 3
    assert data[0]["images"] == 2
    assert data[0]["status"] == "completed"


def test_documents_job_wins_over_store(monkeypatch):
    job = Job(id="j1", document_name="Manual_Motor.pdf", document_id="d1",
              chunks_count=18, images_count=4, status=JobStatus.COMPLETED)
    monkeypatch.setattr(main_module.container, "jobs", FakeJobs([job]))
    monkeypatch.setattr(main_module.container, "store", FakeStore([
        _chunk("d1", "Manual_Motor.pdf")]))
    data = client.get("/documents").json()
    assert len(data) == 1
    assert data[0]["chunks"] == 18  # el del job, no el del store
    assert data[0]["images"] == 4


def test_documents_store_unavailable_falls_back_to_jobs(monkeypatch):
    monkeypatch.setattr(main_module.container, "jobs", FakeJobs([]))
    monkeypatch.setattr(main_module.container, "store", FakeStore(fail=True))
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == []
