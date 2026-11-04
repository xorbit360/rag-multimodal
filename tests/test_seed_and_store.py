"""Seed del manual de ejemplo + robustez del store Qdrant.

El seed corre en el lifespan de la API (no en importacion): estos tests
llaman a seed_example_manual directamente con contenedores falsos, sin
tocar el store real de la app. Los tests de Qdrant usan un cliente
falso: nunca se contacta un servidor Qdrant real.
"""
import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

import app.main as main_module
from app.adapters.vector_store import QdrantVectorStore
from app.domain.models import Job, JobStatus
from app.domain.ports import StoreUnavailableError
from app.main import app, seed_example_manual

client = TestClient(app)


# ---------------- Fakes del seed ----------------

class FakeStore:
    def __init__(self, chunks=()):
        self._chunks = list(chunks)

    def all_chunks(self):
        return list(self._chunks)


class FakeIngest:
    def __init__(self, job=None, fail=False):
        self.job = job
        self.fail = fail
        self.calls = []

    def run_sync(self, pdf_path, document_name, file_hash=None):
        self.calls.append((pdf_path, document_name, file_hash))
        if self.fail:
            raise RuntimeError("fallo de ingesta simulado")
        return self.job


class FakeContainer:
    def __init__(self, store, ingest):
        self.store = store
        self.ingest = ingest


class FakeSettings:
    def __init__(self, demo_mode=True):
        self.demo_mode = demo_mode


def _completed_job():
    return Job(id="seed-job", document_name="Manual_Motor.pdf",
               status=JobStatus.COMPLETED, progress=1.0,
               chunks_count=10, images_count=4)


def test_seed_runs_when_store_empty():
    ingest = FakeIngest(job=_completed_job())
    container = FakeContainer(FakeStore(), ingest)
    job = seed_example_manual(container, FakeSettings(demo_mode=True))
    assert job is not None and job.status == JobStatus.COMPLETED
    assert len(ingest.calls) == 1
    assert ingest.calls[0][0].endswith("Manual_Motor.pdf")
    assert ingest.calls[0][1] == "Manual_Motor.pdf"
    assert ingest.calls[0][2]  # hash SHA-256 registrado


def test_seed_skips_when_store_not_empty():
    ingest = FakeIngest(job=_completed_job())
    container = FakeContainer(FakeStore(chunks=["chunk"]), ingest)
    assert seed_example_manual(container, FakeSettings(demo_mode=True)) is None
    assert ingest.calls == []


def test_seed_skips_when_not_demo_mode():
    ingest = FakeIngest(job=_completed_job())
    container = FakeContainer(FakeStore(), ingest)
    assert seed_example_manual(container, FakeSettings(demo_mode=False)) is None
    assert ingest.calls == []


def test_seed_survives_store_error():
    class BrokenStore:
        def all_chunks(self):
            raise RuntimeError("qdrant caido")

    ingest = FakeIngest(job=_completed_job())
    container = FakeContainer(BrokenStore(), ingest)
    assert seed_example_manual(container, FakeSettings(demo_mode=True)) is None
    assert ingest.calls == []


def test_seed_survives_ingest_failure():
    ingest = FakeIngest(fail=True)
    container = FakeContainer(FakeStore(), ingest)
    assert seed_example_manual(container, FakeSettings(demo_mode=True)) is None


def test_seed_survives_failed_job():
    failed = Job(id="seed-fail", document_name="Manual_Motor.pdf",
                 status=JobStatus.FAILED, error="boom")
    ingest = FakeIngest(job=failed)
    container = FakeContainer(FakeStore(), ingest)
    assert seed_example_manual(container, FakeSettings(demo_mode=True)) is None


def test_seed_skips_missing_pdf(monkeypatch):
    ingest = FakeIngest(job=_completed_job())
    container = FakeContainer(FakeStore(), ingest)
    monkeypatch.setattr(main_module, "EXAMPLE_MANUAL",
                        Path("/no-existe/Manual_Motor.pdf"))
    assert seed_example_manual(container, FakeSettings(demo_mode=True)) is None
    assert ingest.calls == []


def test_lifespan_calls_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "seed_example_manual",
                        lambda container, settings: calls.append(True))

    async def _run():
        async with main_module.lifespan(app):
            pass

    asyncio.run(_run())
    assert calls == [True]


# ---------------- Qdrant: coleccion inexistente y errores ----------------

class FakePoint:
    def __init__(self, payload, score=0.9):
        self.payload = payload
        self.score = score


class FakeQdrantClient:
    def __init__(self, exists=True, points=(), fail=None):
        self._exists = exists
        self._points = list(points)
        self.fail = fail

    def collection_exists(self, name):
        if self.fail is not None:
            raise self.fail
        return self._exists

    def search(self, collection_name, query_vector, limit):
        if self.fail is not None:
            raise self.fail
        return self._points[:limit]

    def scroll(self, collection_name, limit, offset, with_payload):
        if self.fail is not None:
            raise self.fail
        return ([], None)


def _qdrant_store(fake_client):
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = fake_client
    store._collection = "rag_chunks"
    store._dimension = 384
    return store


def _payload():
    return {"chunk_id": "c1", "document_id": "d", "document_name": "Manual_Motor.pdf",
            "page": 4, "text": "El par de apriete de los tornillos de culata es de 24 Nm.",
            "kind": "table", "image_id": None, "section": "Pares"}


def test_qdrant_missing_collection_returns_empty():
    store = _qdrant_store(FakeQdrantClient(exists=False))
    assert store.search([0.1] * 384, top_k=5) == []
    assert store.all_chunks() == []


def test_qdrant_search_maps_payload():
    store = _qdrant_store(FakeQdrantClient(
        exists=True, points=[FakePoint(_payload(), score=0.8)]))
    results = store.search([0.1] * 384, top_k=5)
    assert len(results) == 1
    assert results[0].chunk.text.startswith("El par de apriete")
    assert results[0].chunk.page == 4
    assert results[0].semantic_score == 0.8


def test_qdrant_connection_error_becomes_store_unavailable():
    err = httpx.ConnectError("conexion rechazada",
                             request=httpx.Request("GET", "http://qdrant:6333"))
    store = _qdrant_store(FakeQdrantClient(fail=err))
    with pytest.raises(StoreUnavailableError):
        store.search([0.1] * 384, top_k=5)
    with pytest.raises(StoreUnavailableError):
        store.all_chunks()


def test_qdrant_unexpected_response_becomes_store_unavailable():
    err = UnexpectedResponse(500, "Internal Server Error", b"{}",
                             httpx.Headers())
    store = _qdrant_store(FakeQdrantClient(exists=True, fail=err))
    with pytest.raises(StoreUnavailableError):
        store.search([0.1] * 384, top_k=5)


# ---------------- Endpoint: 503 limpio ----------------

def test_ask_store_unavailable_returns_503(monkeypatch):
    class RaisingStore:
        def search(self, *args, **kwargs):
            raise StoreUnavailableError("qdrant caido")

        def all_chunks(self):
            return []

    monkeypatch.setattr(main_module.container.ask, "store", RaisingStore())
    resp = client.post("/ask", json={"question": "Cual es el par de apriete?"})
    assert resp.status_code == 503
    assert "busqueda" in resp.json()["detail"]
