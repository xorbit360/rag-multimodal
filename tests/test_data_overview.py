"""Panel de datos solo lectura: endpoint /data/overview y stats/sample
de los stores. Solo lectura: ningun test escribe ni borra nada real."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.adapters.vector_store import InMemoryVectorStore, QdrantVectorStore
from app.domain.models import Chunk
from app.domain.ports import StoreUnavailableError
from app.main import app

client = TestClient(app)


def _chunk():
    return Chunk(id="c1", document_id="d", document_name="Manual_Motor.pdf",
                 page=4, text="El par de apriete de los tornillos de culata es de 24 Nm.",
                 bbox=None, kind="table", image_id=None, section="Pares")


class FakeStore:
    def __init__(self, chunks=()):
        self._chunks = list(chunks)

    def stats(self):
        return {"collection": "memory", "points": len(self._chunks),
                "dimension": 384}

    def sample_points(self, limit=5):
        return list(self._chunks[:limit])


def test_overview_returns_stats_and_samples(monkeypatch):
    monkeypatch.setattr(main_module.container, "store", FakeStore([_chunk()]))
    resp = client.get("/data/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["points"] == 1
    assert data["dimension"] == 384
    assert data["samples"][0]["page"] == 4
    assert "24 Nm" in data["samples"][0]["text"]
    assert data["vector_store"] in ("memory", "qdrant")


def test_overview_empty_store_returns_zero_samples(monkeypatch):
    monkeypatch.setattr(main_module.container, "store", FakeStore([]))
    resp = client.get("/data/overview")
    assert resp.status_code == 200
    assert resp.json()["points"] == 0
    assert resp.json()["samples"] == []


def test_overview_store_unavailable_returns_503(monkeypatch):
    class RaisingStore:
        def stats(self):
            raise StoreUnavailableError("qdrant caido")

        def sample_points(self, limit=5):
            raise StoreUnavailableError("qdrant caido")

    monkeypatch.setattr(main_module.container, "store", RaisingStore())
    resp = client.get("/data/overview")
    assert resp.status_code == 503


def test_overview_outside_demo_mode_returns_404(monkeypatch):
    monkeypatch.setattr(main_module, "settings",
                        SimpleNamespace(demo_mode=False))
    resp = client.get("/data/overview")
    assert resp.status_code == 404


def test_memory_store_stats_and_sample():
    store = InMemoryVectorStore()
    assert store.stats()["points"] == 0
    assert store.sample_points() == []
    store.upsert([_chunk()], [[0.1] * 384])
    assert store.stats() == {"collection": "memory", "points": 1,
                             "dimension": 384}
    assert store.sample_points()[0].page == 4


class _FakePoint:
    def __init__(self, payload, score=0.9):
        self.payload = payload
        self.score = score


class _FakeQdrantClient:
    def __init__(self, exists=True, points=()):
        self._exists = exists
        self._points = list(points)

    def collection_exists(self, name):
        return self._exists

    def query_points(self, collection_name, query, limit):
        return SimpleNamespace(points=self._points[:limit])

    def scroll(self, collection_name, limit, offset, with_payload):
        return (self._points[:limit], None)

    def count(self, collection_name):
        return SimpleNamespace(count=len(self._points))

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"size": 384})))


def _qdrant_store(fake_client):
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = fake_client
    store._collection = "rag_chunks"
    store._dimension = 384
    return store


def test_qdrant_search_uses_query_points():
    payload = {"chunk_id": "c1", "document_id": "d",
               "document_name": "Manual_Motor.pdf", "page": 4,
               "text": "El par de apriete es de 24 Nm.", "kind": "table",
               "image_id": None, "section": "Pares"}
    store = _qdrant_store(_FakeQdrantClient(
        exists=True, points=[_FakePoint(payload, score=0.8)]))
    results = store.search([0.1] * 384, top_k=5)
    assert len(results) == 1
    assert results[0].semantic_score == 0.8
    assert results[0].chunk.page == 4


def test_qdrant_stats_and_sample():
    payload = {"chunk_id": "c1", "document_id": "d",
               "document_name": "Manual_Motor.pdf", "page": 4,
               "text": "El par de apriete es de 24 Nm.", "kind": "table",
               "image_id": None, "section": "Pares"}
    store = _qdrant_store(_FakeQdrantClient(
        exists=True, points=[_FakePoint(payload)]))
    assert store.stats() == {"collection": "rag_chunks", "points": 1,
                             "dimension": 384}
    assert store.sample_points()[0].page == 4


def test_qdrant_stats_missing_collection_returns_zeros():
    store = _qdrant_store(_FakeQdrantClient(exists=False))
    assert store.stats() == {"collection": "rag_chunks", "points": 0,
                             "dimension": 384}
    assert store.sample_points() == []
