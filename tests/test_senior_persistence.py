"""Senior: JobStore persistente compartido, worker real y embeddings reales.

Estos tests no necesitan Redis ni Qdrant reales: usan un cliente Redis
falso en memoria con la misma interfaz (hset/hget/hvals/hdel/eval) que
usa RedisJobStore, para probar que el estado sobrevive a un "reinicio"
(una segunda instancia del store sobre el mismo cliente) y que un
worker que solo conoce el job_id puede re-ejecutar la ingesta desde
la pdf_path guardada en el Job.
"""
import sys
import types

from app.adapters.embeddings import (
    HashEmbedding, OpenAICompatibleEmbedding,
)
from app.adapters.queue import RedisJobStore, _job_from_dict, _job_to_dict
from app.config import Settings
from app.dependencies import Container
from app.domain.models import Job, JobStatus
from app.domain.use_cases import IngestUseCase


class FakeRedis:
    def __init__(self):
        self.hashes = {}

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hvals(self, key):
        return list(self.hashes.get(key, {}).values())

    def hdel(self, key, field):
        return self.hashes.get(key, {}).pop(field, None) is not None

    def eval(self, script, numkeys, key, *args):
        # Solo se usa para clear_hash con la guarda job_id.
        field, job_id = args
        if self.hashes.get(key, {}).get(field) == job_id:
            del self.hashes[key][field]
            return 1
        return 0


def _fake_redis_module(fake):
    module = types.ModuleType("redis")
    module.Redis = types.SimpleNamespace(from_url=lambda url, **kw: fake)
    return module


def test_job_roundtrip_preserves_pdf_path():
    job = Job(id="j1", document_name="Manual.pdf", file_hash="abc",
              pdf_path="/app/storage/uploads/x_Manual.pdf")
    restored = _job_from_dict(_job_to_dict(job))
    assert restored.pdf_path == "/app/storage/uploads/x_Manual.pdf"
    assert restored.file_hash == "abc"


def test_redis_jobstore_survives_new_instance(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", _fake_redis_module(fake))
    store1 = RedisJobStore("redis://localhost:6379/0")
    job = Job(id="j2", document_name="A.pdf", file_hash="h1",
              pdf_path="/tmp/a.pdf")
    store1.save(job)
    # "Reinicio": una segunda instancia sobre el mismo Redis ve el job.
    store2 = RedisJobStore("redis://localhost:6379/0")
    found = store2.get("j2")
    assert found is not None
    assert found.pdf_path == "/tmp/a.pdf"
    assert store2.get_by_hash("h1") is not None
    assert store2.get_by_hash("h1").id == "j2"
    # Un fallo libera el hash solo para ese job.
    store2.clear_hash("h1", "j2")
    assert store2.get_by_hash("h1") is None
    assert store2.get("j2") is not None


def test_worker_can_run_from_job_only(monkeypatch, tmp_path):
    """El worker llega solo con job_id: run() usa job.pdf_path."""
    from app.adapters.chunking import chunk_document
    from app.domain.models import Chunk

    class FakeParser:
        def parse(self, path, document_id, document_name):
            from app.domain.models import ParsedBlock, ParsedDocument, BoundingBox
            block = ParsedBlock(
                text="par de apriete 24 Nm", page=1,
                bbox=BoundingBox(0, 0, 10, 10, 1))
            return ParsedDocument(document_id=document_id,
                                  document_name=document_name, blocks=[block])

    class FakeStore:
        def __init__(self):
            self.chunks = []

        def upsert(self, chunks, vectors):
            self.chunks = chunks

    class NoopQueue:
        def enqueue(self, job, fn):
            pass

    fake = FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", _fake_redis_module(fake))
    store = RedisJobStore("redis://localhost:6379/0")
    use_case = IngestUseCase(FakeParser(), HashEmbedding(), FakeStore(),
                             NoopQueue(), store)
    job = use_case.start(str(tmp_path / "manual.pdf"), "manual.pdf",
                         file_hash="h9")
    assert job.pdf_path is not None
    # Simula el worker: nueva instancia del caso de uso, mismo store,
    # solo con el job_id.
    worker_case = IngestUseCase(FakeParser(), HashEmbedding(), FakeStore(),
                                NoopQueue(), store)
    worker_case.run(job.id)
    done = store.get(job.id)
    assert done.status == JobStatus.COMPLETED
    assert done.chunks_count >= 1


def test_embedding_selection_is_real_when_key_present(tmp_path):
    settings = Settings(
        storage_dir=str(tmp_path), embedding_provider="openai",
        openai_api_key="sk-test", vector_store="memory",
        queue_backend="memory", demo_mode=True)
    container = Container(settings)
    assert isinstance(container.embedding, OpenAICompatibleEmbedding)
    settings2 = Settings(
        storage_dir=str(tmp_path), embedding_provider="hash",
        vector_store="memory", queue_backend="memory", demo_mode=True)
    container2 = Container(settings2)
    assert isinstance(container2.embedding, HashEmbedding)
