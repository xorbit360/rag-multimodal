from app.adapters.embeddings import HashEmbedding
from app.adapters.vector_store import InMemoryVectorStore
from app.domain.models import Chunk
from app.domain.retrieval import bm25_scores, hybrid_retrieve


def _store():
    store = InMemoryVectorStore()
    emb = HashEmbedding()
    chunks = [
        Chunk(id="c1", document_id="d", document_name="Manual_Motor.pdf", page=3,
              text="El par de apriete de los tornillos de culata es de 24 Nm.", bbox=None),
        Chunk(id="c2", document_id="d", document_name="Manual_Motor.pdf", page=2,
              text="La potencia maxima es de 12 kW a 3600 rpm.", bbox=None),
        Chunk(id="c3", document_id="d", document_name="Manual_Motor.pdf", page=5,
              text="Si el motor no arranca, revise el combustible y la bujia.", bbox=None),
    ]
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    return store, emb, chunks


def test_bm25_ranks_exact_term_first():
    _, _, chunks = _store()
    scores = bm25_scores("par de apriete culata", chunks)
    assert max(scores, key=scores.get) == "c1"


def test_hybrid_retrieval_returns_torque_chunk_first():
    store, emb, _ = _store()
    results = hybrid_retrieve("Cual es el par de apriete de los tornillos de culata?",
                              emb, store, top_k=2)
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0
