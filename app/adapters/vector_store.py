"""Vector stores: en memoria (tests/demo) + Qdrant (produccion).

InMemoryVectorStore guarda chunks + vectores y hace busqueda coseno.
QdrantVectorStore delega la busqueda semantica a Qdrant; la busqueda por
palabras clave se hace en Python sobre los chunks (BM25, ver retrieval),
porque Qdrant no es un motor BM25 nativo para este caso.
"""
from __future__ import annotations

import logging
import math
import uuid

from app.domain.models import Chunk, RetrievedChunk

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=c, semantic_score=_cosine(vector, v))
            for c, v in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.semantic_score, reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


class QdrantVectorStore:
    """Adaptador Qdrant. La importacion es perezosa para que los tests y la
    demo en memoria no necesiten qdrant-client ni un servidor Qdrant."""

    def __init__(self, url: str, collection: str, dimension: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dimension = dimension
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v,
                        payload={"chunk_id": c.id, "document_id": c.document_id,
                                 "document_name": c.document_name, "page": c.page,
                                 "text": c.text, "kind": c.kind,
                                 "image_id": c.image_id, "section": c.section})
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        from qdrant_client.models import SearchRequest

        results = self._client.search(
            collection_name=self._collection, query_vector=vector, limit=top_k)
        out: list[RetrievedChunk] = []
        for r in results:
            p = r.payload
            chunk = Chunk(id=p["chunk_id"], document_id=p["document_id"],
                          document_name=p["document_name"], page=p["page"],
                          text=p["text"], bbox=None, kind=p.get("kind", "text"),
                          image_id=p.get("image_id"), section=p.get("section", ""))
            out.append(RetrievedChunk(chunk=chunk, semantic_score=float(r.score)))
        return out

    def all_chunks(self) -> list[Chunk]:
        # Scroll completo: aceptable para el volumen de la prueba; en
        # produccion con 100+ manuales se pagina y se cachea el indice BM25.
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection, limit=100, offset=offset,
                with_payload=True)
            for pt in points:
                p = pt.payload
                chunks.append(Chunk(id=p["chunk_id"], document_id=p["document_id"],
                                    document_name=p["document_name"], page=p["page"],
                                    text=p["text"], bbox=None, kind=p.get("kind", "text"),
                                    image_id=p.get("image_id"),
                                    section=p.get("section", "")))
            if offset is None:
                break
        return chunks
