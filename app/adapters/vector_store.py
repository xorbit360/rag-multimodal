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

import httpx

from app.domain.models import Chunk, RetrievedChunk
from app.domain.ports import StoreUnavailableError

log = logging.getLogger(__name__)


def _qdrant_error(exc: Exception) -> StoreUnavailableError:
    """Traduce un fallo de Qdrant a error de dominio. Nunca se deja
    pasar la excepcion cruda del cliente: el endpoint la convierte
    en un 503 limpio."""
    return StoreUnavailableError(
        "El servicio de busqueda no esta disponible en este momento. "
        "Intentalo de nuevo en unos minutos.")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _chunk_from_payload(p: dict) -> Chunk:
    """Chunk desde el payload de un punto Qdrant. El bbox no se guarda
    en Qdrant (queda en el PDF): se reconstruye como None."""
    return Chunk(id=p["chunk_id"], document_id=p["document_id"],
                 document_name=p["document_name"], page=p["page"],
                 text=p["text"], bbox=None, kind=p.get("kind", "text"),
                 image_id=p.get("image_id"), section=p.get("section", ""))


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

    def stats(self) -> dict:
        dim = len(self._vectors[0]) if self._vectors else 0
        return {"collection": "memory", "points": len(self._chunks),
                "dimension": dim}

    def sample_points(self, limit: int = 5) -> list[Chunk]:
        return list(self._chunks[:limit])

    def delete_document(self, document_id: str) -> int:
        before = len(self._chunks)
        kept = [(c, v) for c, v in zip(self._chunks, self._vectors)
                if c.document_id != document_id]
        self._chunks = [c for c, _ in kept]
        self._vectors = [v for _, v in kept]
        return before - len(self._chunks)


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

    def _collection_exists(self) -> bool:
        """True si la coleccion existe. Un fallo de conexion u otro
        error de Qdrant se traduce a StoreUnavailableError (no a un
        booleano mentiroso ni a la excepcion cruda del cliente)."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            return bool(self._client.collection_exists(self._collection))
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        from qdrant_client.http.exceptions import UnexpectedResponse

        # Coleccion inexistente = sin documentos, no un error: se
        # devuelve vacio para que el caso de uso responda la negativa
        # defensiva en vez de un 500 (era el fallo visto en vivo al
        # preguntar antes de ingerir nada).
        if not self._collection_exists():
            return []
        try:
            # query_points es la API vigente desde qdrant-client 1.14:
            # .search() fue eliminado en 1.20 y toda pregunta fallaba
            # con AttributeError antes de llamar al LLM.
            resp = self._client.query_points(
                collection_name=self._collection, query=vector, limit=top_k)
            results = resp.points
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc
        out: list[RetrievedChunk] = []
        for r in results:
            out.append(RetrievedChunk(
                chunk=_chunk_from_payload(r.payload),
                semantic_score=float(r.score)))
        return out

    def all_chunks(self) -> list[Chunk]:
        from qdrant_client.http.exceptions import UnexpectedResponse

        # Scroll completo: aceptable para el volumen de la prueba; en
        # produccion con 100+ manuales se pagina y se cachea el indice BM25.
        if not self._collection_exists():
            return []
        chunks: list[Chunk] = []
        offset = None
        try:
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection, limit=100, offset=offset,
                    with_payload=True)
                for pt in points:
                    chunks.append(_chunk_from_payload(pt.payload))
                if offset is None:
                    break
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc
        return chunks

    def stats(self) -> dict:
        """Solo lectura, para el panel de datos: coleccion, puntos y
        dimension. Sin coleccion: ceros, no error."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        if not self._collection_exists():
            return {"collection": self._collection, "points": 0,
                    "dimension": self._dimension}
        try:
            count = self._client.count(collection_name=self._collection).count
            info = self._client.get_collection(collection_name=self._collection)
            vectors = getattr(info.config.params, "vectors", None)
            if isinstance(vectors, dict):
                dim = vectors.get("size", self._dimension)
            else:
                dim = getattr(vectors, "size", self._dimension)
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc
        return {"collection": self._collection, "points": int(count),
                "dimension": int(dim)}

    def sample_points(self, limit: int = 5) -> list[Chunk]:
        """Solo lectura, para el panel de datos: primeros puntos de
        ejemplo. Sin coleccion: vacio, no error."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        if not self._collection_exists():
            return []
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection, limit=limit, offset=None,
                with_payload=True)
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc
        return [_chunk_from_payload(pt.payload) for pt in points]

    def delete_document(self, document_id: str) -> int:
        """Borra por filtro document_id, no por ids de punto: los ids
        de punto son UUID aleatorios y no se guardan en ningun indice.
        Sin coleccion: 0, no error."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        from qdrant_client.models import (
            FieldCondition, Filter, FilterSelector, MatchValue)

        if not self._collection_exists():
            return 0
        try:
            before = self._client.count(
                collection_name=self._collection,
                count_filter=Filter(must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id))])).count
            self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="document_id",
                                   match=MatchValue(value=document_id))])))
            return int(before)
        except (UnexpectedResponse, httpx.HTTPError) as exc:
            raise _qdrant_error(exc) from exc
