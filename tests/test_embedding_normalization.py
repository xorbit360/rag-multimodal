"""Embeddings reales: normalizar a la dimension configurada y etiquetar
el adaptador real, no solo el ajuste.

Si el proveedor es openai pero no hay llave, Container cae a hash y
/data/overview debe decir hash, no openai. Si el modelo devuelve mas
dimensiones que la coleccion (1536 frente a 384), embed() debe
truncar y renormalizar, no tumbar la ingesta en Qdrant."""

import math
from unittest.mock import patch

from app.adapters.embeddings import HashEmbedding, OpenAICompatibleEmbedding


class _Resp:
    def __init__(self, vectors):
        self._vectors = vectors

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": v} for v in self._vectors]}


def test_openai_embedding_truncates_to_configured_dimension():
    emb = OpenAICompatibleEmbedding(
        api_key="k", base_url="https://example.invalid/v1",
        model="text-embedding-3-small", dimension=384)
    big = [0.01] * 1536
    with patch("app.adapters.embeddings.httpx.post",
               return_value=_Resp([big])):
        out = emb.embed(["texto de prueba"])
    assert len(out[0]) == 384
    norm = math.sqrt(sum(x * x for x in out[0]))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedding_is_not_openai():
    assert not isinstance(HashEmbedding(), OpenAICompatibleEmbedding)
    assert isinstance(
        OpenAICompatibleEmbedding(
            api_key="k", base_url="https://example.invalid/v1",
            model="text-embedding-3-small", dimension=384),
        OpenAICompatibleEmbedding)
