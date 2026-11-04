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


def test_embedding_uses_default_token_when_env_key_missing(tmp_path):
    """Autorizado por el dueno: sin OPENAI_API_KEY en el entorno, el
    token predeterminado cifrado del servidor sirve para embeddings
    reales, sin exponer su valor en logs ni respuestas."""
    from cryptography.fernet import Fernet

    from app.adapters.token_vault import TokenVault
    from app.config import Settings
    from app.dependencies import Container

    key = Fernet.generate_key().decode()
    vault = TokenVault(store_dir=str(tmp_path / "tokens"),
                       encryption_key=key,
                       default_path=str(tmp_path / "default_token.enc"))
    vault.save_default({"api_key": "fake-default-key", "provider": "openai",
                        "base_url": None, "model": "gpt-4o-mini",
                        "auto_model": False})
    settings = Settings(
        storage_dir=str(tmp_path), embedding_provider="openai",
        openai_api_key="", token_encryption_key=key,
        vector_store="memory", queue_backend="memory", demo_mode=True)
    container = Container(settings)
    assert isinstance(container.embedding, OpenAICompatibleEmbedding)
    assert container.embedding.api_key == "fake-default-key"


def test_hash_embedding_is_not_openai():
    assert not isinstance(HashEmbedding(), OpenAICompatibleEmbedding)
    assert isinstance(
        OpenAICompatibleEmbedding(
            api_key="k", base_url="https://example.invalid/v1",
            model="text-embedding-3-small", dimension=384),
        OpenAICompatibleEmbedding)
