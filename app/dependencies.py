"""Wiring: aqui se eligen los adaptadores segun .env. El core no cambia."""
from __future__ import annotations

import os

from app.adapters.embeddings import HashEmbedding, OpenAICompatibleEmbedding
from app.adapters.llm import FakeLLM, OpenAICompatibleLLM
from app.adapters.pdf_parser import PyMuPDFParser
from app.adapters.queue import (
    InMemoryJobQueue, InMemoryJobStore, RedisJobQueue, RedisJobStore,
)
from app.adapters.token_vault import TokenVault
from app.adapters.vector_store import InMemoryVectorStore, QdrantVectorStore
from app.config import Settings
from app.domain.use_cases import AskUseCase, IngestUseCase


class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
        storage = settings.storage_dir
        self.images_dir = os.path.join(storage, "images")
        self.uploads_dir = os.path.join(storage, "uploads")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

        self.parser = PyMuPDFParser(images_dir=self.images_dir)
        # Boveda de tokens: se crea ANTES de los embeddings, porque el
        # token predeterminado del servidor (autorizado por el dueno)
        # puede ser la llave para embeddings reales cuando el entorno
        # no trae OPENAI_API_KEY. La construccion nunca falla por falta
        # de clave; el fallo claro ocurre al guardar o usar.
        token_store = (settings.token_store_dir
                       or os.path.join(storage, "tokens"))
        default_path = (settings.default_token_path
                        or os.path.join(storage, "default_token.enc"))
        self.token_vault = TokenVault(
            store_dir=token_store,
            encryption_key=settings.token_encryption_key,
            default_path=default_path)
        embedding_key = settings.openai_api_key
        embedding_base_url = settings.openai_base_url
        if settings.embedding_provider == "openai" and not embedding_key:
            # Solo si el dueno autorizo un token predeterminado en el
            # servidor: se usa su llave y su base_url (si la trae) para
            # embeddings, nunca se registra ni se devuelve su valor.
            try:
                default_cfg = self.token_vault.load_default()
            except Exception:
                default_cfg = None
            if default_cfg and default_cfg.get("api_key"):
                embedding_key = default_cfg["api_key"]
                if default_cfg.get("base_url"):
                    embedding_base_url = default_cfg["base_url"]
        if settings.embedding_provider == "openai" and embedding_key:
            self.embedding = OpenAICompatibleEmbedding(
                api_key=embedding_key, base_url=embedding_base_url,
                model=settings.embedding_model, dimension=settings.embedding_dimension)
        else:
            self.embedding = HashEmbedding(dimension=settings.embedding_dimension)

        if settings.vector_store == "qdrant":
            self.store = QdrantVectorStore(
                url=settings.qdrant_url, collection=settings.qdrant_collection,
                dimension=settings.embedding_dimension)
        else:
            self.store = InMemoryVectorStore()

        if settings.use_demo_llm:
            self.llm = FakeLLM()
        else:
            self.llm = OpenAICompatibleLLM(
                api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                model=settings.openai_model)

        # Senior: el JobStore acompana a la cola. Con Redis, API y
        # workers comparten el mismo estado persistente; con memoria,
        # el estado vive solo en el proceso (modo local / tests).
        if settings.queue_backend == "redis":
            self.jobs = RedisJobStore(settings.redis_url)
            self.queue = RedisJobQueue(settings.redis_url)
        else:
            self.jobs = InMemoryJobStore()
            self.queue = InMemoryJobQueue()
        self.ingest = IngestUseCase(self.parser, self.embedding, self.store,
                                    self.queue, self.jobs)
        self.ask = AskUseCase(self.llm, self.embedding, self.store)
