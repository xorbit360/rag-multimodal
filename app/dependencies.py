"""Wiring: aqui se eligen los adaptadores segun .env. El core no cambia."""
from __future__ import annotations

import os

from app.adapters.embeddings import HashEmbedding, OpenAICompatibleEmbedding
from app.adapters.llm import FakeLLM, OpenAICompatibleLLM
from app.adapters.pdf_parser import PyMuPDFParser
from app.adapters.queue import InMemoryJobQueue, InMemoryJobStore, RedisJobQueue
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
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            self.embedding = OpenAICompatibleEmbedding(
                api_key=settings.openai_api_key, base_url=settings.openai_base_url,
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

        self.jobs = InMemoryJobStore()
        self.queue = (RedisJobQueue(settings.redis_url) if settings.queue_backend == "redis"
                      else InMemoryJobQueue())
        self.ingest = IngestUseCase(self.parser, self.embedding, self.store,
                                    self.queue, self.jobs)
        self.ask = AskUseCase(self.llm, self.embedding, self.store)
