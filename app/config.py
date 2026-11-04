"""Configuracion por entorno (pydantic-settings). Sin secretos en el codigo."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 384
    vector_store: str = "memory"  # memory | qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    queue_backend: str = "memory"  # memory | redis
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "storage"
    demo_mode: bool = True
    log_level: str = "INFO"
    # Guardado de tokens en el servidor (cifrado Fernet). Sin clave,
    # guardar y usar tokens guardados fallan con mensaje claro: nunca
    # se guarda nada en plano. La clave nunca va en Git (.env.example
    # la trae vacia) ni se devuelve en ninguna respuesta.
    token_encryption_key: str = ""
    # Opcionales: si no se fijan, cuelgan de storage_dir.
    token_store_dir: str | None = None
    default_token_path: str | None = None

    @property
    def use_demo_llm(self) -> bool:
        return self.demo_mode or not self.openai_api_key


def get_settings() -> Settings:
    return Settings()
