from __future__ import annotations

from pydantic import BaseModel, Field


class LlmOverride(BaseModel):
    """Token y modelo elegidos en la web. Solo vive en la peticion:
    nunca se guarda en disco ni en logs."""
    provider: str = "openai"  # openai | openrouter | custom
    base_url: str | None = None
    model: str
    api_key: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 5
    llm: LlmOverride | None = None


class SourceOut(BaseModel):
    document_name: str
    page: int
    kind: str
    image_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    image_id: str | None = None
    image_page: int | None = None
    image_url: str | None = None


class JobOut(BaseModel):
    job_id: str
    document_name: str
    status: str
    progress: float
    message: str
    chunks_count: int = 0
    images_count: int = 0
    error: str | None = None


class DetectRequest(BaseModel):
    api_key: str = Field(min_length=1)
    base_url: str | None = None


class DetectResponse(BaseModel):
    detected: bool
    provider: str | None = None
    base_url: str | None = None
    models: list[str] = []
    message: str
