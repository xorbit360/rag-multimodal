from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 5


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
