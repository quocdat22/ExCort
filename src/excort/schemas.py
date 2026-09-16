"""Validated HTTP request and response shapes for the ExCort API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """One user question and an optional retrieval depth override."""

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    """A document passage used as evidence for an answer."""

    chunk_id: str
    source: str
    page_number: int
    similarity: float
    text: str


class ChatResponse(BaseModel):
    """Generated answer accompanied by its retrieved evidence."""

    question: str
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """Readiness information for the local vector index."""

    status: Literal["ok"]
    collection: str
    record_count: int = Field(ge=0)
