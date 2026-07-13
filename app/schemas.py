"""Pydantic request/response models for the chatbot API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's question")
    session_id: str = Field("default", description="Conversation/session id")
    institution: str | None = Field(
        None, description="Override institution: SRKI or SU (defaults to server setting)"
    )


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    confidence: float | None = None
    institution: str
    source: str
    in_domain: bool = True
    sources: list[Source] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    active_institution: str
    intent_backend: str
    generator_ready: bool
    llm_brain: bool = False
    web_cache_pages: int
    labels: list[str]
