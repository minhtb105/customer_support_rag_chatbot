"""Pydantic schemas for the API composition root (per-service schemas live in src/<service>/schemas.py)."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_model: str
    pdf_count: int
    vector_db_ready: bool
