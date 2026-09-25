"""Monitors service schemas — guideline corpus status."""
from __future__ import annotations

from typing import Optional, Dict
from pydantic import BaseModel


class GuidelineStatus(BaseModel):
    total_pdfs: int
    by_source: Dict[str, int]
    manifest_path: str
    last_updated: Optional[str] = None
