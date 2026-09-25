"""Labs Q&A schemas — POST /v1/labs/ask (Nura B2).

LLM-proof note (D2 pin): parse output schema (_ParsedQuestion trong router)
KHONG bao gio co key panic/emergency/override — panic chi duoc quyet boi
red_flag (text) + lab_thresholds.is_panic (numeric), o muc call truoc moi
NLU/RAG/LLM. Pydantic schemas o day cung khong co key override nao.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class LabAskRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional; anonymous chi doc duoc khi kem user_id/report_id")
    report_id: Optional[str] = Field(None, description="Phieu cu the; bo trong -> lay phieu moi nhat cua user")
    question: str = Field(..., min_length=1, description="Cau hoi ve phieu XN (VI, co/khong dau)")


class LabCitation(BaseModel):
    report_id: str
    loinc: str
    source: str = Field(..., description="Nguon guideline (VD: BYT QD5481/ADA 2024)")


class LabAnswerSegment(BaseModel):
    text: str
    citations: List[LabCitation] = Field(default_factory=list)


class LabAskResponse(BaseModel):
    panic: bool = False
    emergency: bool = False
    red_flag_type: Optional[str] = None
    message: Optional[str] = None
    report_id: Optional[str] = None
    user_id: Optional[str] = None
    answer_segments: List[LabAnswerSegment] = Field(default_factory=list)
    handoff_tier: Optional[str] = Field(None, description="T0 auto / T1 admin / T2 pharmacist|specialist / T3 doctor (B3 routing; panic luon T3)")
    review_id: Optional[str] = Field(None, description="review_requests id khi T1-T3 (None khi T0)")
    what_happens_next: Optional[str] = Field(None, description="Giai thich VI buoc tiep theo cho benh nhan")
