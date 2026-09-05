"""Pydantic schemas for FastAPI — Hướng C: WHO-RAG Infrastructure API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ---------- RAG Query (Hướng C) ----------
class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi y tế của người dùng", examples=["Dấu hiệu đái tháo đường type 2 là gì?"])
    top_k: int = Field(5, ge=1, le=20, description="Số chunk truy vấn")
    tone: Optional[Literal["strict", "friendly", "balanced"]] = Field(None, description="Tone trả lời; auto nếu None")
    user_id: str = Field("default_user", description="ID người dùng cho memory")
    include_audit: bool = Field(True, description="Có trả về audit trail (citations, faithfulness) không")


class CitationItem(BaseModel):
    source_id: str
    dataset: Optional[str] = None
    section_path: Optional[List[str]] = None
    page_numbers: Optional[List[int]] = None
    score: Optional[float] = None
    content_snippet: str = Field(..., description="Đoạn trích 300 ký tự đầu")


class AuditTrail(BaseModel):
    cited_sources: List[int]
    citations: List[CitationItem]
    prompt_version: Optional[str] = None
    reranker_model: Optional[str] = None
    latency_ms: Optional[float] = None
    faithfulness_hint: Optional[str] = Field(None, description="Gợi ý faithfulness (nếu bật LLM-as-judge)")


class QueryResponse(BaseModel):
    answer: str
    cited_sources: List[int]
    contexts: List[Dict[str, Any]]
    audit: Optional[AuditTrail] = None
    langsmith: Optional[Dict[str, Any]] = None
    cache_hit: bool = False
    timings: Optional[Dict[str, float]] = None


# ---------- Glucose Tracking (Hướng A) ----------
class GlucoseLogCreate(BaseModel):
    user_id: str
    value_mgdl: float = Field(..., ge=20, le=800, description="Chỉ số đường huyết mg/dL")
    measured_at: Optional[datetime] = Field(None, description="Thời điểm đo; mặc định now")
    context: Literal["fasting", "pre_meal", "post_meal_2h", "bedtime", "random"] = Field("random")
    notes: Optional[str] = None


class GlucoseLogOut(BaseModel):
    id: int
    user_id: str
    value_mgdl: float
    measured_at: datetime
    context: str
    notes: Optional[str]
    classification: str  # normal / elevated / high / critical / low
    message: str


class GlucoseStats(BaseModel):
    user_id: str
    total_logs: int
    avg_mgdl: Optional[float]
    last_7_days_avg: Optional[float]
    streak_days: int
    logs_per_week: float
    classification_counts: Dict[str, int]


# ---------- SOAP Summary (Hướng B) ----------
class SoapGenerateRequest(BaseModel):
    user_id: str = Field(..., examples=["default_user"])
    days: int = Field(14, ge=1, le=90, description="Số ngày dữ liệu để tóm tắt")
    include_logs: bool = True
    language: Literal["vi", "en"] = "vi"


class SoapSection(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class SoapResponse(BaseModel):
    user_id: str
    generated_at: datetime
    period: str
    soap: SoapSection
    stats: GlucoseStats
    pdf_url: Optional[str] = None  # nếu có export


# ---------- Guidelines / Corpus ----------
class GuidelineStatus(BaseModel):
    total_pdfs: int
    by_source: Dict[str, int]
    manifest_path: str
    last_updated: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_model: str
    pdf_count: int
    vector_db_ready: bool
