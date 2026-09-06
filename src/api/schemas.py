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


# ---------- Hypertension — BP Tracking (2B.1, larger than diabetes) ----------
class BpLogCreate(BaseModel):
    user_id: str
    systolic: int = Field(..., ge=50, le=300, description="Systolic mmHg")
    diastolic: int = Field(..., ge=30, le=200, description="Diastolic mmHg")
    measured_at: Optional[datetime] = Field(None, description="Time; default now")
    context: Literal["morning", "evening", "random", "post_exercise", "clinic"] = Field("random")
    notes: Optional[str] = None

class BpLogOut(BaseModel):
    id: int
    user_id: str
    systolic: int
    diastolic: int
    measured_at: datetime
    context: str
    notes: Optional[str]
    classification: str  # normal / elevated / stage1 / stage2 / crisis
    message: str

class BpStats(BaseModel):
    user_id: str
    total_logs: int
    avg_sys: Optional[float]
    avg_dia: Optional[float]
    last_7_days_avg: Optional[Dict[str, float]] = None
    streak_days: int
    logs_per_week: float
    classification_counts: Dict[str, int]
    at_target_rate: float

# ---------- Respiratory — Asthma/COPD (2B.2, inhaler technique) ----------
class RespiratoryLogCreate(BaseModel):
    user_id: str
    peak_flow_percent: Optional[int] = Field(None, ge=10, le=150, description="% personal best")
    personal_best: Optional[int] = Field(None, ge=50, le=1000)
    cat_score: Optional[int] = Field(None, ge=0, le=40, description="COPD Assessment Test 0-40")
    inhaler_correct: Optional[bool] = None
    inhaler_steps_correct: Optional[int] = Field(None, ge=0, le=20)
    inhaler_steps_total: Optional[int] = Field(None, ge=1, le=20)
    measured_at: Optional[datetime] = None
    context: Literal["morning", "evening", "post_exercise", "random", "clinic"] = Field("random")
    notes: Optional[str] = None

class RespiratoryLogOut(BaseModel):
    id: int
    user_id: str
    peak_flow_percent: Optional[int]
    gold_stage: Optional[str]
    classification: str  # green / yellow / red
    message: str
    measured_at: datetime
    context: str
    notes: Optional[str]

class RespiratoryStats(BaseModel):
    user_id: str
    total_logs: int
    avg_peak_flow: Optional[float]
    red_rate: float
    incorrect_inhaler_rate: float
    classification_counts: Dict[str, int]

# ---------- Mental Health — PHQ-9/GAD-7 + crisis + PII redact (2B.3) ----------
class MoodLogCreate(BaseModel):
    user_id: str
    phq9_score: Optional[int] = Field(None, ge=0, le=27, description="PHQ-9 0-27")
    gad7_score: Optional[int] = Field(None, ge=0, le=21, description="GAD-7 0-21")
    mood_notes: Optional[str] = Field(None, description="Free text, will be PII-redacted before storage")
    measured_at: Optional[datetime] = None
    context: Literal["random", "morning", "evening", "clinic"] = Field("random")

class MoodLogOut(BaseModel):
    id: int
    user_id: str
    phq9_score: Optional[int]
    gad7_score: Optional[int]
    mood_notes: Optional[str]  # redacted
    original_notes_had_pii: Optional[bool] = None
    classification: str  # minimal / mild / moderate / severe / crisis
    message: str
    crisis_flag: bool
    crisis_keywords: List[str] = Field(default_factory=list)
    measured_at: datetime
    context: str

class MoodStats(BaseModel):
    user_id: str
    total_logs: int
    avg_phq9: Optional[float]
    avg_gad7: Optional[float]
    crisis_count: int
    classification_counts: Dict[str, int]

# ---------- Unified Vitals (optional) ----------
class VitalsLogCreate(BaseModel):
    user_id: str
    disease: Literal["diabetes", "hypertension", "respiratory", "mental"] = Field(..., description="Disease type")
    # diabetes
    value_mgdl: Optional[float] = None
    # hypertension
    systolic: Optional[int] = None
    diastolic: Optional[int] = None
    # respiratory
    peak_flow_percent: Optional[int] = None
    cat_score: Optional[int] = None
    # mental
    phq9_score: Optional[int] = None
    gad7_score: Optional[int] = None
    measured_at: Optional[datetime] = None
    context: str = Field("random")
    notes: Optional[str] = None

# ---------- SOAP Summary (Hướng B) ----------
class SoapGenerateRequest(BaseModel):
    user_id: str = Field(..., examples=["default_user"])
    days: int = Field(14, ge=1, le=90, description="Số ngày dữ liệu để tóm tắt")
    include_logs: bool = True
    language: Literal["vi", "en"] = "vi"
    disease: Literal["diabetes", "hypertension", "respiratory", "mental", "all"] = Field("diabetes", description="Disease for SOAP context")


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
    stats: Dict[str, Any]  # GlucoseStats | BpStats | RespiratoryStats | MoodStats (generic)
    pdf_url: Optional[str] = None  # nếu có export
    disease: str = "diabetes"


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
