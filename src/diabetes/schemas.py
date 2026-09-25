"""Diabetes service schemas — glucose tracking + SOAP."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


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
    anomaly: Optional[Dict[str, Any]] = Field(None, description="Anomaly middleware: {type: spike|trend|none, ...}")
    follow_up_questions: List[str] = Field(default_factory=list, description="FQG context questions")


class GlucoseStats(BaseModel):
    user_id: str
    total_logs: int
    avg_mgdl: Optional[float]
    last_7_days_avg: Optional[float]
    streak_days: int
    logs_per_week: float
    classification_counts: Dict[str, int]


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
