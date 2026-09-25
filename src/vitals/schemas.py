"""Vitals service schemas — BP, respiratory, mental, unified vitals."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal, Dict, List
from pydantic import BaseModel, Field


# ---------- Hypertension — BP Tracking ----------
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

# ---------- Respiratory — Asthma/COPD (inhaler technique) ----------
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

# ---------- Mental Health — PHQ-9/GAD-7 + crisis + PII redact ----------
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

# ---------- Unified Vitals ----------
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
