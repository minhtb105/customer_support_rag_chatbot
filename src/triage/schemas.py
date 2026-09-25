"""Triage service schemas — smart triage, doctor queue, triage events."""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional; anonymous when omitted (response-only, zero DB writes)")
    message: str = Field(..., description="Tin nhắn tự nhiên của bệnh nhân (VI, có/không dấu)")


class TriageResponse(BaseModel):
    emergency: bool
    red_flag_type: Optional[str] = None
    message: Optional[str] = None
    urgency: Optional[str] = None
    symptoms: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_specialty: Optional[str] = None
    recommended_doctors: List[Dict[str, Any]] = Field(default_factory=list)
    previsit_notes: Optional[str] = None
    followup_question: Optional[str] = None
    # Dynamic Solver Routing (backward-compatible: all Optional, old fields intact)
    solver_used: Optional[str] = Field(None, description="greedy | dfs | ga")
    routing_reason: Optional[str] = Field(None, description="Why this solver (VI, echoes constraints)")
    simulation_summary: Optional[Dict[str, Any]] = Field(None, description="GA bulk-reschedule summary")


class TriageEventOut(BaseModel):
    ts: Optional[str] = None
    user_id: Optional[str] = None
    excerpt: Optional[str] = None
    emergency: bool = False
    red_flag_type: Optional[str] = None
    specialty: Optional[str] = None
    solver_used: Optional[str] = None


class TriageEventListOut(BaseModel):
    events: List[TriageEventOut] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    limit: int = 20


class DoctorPatientOut(BaseModel):
    user_id: str
    username: Optional[str] = None
    last_value: Optional[float] = None
    last_classification: Optional[str] = None
    last_measured_at: Optional[str] = None
    level: Optional[str] = None
    should_escalate: bool = False
    anomaly_count: int = 0


class DoctorPatientListOut(BaseModel):
    patients: List[DoctorPatientOut] = Field(default_factory=list)
    total: Optional[int] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
