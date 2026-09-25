"""Vitals service router — /v1/bp*, /v1/respiratory*, /v1/mood*, /v1/vitals."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

try:
    from src.shared.config import API_PREFIX
    from src.vitals.schemas import (
        BpLogCreate, BpLogOut, BpStats,
        RespiratoryLogCreate, RespiratoryLogOut, RespiratoryStats,
        MoodLogCreate, MoodLogOut, MoodStats,
        VitalsLogCreate,
    )
    from src.vitals.bp_tracker import add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp
    from src.vitals.respiratory_tracker import add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory
    from src.vitals.mood_tracker import add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood
    from src.diabetes.glucose_tracker import add_log
    from src.auth.dependencies import get_current_user, enforce_user_ownership
    AUTH_ENABLED = True
except ImportError:
    from shared.config import API_PREFIX  # type: ignore
    from vitals.schemas import (  # type: ignore
        BpLogCreate, BpLogOut, BpStats,
        RespiratoryLogCreate, RespiratoryLogOut, RespiratoryStats,
        MoodLogCreate, MoodLogOut, MoodStats,
        VitalsLogCreate,
    )
    from vitals.bp_tracker import add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp  # type: ignore
    from vitals.respiratory_tracker import add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory  # type: ignore
    from vitals.mood_tracker import add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood  # type: ignore
    from diabetes.glucose_tracker import add_log  # type: ignore
    from auth.dependencies import get_current_user, enforce_user_ownership  # type: ignore
    AUTH_ENABLED = True

try:
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from api.deps import _resolve_user_id  # type: ignore

router = APIRouter(prefix=API_PREFIX, tags=["vitals"])


@router.post("/bp", response_model=BpLogOut)
def create_bp_log(payload: BpLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_bp_log(user_id=user_id, systolic=payload.systolic, diastolic=payload.diastolic, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BpLogOut(id=rec["id"], user_id=rec["user_id"], systolic=rec["systolic"], diastolic=rec["diastolic"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"], classification=rec["classification"], message=rec["message"])

@router.get("/bp/{user_id}")
def list_bp_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_bp_logs(user_id, limit=limit, days=days)
    stats = get_bp_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": BpStats(**stats).model_dump(), "should_escalate": should_escalate_bp(user_id)}

@router.post("/respiratory", response_model=RespiratoryLogOut)
def create_respiratory_log(payload: RespiratoryLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_respiratory_log(user_id=user_id, peak_flow_percent=payload.peak_flow_percent, personal_best=payload.personal_best, cat_score=payload.cat_score, inhaler_correct=payload.inhaler_correct, inhaler_steps_correct=payload.inhaler_steps_correct, inhaler_steps_total=payload.inhaler_steps_total, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RespiratoryLogOut(id=rec["id"], user_id=rec["user_id"], peak_flow_percent=rec["peak_flow_percent"], gold_stage=rec["gold_stage"], classification=rec["classification"], message=rec["message"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"])

@router.get("/respiratory/{user_id}")
def list_respiratory_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_respiratory_logs(user_id, limit=limit, days=days)
    stats = get_respiratory_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": RespiratoryStats(**stats).model_dump(), "should_escalate": should_escalate_respiratory(user_id)}

@router.post("/mood", response_model=MoodLogOut)
def create_mood_log(payload: MoodLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_mood_log(user_id=user_id, phq9_score=payload.phq9_score, gad7_score=payload.gad7_score, mood_notes=payload.mood_notes, measured_at=payload.measured_at, context=payload.context)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return MoodLogOut(id=rec["id"], user_id=rec["user_id"], phq9_score=rec["phq9_score"], gad7_score=rec["gad7_score"], mood_notes=rec["mood_notes"], original_notes_had_pii=rec["original_notes_had_pii"], classification=rec["classification"], message=rec["message"], crisis_flag=rec["crisis_flag"], crisis_keywords=rec["crisis_keywords"], measured_at=rec["measured_at"], context=rec["context"])

@router.get("/mood/{user_id}")
def list_mood_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_mood_logs(user_id, limit=limit, days=days)
    stats = get_mood_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": MoodStats(**stats).model_dump(), "should_escalate": should_escalate_mood(user_id)}

@router.post("/vitals")
def create_vitals(payload: VitalsLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        if payload.disease == "diabetes":
            if payload.value_mgdl is None:
                raise ValueError("value_mgdl required for diabetes")
            rec = add_log(user_id=user_id, value_mgdl=payload.value_mgdl, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
            return {"disease": "diabetes", "result": rec}
        elif payload.disease == "hypertension":
            if payload.systolic is None or payload.diastolic is None:
                raise ValueError("systolic/diastolic required for hypertension")
            rec = add_bp_log(user_id=user_id, systolic=payload.systolic, diastolic=payload.diastolic, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
            return {"disease": "hypertension", "result": rec}
        elif payload.disease == "respiratory":
            rec = add_respiratory_log(user_id=user_id, peak_flow_percent=payload.peak_flow_percent, cat_score=payload.cat_score, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
            return {"disease": "respiratory", "result": rec}
        elif payload.disease == "mental":
            rec = add_mood_log(user_id=user_id, phq9_score=payload.phq9_score, gad7_score=payload.gad7_score, mood_notes=payload.notes, measured_at=payload.measured_at, context=payload.context)
            return {"disease": "mental", "result": rec}
        else:
            raise ValueError(f"Unknown disease {payload.disease}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
