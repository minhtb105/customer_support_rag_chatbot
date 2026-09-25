"""Diabetes service router — /v1/glucose*, /v1/soap/*."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

try:
    from src.shared.config import API_PREFIX
    from src.diabetes.schemas import (
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        SoapGenerateRequest, SoapResponse, SoapSection,
    )
    from src.diabetes.glucose_tracker import add_log, get_logs, get_stats, should_escalate_to_doctor
    from src.diabetes.soap_summary import generate_soap, soap_to_markdown
    from src.auth.dependencies import get_current_user, enforce_user_ownership
    AUTH_ENABLED = True
except ImportError:
    from shared.config import API_PREFIX  # type: ignore
    from diabetes.schemas import (  # type: ignore
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        SoapGenerateRequest, SoapResponse, SoapSection,
    )
    from diabetes.glucose_tracker import add_log, get_logs, get_stats, should_escalate_to_doctor  # type: ignore
    from diabetes.soap_summary import generate_soap, soap_to_markdown  # type: ignore
    from auth.dependencies import get_current_user, enforce_user_ownership  # type: ignore
    AUTH_ENABLED = True

try:
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from api.deps import _resolve_user_id  # type: ignore

router = APIRouter(prefix=API_PREFIX, tags=["diabetes"])


@router.post("/glucose", response_model=GlucoseLogOut)
def create_glucose_log(payload: GlucoseLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_log(user_id=user_id, value_mgdl=payload.value_mgdl, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    # Anomaly middleware (fail-open: never break the primary log write)
    _anomaly: dict = {"type": "none", "reason": ""}
    _fqg: list = []
    try:
        try:
            from src.diabetes.anomaly_detector import analyze_glucose_log
        except ImportError:
            from diabetes.anomaly_detector import analyze_glucose_log  # type: ignore
        _recent = get_logs(user_id, limit=30)
        _res = analyze_glucose_log(payload.value_mgdl, _recent)
        _anomaly = _res.get("anomaly", _anomaly)
        _fqg = _res.get("follow_up_questions", [])
    except Exception:
        pass
    return GlucoseLogOut(id=rec["id"], user_id=rec["user_id"], value_mgdl=rec["value_mgdl"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"], classification=rec["classification"], message=rec["message"], anomaly=_anomaly, follow_up_questions=_fqg)

@router.get("/glucose/{user_id}")
def list_glucose_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_logs(user_id, limit=limit, days=days)
    stats = get_stats(user_id)
    escalate = should_escalate_to_doctor(user_id)
    # Server-side anomaly ids from the SAME fetched logs (no second DB query).
    _anomaly_ids: list = []
    try:
        try:
            from src.diabetes.anomaly_detector import SPIKE_HIGH, SPIKE_LOW
        except ImportError:
            from diabetes.anomaly_detector import SPIKE_HIGH, SPIKE_LOW  # type: ignore
        for _l in logs:
            try:
                _v = float(_l.get("value_mgdl"))
            except (TypeError, ValueError):
                continue
            if _v > SPIKE_HIGH or _v < SPIKE_LOW:
                try:
                    _anomaly_ids.append(int(_l.get("id")))
                except (TypeError, ValueError):
                    pass
        # Trend window: regroup fasting by day (same logs), find 10-15%/day x3 cascade.
        _by_day: dict = {}
        _day_ids: dict = {}
        for _l in logs:
            if (_l.get("context") or "") != "fasting":
                continue
            try:
                _vv = float(_l.get("value_mgdl"))
                _day = str(_l.get("measured_at"))[:10]
            except (TypeError, ValueError):
                continue
            _by_day.setdefault(_day, []).append(_vv)
            try:
                _day_ids.setdefault(_day, []).append(int(_l.get("id")))
            except (TypeError, ValueError):
                pass
        _ds = sorted(_by_day.keys())
        if len(_ds) >= 3:
            from datetime import datetime as _dt
            try:
                _ord = {d: _dt.fromisoformat(d).date().toordinal() for d in _ds}
            except Exception:
                _ord = {}
            if _ord:
                _avg = {d: sum(v) / len(v) for d, v in _by_day.items()}
                for _i in range(len(_ds) - 2):
                    _d0, _d1, _d2 = _ds[_i], _ds[_i + 1], _ds[_i + 2]
                    if _ord[_d1] != _ord[_d0] + 1 or _ord[_d2] != _ord[_d1] + 1:
                        continue
                    _a0, _a1, _a2 = _avg[_d0], _avg[_d1], _avg[_d2]
                    if not _a0 or not _a1:
                        continue
                    _r1, _r2 = (_a1 - _a0) / _a0, (_a2 - _a1) / _a1
                    if 0.10 <= _r1 <= 0.15 and 0.10 <= _r2 <= 0.15:
                        for _d in (_d0, _d1, _d2):
                            for _lid in _day_ids.get(_d, []):
                                if _lid not in _anomaly_ids:
                                    _anomaly_ids.append(_lid)
                        break
    except Exception:
        _anomaly_ids = []
    return {"user_id": user_id, "logs": logs, "stats": GlucoseStats(**stats).model_dump(), "should_escalate": escalate, "anomaly_ids": _anomaly_ids}

@router.post("/glucose/followup")
def create_glucose_followup(payload: dict, current_user=Depends(get_current_user)):
    """Persist an FQG answer: update notes of related/latest log + episodic buffer (fail-open)."""
    req_uid = (payload or {}).get("user_id", "")
    text = str((payload or {}).get("text", "") or "").strip()
    related_log_id = (payload or {}).get("related_log_id")
    user_id = req_uid
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(req_uid, current_user)
    if not text:
        raise HTTPException(status_code=422, detail="text must be non-empty")
    try:
        try:
            from src.diabetes.followup_notes import save_followup_notes
        except ImportError:
            from diabetes.followup_notes import save_followup_notes  # type: ignore
        saved = save_followup_notes(user_id, text, related_log_id)
    except Exception:
        saved = {"saved_note": False, "saved_episodic": False, "related_log_id": related_log_id}
    return {"saved": True, "saved_note": saved["saved_note"], "saved_episodic": saved["saved_episodic"], "related_log_id": saved["related_log_id"]}

@router.post("/soap/generate", response_model=SoapResponse)
def generate_soap_endpoint(req: SoapGenerateRequest, current_user=Depends(get_current_user)):
    user_id = req.user_id
    if AUTH_ENABLED and current_user:
        # user can only generate for self, expert/admin for any
        if current_user["role"] == "user" and req.user_id != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        user_id = req.user_id or current_user["id"]
    try:
        data = generate_soap(user_id, days=req.days, language=req.language, disease=req.disease)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    soap = data["soap"]
    return SoapResponse(user_id=data["user_id"], generated_at=data["generated_at"], period=data["period"], soap=SoapSection(**soap), stats=data["stats"], disease=req.disease)

@router.post("/soap/generate/markdown", response_class=PlainTextResponse)
def generate_soap_markdown(req: SoapGenerateRequest, current_user=Depends(get_current_user)):
    user_id = req.user_id
    if AUTH_ENABLED and current_user:
        if current_user["role"] == "user" and req.user_id != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        user_id = req.user_id or current_user["id"]
    data = generate_soap(user_id, days=req.days, language=req.language, disease=req.disease)
    md = soap_to_markdown(data)
    return md
