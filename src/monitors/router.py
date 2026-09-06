"""FastAPI router for /v1/monitors — Guideline & Safety agents"""
from __future__ import annotations
from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

try:
    from src.auth.dependencies import get_current_user, require_admin, require_expert
    from src.monitors.db import (
        list_guideline_versions, get_guideline_version,
        list_safety_alerts, get_safety_alert,
        list_sources, get_source, list_runs, get_stats,
        create_run, finish_run,
    )
    from src.monitors.guideline_fetcher import check_guideline_update, check_all_guidelines
    from src.monitors.safety_fetcher import check_all_safety, check_fda_alerts, fetch_byt_dav_alerts
    from src.monitors.service import promote_guideline_to_corpus, decide_safety_alert, cleanup_superseded
    from src.config import GUIDELINE_SOURCE_DIRS
except ImportError:
    from auth.dependencies import get_current_user, require_admin, require_expert  # type: ignore
    from monitors.db import list_guideline_versions, get_guideline_version, list_safety_alerts, get_safety_alert, list_sources, get_source, list_runs, get_stats, create_run, finish_run  # type: ignore
    from monitors.guideline_fetcher import check_guideline_update, check_all_guidelines  # type: ignore
    from monitors.safety_fetcher import check_all_safety, check_fda_alerts, fetch_byt_dav_alerts  # type: ignore
    from monitors.service import promote_guideline_to_corpus, decide_safety_alert, cleanup_superseded  # type: ignore
    from config import GUIDELINE_SOURCE_DIRS  # type: ignore

router = APIRouter(prefix="/v1/monitors", tags=["monitors"])

# ---------- helpers ----------
def _ensure_guideline_roles(user: dict):
    if user["role"] not in ("specialist", "doctor", "admin"):
        raise HTTPException(status_code=403, detail="Requires specialist/doctor/admin for guideline decisions")

def _ensure_safety_roles(user: dict):
    if user["role"] not in ("pharmacist", "admin"):
        raise HTTPException(status_code=403, detail="Requires pharmacist/admin for safety decisions")

# ---------- status ----------
@router.get("/status")
def monitors_status(current_user=Depends(get_current_user)):
    sources = list_sources()
    stats = get_stats()
    runs = list_runs(limit=10)
    return {
        "stats": stats,
        "sources": sources,
        "recent_runs": runs,
    }

@router.get("/sources")
def list_monitor_sources(current_user=Depends(get_current_user)):
    return {"sources": list_sources()}

@router.get("/runs")
def list_monitor_runs(
    source_key: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    return {"runs": list_runs(source_key=source_key, limit=limit)}

# ---------- trigger checks ----------
@router.post("/check/guidelines")
def trigger_guideline_check(
    source_key: Optional[str] = Query(None, description="Specific source or all if empty"),
    force: bool = Query(False, description="Force download even if etag unchanged"),
    current_user=Depends(get_current_user),
):
    # Allow specialist/doctor/admin to trigger guideline checks, pharmacist/admin for safety but guideline needs specialist
    if current_user["role"] not in ("specialist", "doctor", "admin"):
        raise HTTPException(status_code=403, detail="Requires specialist/doctor/admin")
    if source_key:
        run_id = create_run(source_key)
        try:
            res = check_guideline_update(source_key, force_download=force)
            found = 1 if res.get("found_new") else 0
            finish_run(run_id, found_new=found, status="success" if "error" not in res else "failed", error=res.get("error"))
            return {"source_key": source_key, "result": res, "run_id": run_id}
        except Exception as e:
            finish_run(run_id, found_new=0, status="failed", error=str(e)[:500])
            raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        # all
        results = []
        total_found = 0
        for sk in ["ada_soc", "gold", "gina", "who_mhgap", "who_diabetes", "aha_acc_htn", "byt_diabetes"]:
            rid = create_run(sk)
            try:
                res = check_guideline_update(sk, force_download=force)
                found = 1 if res.get("found_new") else 0
                total_found += found
                finish_run(rid, found_new=found, status="success" if "error" not in res else "failed", error=res.get("error"))
                results.append(res)
            except Exception as e:
                finish_run(rid, status="failed", error=str(e)[:500])
                results.append({"source_key": sk, "error": str(e)[:500]})
        return {"results": results, "total_found": total_found}

@router.post("/check/safety")
def trigger_safety_check(
    source: Optional[Literal["fda", "byt", "all"]] = Query("all"),
    current_user=Depends(get_current_user),
):
    if current_user["role"] not in ("pharmacist", "admin"):
        raise HTTPException(status_code=403, detail="Requires pharmacist/admin")
    if source == "fda":
        run_id = create_run("fda_recall")
        try:
            res = check_fda_alerts()
            finish_run(run_id, found_new=res.get("found_new", 0), status="success")
            return res
        except Exception as e:
            finish_run(run_id, status="failed", error=str(e)[:500])
            raise HTTPException(status_code=500, detail=str(e)) from e
    elif source == "byt":
        run_id = create_run("dav_thuhoi")
        try:
            res = fetch_byt_dav_alerts()
            finish_run(run_id, found_new=res.get("found_new", 0), status="success")
            return res
        except Exception as e:
            finish_run(run_id, status="failed", error=str(e)[:500])
            raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        run_id = create_run("safety_all")
        try:
            res = check_all_safety()
            finish_run(run_id, found_new=res.get("total_found", 0), status="success")
            return res
        except Exception as e:
            finish_run(run_id, status="failed", error=str(e)[:500])
            raise HTTPException(status_code=500, detail=str(e)) from e

# ---------- guideline versions ----------
@router.get("/guidelines")
def list_guidelines(
    status: Optional[Literal["pending_review", "approved", "rejected", "superseded", "archived"]] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
):
    rows, total = list_guideline_versions(status=status, source=source, limit=limit, offset=offset)
    return {"total": total, "items": rows, "limit": limit, "offset": offset}

@router.get("/guidelines/{gid}")
def get_guideline(gid: str, current_user=Depends(get_current_user)):
    gv = get_guideline_version(gid)
    if not gv:
        raise HTTPException(status_code=404, detail="Guideline version not found")
    return gv

class GuidelineDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    notes: Optional[str] = None

@router.post("/guidelines/{gid}/decision")
def decide_guideline(gid: str, payload: GuidelineDecision, current_user=Depends(get_current_user)):
    _ensure_guideline_roles(current_user)
    try:
        result = promote_guideline_to_corpus(gid, reviewer_id=current_user["id"], approve=(payload.decision == "approved"), review_notes=payload.notes)
        return {"id": gid, "decision": payload.decision, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# ---------- safety alerts ----------
@router.get("/alerts")
def list_alerts(
    status: Optional[Literal["pending_review", "approved", "dismissed", "archived"]] = Query(None),
    severity: Optional[Literal["critical", "high", "medium", "low"]] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
):
    rows, total = list_safety_alerts(status=status, severity=severity, source=source, limit=limit, offset=offset)
    return {"total": total, "items": rows, "limit": limit, "offset": offset}

@router.get("/alerts/{aid}")
def get_alert(aid: str, current_user=Depends(get_current_user)):
    alert = get_safety_alert(aid)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

class SafetyDecision(BaseModel):
    decision: Literal["approved", "dismissed"]
    notes: Optional[str] = None

@router.post("/alerts/{aid}/decision")
def decide_alert(aid: str, payload: SafetyDecision, current_user=Depends(get_current_user)):
    _ensure_safety_roles(current_user)
    try:
        result = decide_safety_alert(aid, reviewer_id=current_user["id"], decision=payload.decision, review_notes=payload.notes)
        return {"id": aid, "decision": payload.decision, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# ---------- maintenance ----------
@router.post("/maintenance/cleanup-superseded")
def cleanup_superseded_endpoint(current_user=Depends(require_admin)):
    deleted = cleanup_superseded()
    return {"deleted": deleted}

@router.get("/stats")
def monitors_stats(current_user=Depends(get_current_user)):
    return get_stats()
