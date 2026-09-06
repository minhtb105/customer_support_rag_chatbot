"""
FastAPI — WHO-RAG Infrastructure API (Hướng C) — with Auth + HILT
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

try:
    from src.config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, PDF_DIR, BASE_DIR
    from src.api.schemas import (
        QueryRequest, QueryResponse, CitationItem, AuditTrail,
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        BpLogCreate, BpLogOut, BpStats,
        RespiratoryLogCreate, RespiratoryLogOut, RespiratoryStats,
        MoodLogCreate, MoodLogOut, MoodStats,
        VitalsLogCreate,
        SoapGenerateRequest, SoapResponse, SoapSection,
        GuidelineStatus, HealthResponse,
    )
    from src.features.glucose_tracker import add_log, get_logs, get_stats, classify_glucose, should_escalate_to_doctor
    from src.features.bp_tracker import add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp
    from src.features.respiratory_tracker import add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory
    from src.features.mood_tracker import add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood
    from src.features.soap_summary import generate_soap, soap_to_markdown
except ImportError:  # pragma: no cover
    from config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, PDF_DIR, BASE_DIR  # type: ignore
    from api.schemas import (  # type: ignore
        QueryRequest, QueryResponse, CitationItem, AuditTrail,
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        BpLogCreate, BpLogOut, BpStats,
        RespiratoryLogCreate, RespiratoryLogOut, RespiratoryStats,
        MoodLogCreate, MoodLogOut, MoodStats,
        VitalsLogCreate,
        SoapGenerateRequest, SoapResponse, SoapSection,
        GuidelineStatus, HealthResponse,
    )
    from features.glucose_tracker import add_log, get_logs, get_stats, classify_glucose, should_escalate_to_doctor  # type: ignore
    from features.bp_tracker import add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp  # type: ignore
    from features.respiratory_tracker import add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory  # type: ignore
    from features.mood_tracker import add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood  # type: ignore
    from features.soap_summary import generate_soap, soap_to_markdown  # type: ignore

# Auth imports — lazy to avoid circular
try:
    from src.auth.router import router as auth_router
    from src.auth.dependencies import get_current_user, enforce_user_ownership, require_admin
    from src.reviews.router import router as reviews_router
    from src.auth.db import add_query_history, list_query_history
    from src.reviews.evaluator import evaluate_rag
    from src.reviews.service import create_review_request, get_review
    AUTH_ENABLED = True
except ImportError:
    try:
        from auth.router import router as auth_router  # type: ignore
        from auth.dependencies import get_current_user, enforce_user_ownership, require_admin  # type: ignore
        from reviews.router import router as reviews_router  # type: ignore
        from auth.db import add_query_history, list_query_history  # type: ignore
        from reviews.evaluator import evaluate_rag  # type: ignore
        from reviews.service import create_review_request, get_review  # type: ignore
        AUTH_ENABLED = True
    except Exception as e:
        AUTH_ENABLED = False
        auth_router = None  # type: ignore
        reviews_router = None  # type: ignore
        print(f"[auth] disabled: {e}")

# Admin imports
try:
    from src.admin.tracing_router import router as tracing_router
    from src.admin.prompt_router import router as prompt_router
    ADMIN_ENABLED = True
except ImportError:
    try:
        from admin.tracing_router import router as tracing_router  # type: ignore
        from admin.prompt_router import router as prompt_router  # type: ignore
        ADMIN_ENABLED = True
    except Exception as e:
        ADMIN_ENABLED = False
        tracing_router = None  # type: ignore
        prompt_router = None  # type: ignore
        print(f"[admin] disabled: {e}")

app = FastAPI(title=API_TITLE, version=API_VERSION, description="WHO-RAG Infrastructure API — Hướng C: lớp truy vấn y tế đáng tin cậy cho app bên thứ 3. (Auth + HILT + Local Tracing)")

# CORS — env-driven allowlist
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_URL", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
if AUTH_ENABLED and auth_router is not None:
    app.include_router(auth_router)
    app.include_router(reviews_router)  # type: ignore
if ADMIN_ENABLED and tracing_router is not None:
    app.include_router(tracing_router)
    app.include_router(prompt_router)

# ---------- helpers ----------
def _count_pdfs() -> tuple[int, dict]:
    if not PDF_DIR.exists():
        return 0, {}
    total = 0
    by_src: dict[str, int] = {}
    for sub in PDF_DIR.iterdir():
        if sub.is_dir():
            cnt = len(list(sub.glob("*.pdf")))
            if cnt:
                by_src[sub.name] = cnt
                total += cnt
        elif sub.suffix.lower() == ".pdf":
            total += 1
            by_src["_root"] = by_src.get("_root", 0) + 1
    return total, by_src

def _resolve_user_id(req_user_id: str, current_user: dict) -> str:
    """User thường chỉ được dùng own id, expert/admin được override"""
    role = current_user.get("role", "user")
    if role in ("admin", "doctor", "pharmacist", "specialist"):
        return req_user_id or current_user["id"]
    # user role: enforce own
    if req_user_id and req_user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden: user can only act on own user_id")
    return current_user["id"]

# ---------- health ----------
@app.get("/health", tags=["system"])
@app.get(f"{API_PREFIX}/health", tags=["system"])
def health():
    total, _ = _count_pdfs()
    legacy_ready = (BASE_DIR / "embeddings" / "pdf_db" / "structure").exists()
    openai_ready = (BASE_DIR / "embeddings" / "pdf_db_openai" / "structure").exists()
    if EMBEDDING_PROVIDER == "openai":
        vector_ready = openai_ready or legacy_ready
    else:
        vector_ready = legacy_ready
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
        pdf_count=total,
        vector_db_ready=vector_ready,
    ).model_dump()

# ---------- RAG query (Hướng C) — bắt buộc login ----------
@app.post(f"{API_PREFIX}/query", tags=["rag"])
def rag_query(req: QueryRequest, current_user=Depends(get_current_user)):
    """
    Truy vấn RAG với audit trail + HILT.
    - Retrieval hybrid BM25 + vector
    - Rerank + generation
    - Đánh giá faithfulness 4 metrics → nếu low confidence → tạo review_requests, trả pending
    - Notification push tới expert/admin + user
    """
    t0 = time.perf_counter()
    # Resolve user_id from token
    effective_user_id = current_user["id"] if AUTH_ENABLED and current_user else req.user_id
    # If auth enabled, override req.user_id with effective
    if AUTH_ENABLED and current_user:
        # enforce only if client supplied explicit non-default user_id that mismatches
        # default_user sentinel means "not supplied" — allow
        if req.user_id and req.user_id not in ("default_user", effective_user_id) and current_user["role"] == "user":
            raise HTTPException(status_code=403, detail="user_id mismatch with authenticated user")
    try:
        from src.rag_pipeline import rag_chat
    except ImportError:
        from rag_pipeline import rag_chat  # type: ignore
    try:
        # pass username for tracing
        username = current_user.get("username") if isinstance(current_user, dict) else None
        result = rag_chat(req.query, top_k=req.top_k, user_id=effective_user_id, username=username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}") from e
    raw = result.get("raw_answer", {})
    answer = raw.get("answer") if isinstance(raw, dict) else result.get("formatted_answer", "")
    contexts = result.get("contexts", [])
    cited_sources = raw.get("cited_sources", []) if isinstance(raw, dict) else []
    # HILT evaluation
    evaluation = None
    review_id = None
    status_str = "answered"
    is_low = False
    if AUTH_ENABLED:
        try:
            evaluation = evaluate_rag(req.query, answer, contexts)
            is_low = evaluation.get("is_low_confidence", False)
            if is_low:
                # create review request
                trace_id_val = result.get("trace_id")
                run_id = trace_id_val
                # disease detection via tone?
                disease = None
                try:
                    from src.generator import detect_tone_and_temp
                    tone, _, _ = detect_tone_and_temp(req.query)
                    disease = tone if tone in ("diabetes","hypertension","respiratory","mental") else None
                except Exception:
                    pass
                rev = create_review_request(
                    query=req.query,
                    draft_answer=answer,
                    contexts=contexts,
                    evaluation=evaluation,
                    requester_id=effective_user_id,
                    disease=disease,
                    langsmith_run_id=run_id,
                )
                # link trace to review
                try:
                    from src.observability.tracing_db import update_trace as _ut
                    _ut(trace_id_val, review_id=rev["id"], status="pending_review", is_low_confidence=1, routed_role=evaluation.get("routed_role"))
                except Exception:
                    pass
                review_id = rev["id"]
                status_str = "pending_review"
                # Override answer for pending: keep draft but signal pending
                # Frontend will show pending UI
                # Still log history as pending (already done in create_review_request)
            else:
                # high confidence: log to history directly and update trace
                try:
                    add_query_history(user_id=effective_user_id, query=req.query, answer=answer, status="answered", confidence=evaluation.get("confidence"))
                    from src.observability.tracing_db import update_trace as _ut2
                    _ut2(trace_id_val, status="answered", is_low_confidence=0)
                except Exception:
                    pass
        except Exception as e:
            # evaluator failure should not block answer
            print(f"[HILT] evaluator error: {e}")
            try:
                add_query_history(user_id=effective_user_id, query=req.query, answer=answer, status="answered", confidence=None)
            except Exception:
                pass
    audit = None
    if req.include_audit:
        citations = []
        for c in contexts:
            snippet = (c.get("content") or "")[:300]
            citations.append(CitationItem(
                source_id=str(c.get("source_id", "")),
                dataset=c.get("dataset"),
                section_path=c.get("section_path"),
                page_numbers=c.get("page_numbers"),
                score=c.get("score"),
                content_snippet=snippet,
            ))
        trace_id = result.get("trace_id")
        prompt_ver = result.get("prompt_version")
        audit = AuditTrail(
            cited_sources=cited_sources,
            citations=citations,
            prompt_version=prompt_ver,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    # Build response extras: inject HILT fields via headers? Instead extend QueryResponse model — for backward compat add to langsmith extras
    resp = QueryResponse(
        answer=answer,
        cited_sources=cited_sources,
        contexts=contexts,
        audit=audit,
        langsmith={"trace_id": result.get("trace_id"), "prompt_version": prompt_ver} if result.get("trace_id") else None,
        cache_hit=result.get("cache_hit", False),
        timings=result.get("timings"),
    )
    # Attach HILT via model_extra if available (pydantic v2)
    # We set attributes dynamically
    try:
        resp.status = status_str  # type: ignore
        resp.review_id = review_id  # type: ignore
        resp.evaluation = evaluation  # type: ignore
        resp.is_low_confidence = is_low  # type: ignore
    except Exception:
        pass
    # Also include in dict when serialized (FastAPI will include extra fields if model allows)
    # So we return dict merging
    out = resp.model_dump()
    out["status"] = status_str
    out["review_id"] = review_id
    out["evaluation"] = evaluation
    out["is_low_confidence"] = is_low
    out["effective_user_id"] = effective_user_id
    out["trace_id"] = result.get("trace_id")
    return out

# ---------- Glucose (Hướng A) ----------
@app.post(f"{API_PREFIX}/glucose", response_model=GlucoseLogOut, tags=["glucose"])
def create_glucose_log(payload: GlucoseLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_log(user_id=user_id, value_mgdl=payload.value_mgdl, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return GlucoseLogOut(id=rec["id"], user_id=rec["user_id"], value_mgdl=rec["value_mgdl"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"], classification=rec["classification"], message=rec["message"])

@app.get(f"{API_PREFIX}/glucose/{{user_id}}", tags=["glucose"])
def list_glucose_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_logs(user_id, limit=limit, days=days)
    stats = get_stats(user_id)
    escalate = should_escalate_to_doctor(user_id)
    return {"user_id": user_id, "logs": logs, "stats": GlucoseStats(**stats).model_dump(), "should_escalate": escalate}

# ---------- BP ----------
@app.post(f"{API_PREFIX}/bp", response_model=BpLogOut, tags=["hypertension"])
def create_bp_log(payload: BpLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_bp_log(user_id=user_id, systolic=payload.systolic, diastolic=payload.diastolic, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BpLogOut(id=rec["id"], user_id=rec["user_id"], systolic=rec["systolic"], diastolic=rec["diastolic"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"], classification=rec["classification"], message=rec["message"])

@app.get(f"{API_PREFIX}/bp/{{user_id}}", tags=["hypertension"])
def list_bp_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_bp_logs(user_id, limit=limit, days=days)
    stats = get_bp_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": BpStats(**stats).model_dump(), "should_escalate": should_escalate_bp(user_id)}

# ---------- Respiratory ----------
@app.post(f"{API_PREFIX}/respiratory", response_model=RespiratoryLogOut, tags=["respiratory"])
def create_respiratory_log(payload: RespiratoryLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_respiratory_log(user_id=user_id, peak_flow_percent=payload.peak_flow_percent, personal_best=payload.personal_best, cat_score=payload.cat_score, inhaler_correct=payload.inhaler_correct, inhaler_steps_correct=payload.inhaler_steps_correct, inhaler_steps_total=payload.inhaler_steps_total, measured_at=payload.measured_at, context=payload.context, notes=payload.notes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RespiratoryLogOut(id=rec["id"], user_id=rec["user_id"], peak_flow_percent=rec["peak_flow_percent"], gold_stage=rec["gold_stage"], classification=rec["classification"], message=rec["message"], measured_at=rec["measured_at"], context=rec["context"], notes=rec["notes"])

@app.get(f"{API_PREFIX}/respiratory/{{user_id}}", tags=["respiratory"])
def list_respiratory_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_respiratory_logs(user_id, limit=limit, days=days)
    stats = get_respiratory_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": RespiratoryStats(**stats).model_dump(), "should_escalate": should_escalate_respiratory(user_id)}

# ---------- Mental ----------
@app.post(f"{API_PREFIX}/mood", response_model=MoodLogOut, tags=["mental"])
def create_mood_log(payload: MoodLogCreate, current_user=Depends(get_current_user)):
    user_id = payload.user_id
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(payload.user_id, current_user)
    try:
        rec = add_mood_log(user_id=user_id, phq9_score=payload.phq9_score, gad7_score=payload.gad7_score, mood_notes=payload.mood_notes, measured_at=payload.measured_at, context=payload.context)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return MoodLogOut(id=rec["id"], user_id=rec["user_id"], phq9_score=rec["phq9_score"], gad7_score=rec["gad7_score"], mood_notes=rec["mood_notes"], original_notes_had_pii=rec["original_notes_had_pii"], classification=rec["classification"], message=rec["message"], crisis_flag=rec["crisis_flag"], crisis_keywords=rec["crisis_keywords"], measured_at=rec["measured_at"], context=rec["context"])

@app.get(f"{API_PREFIX}/mood/{{user_id}}", tags=["mental"])
def list_mood_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None), current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        enforce_user_ownership(user_id, current_user)
    logs = get_mood_logs(user_id, limit=limit, days=days)
    stats = get_mood_stats(user_id)
    return {"user_id": user_id, "logs": logs, "stats": MoodStats(**stats).model_dump(), "should_escalate": should_escalate_mood(user_id)}

# ---------- Unified Vitals ----------
@app.post(f"{API_PREFIX}/vitals", tags=["vitals"])
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

# ---------- SOAP ----------
@app.post(f"{API_PREFIX}/soap/generate", response_model=SoapResponse, tags=["soap"])
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

@app.post(f"{API_PREFIX}/soap/generate/markdown", response_class=PlainTextResponse, tags=["soap"])
def generate_soap_markdown(req: SoapGenerateRequest, current_user=Depends(get_current_user)):
    user_id = req.user_id
    if AUTH_ENABLED and current_user:
        if current_user["role"] == "user" and req.user_id != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        user_id = req.user_id or current_user["id"]
    data = generate_soap(user_id, days=req.days, language=req.language, disease=req.disease)
    md = soap_to_markdown(data)
    return md

# ---------- Guidelines status (public, but also auth optional) ----------
@app.get(f"{API_PREFIX}/guidelines/status", response_model=GuidelineStatus, tags=["guidelines"])
def guidelines_status():
    total, by_src = _count_pdfs()
    manifest = BASE_DIR / "data" / "guideline_manifest.json"
    last_updated = None
    if manifest.exists():
        last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(manifest.stat().st_mtime))
    return GuidelineStatus(total_pdfs=total, by_source=by_src, manifest_path=str(manifest.relative_to(BASE_DIR)) if manifest.exists() else "data/guideline_manifest.json", last_updated=last_updated)

# ---------- Root ----------
@app.get("/", tags=["system"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "query": f"POST {API_PREFIX}/query (auth required)",
        "glucose": f"POST {API_PREFIX}/glucose (auth)",
        "bp": f"POST {API_PREFIX}/bp (auth)",
        "respiratory": f"POST {API_PREFIX}/respiratory (auth)",
        "mood": f"POST {API_PREFIX}/mood (auth)",
        "vitals": f"POST {API_PREFIX}/vitals (auth)",
        "soap": f"POST {API_PREFIX}/soap/generate (auth)",
        "auth": f"POST {API_PREFIX}/auth/login, /register, /me",
        "reviews": f"GET {API_PREFIX}/reviews (expert/admin), POST {API_PREFIX}/reviews/{{id}}/decide",
        "notifications": f"GET {API_PREFIX}/auth/notifications",
    }
