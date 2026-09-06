"""
FastAPI — WHO-RAG Infrastructure API (Hướng C)

Cung cấp:
  - POST /v1/query              — RAG Q&A với audit trail
  - POST /v1/glucose            — log đường huyết (Hướng A)
  - GET  /v1/glucose/{user_id}  — lịch sử + stats
  - POST /v1/soap/generate      — tóm tắt SOAP trước tái khám (Hướng B)
  - GET  /v1/guidelines/status  — trạng thái corpus guideline
  - GET  /health, GET /v1/health
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

try:
    from src.config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, PDF_DIR, BASE_DIR
    from src.api.schemas import (
        QueryRequest, QueryResponse, CitationItem, AuditTrail,
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        SoapGenerateRequest, SoapResponse, SoapSection,
        GuidelineStatus, HealthResponse,
    )
    from src.features.glucose_tracker import add_log, get_logs, get_stats, classify_glucose, should_escalate_to_doctor
    from src.features.soap_summary import generate_soap, soap_to_markdown
except ImportError:  # pragma: no cover
    from config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, PDF_DIR, BASE_DIR  # type: ignore
    from api.schemas import (  # type: ignore
        QueryRequest, QueryResponse, CitationItem, AuditTrail,
        GlucoseLogCreate, GlucoseLogOut, GlucoseStats,
        SoapGenerateRequest, SoapResponse, SoapSection,
        GuidelineStatus, HealthResponse,
    )
    from features.glucose_tracker import add_log, get_logs, get_stats, classify_glucose, should_escalate_to_doctor  # type: ignore
    from features.soap_summary import generate_soap, soap_to_markdown  # type: ignore

app = FastAPI(title=API_TITLE, version=API_VERSION, description="WHO-RAG Infrastructure API — Hướng C: lớp truy vấn y tế đáng tin cậy cho app bên thứ 3.")

# CORS — env-driven allowlist (OWASP A05 fix: no wildcard with credentials)
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_URL", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------- health ----------
@app.get("/health", tags=["system"])
@app.get(f"{API_PREFIX}/health", tags=["system"])
def health():
    total, _ = _count_pdfs()
    # kiểm tra vector DB có tồn tại không (hỗ trợ cả legacy và openai)
    legacy_ready = (BASE_DIR / "embeddings" / "pdf_db" / "structure").exists()
    openai_ready = (BASE_DIR / "embeddings" / "pdf_db_openai" / "structure").exists()
    if EMBEDDING_PROVIDER == "openai":
        vector_ready = openai_ready or legacy_ready  # fallback
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


# ---------- RAG query (Hướng C) ----------
@app.post(f"{API_PREFIX}/query", response_model=QueryResponse, tags=["rag"])
def rag_query(req: QueryRequest):
    """
    Truy vấn RAG với audit trail.
    - Retrieval: hybrid BM25 + vector (OpenAI embeddings nếu cấu hình)
    - Rerank + generation với system prompt strict
    - Trả về citations, prompt_version, timings
    """
    t0 = time.perf_counter()
    try:
        from src.rag_pipeline import rag_chat  # lazy import để tránh vòng lặp
    except ImportError:
        from rag_pipeline import rag_chat  # type: ignore

    try:
        result = rag_chat(req.query, top_k=req.top_k, user_id=req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}") from e

    raw = result.get("raw_answer", {})
    answer = raw.get("answer") if isinstance(raw, dict) else result.get("formatted_answer", "")
    # contexts
    contexts = result.get("contexts", [])
    # citations
    cited_sources = raw.get("cited_sources", []) if isinstance(raw, dict) else []
    audit = None
    if req.include_audit:
        citations = []
        for c in contexts:
            # c là dict từ ContextItem.model_dump()
            snippet = (c.get("content") or "")[:300]
            citations.append(CitationItem(
                source_id=str(c.get("source_id", "")),
                dataset=c.get("dataset"),
                section_path=c.get("section_path"),
                page_numbers=c.get("page_numbers"),
                score=c.get("score"),
                content_snippet=snippet,
            ))
        langsmith = result.get("langsmith") or {}
        audit = AuditTrail(
            cited_sources=cited_sources,
            citations=citations,
            prompt_version=(langsmith.get("prompt_version") if isinstance(langsmith, dict) else None),
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    return QueryResponse(
        answer=answer,
        cited_sources=cited_sources,
        contexts=contexts,
        audit=audit,
        langsmith=result.get("langsmith"),
        cache_hit=result.get("cache_hit", False),
        timings=result.get("timings"),
    )


# ---------- Glucose (Hướng A) ----------
@app.post(f"{API_PREFIX}/glucose", response_model=GlucoseLogOut, tags=["glucose"])
def create_glucose_log(payload: GlucoseLogCreate):
    try:
        rec = add_log(
            user_id=payload.user_id,
            value_mgdl=payload.value_mgdl,
            measured_at=payload.measured_at,
            context=payload.context,
            notes=payload.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return GlucoseLogOut(
        id=rec["id"],
        user_id=rec["user_id"],
        value_mgdl=rec["value_mgdl"],
        measured_at=rec["measured_at"],
        context=rec["context"],
        notes=rec["notes"],
        classification=rec["classification"],
        message=rec["message"],
    )


@app.get(f"{API_PREFIX}/glucose/{{user_id}}", tags=["glucose"])
def list_glucose_logs(user_id: str, limit: int = Query(50, ge=1, le=200), days: Optional[int] = Query(None)):
    logs = get_logs(user_id, limit=limit, days=days)
    stats = get_stats(user_id)
    escalate = should_escalate_to_doctor(user_id)
    return {
        "user_id": user_id,
        "logs": logs,
        "stats": GlucoseStats(**stats).model_dump(),
        "should_escalate": escalate,
    }


# ---------- SOAP (Hướng B) ----------
@app.post(f"{API_PREFIX}/soap/generate", response_model=SoapResponse, tags=["soap"])
def generate_soap_endpoint(req: SoapGenerateRequest):
    try:
        data = generate_soap(req.user_id, days=req.days, language=req.language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    soap = data["soap"]
    stats = data["stats"]
    return SoapResponse(
        user_id=data["user_id"],
        generated_at=data["generated_at"],
        period=data["period"],
        soap=SoapSection(**soap),
        stats=GlucoseStats(**stats),
    )


@app.post(f"{API_PREFIX}/soap/generate/markdown", response_class=PlainTextResponse, tags=["soap"])
def generate_soap_markdown(req: SoapGenerateRequest):
    data = generate_soap(req.user_id, days=req.days, language=req.language)
    md = soap_to_markdown(data)
    return md


# ---------- Guidelines status ----------
@app.get(f"{API_PREFIX}/guidelines/status", response_model=GuidelineStatus, tags=["guidelines"])
def guidelines_status():
    total, by_src = _count_pdfs()
    manifest = BASE_DIR / "data" / "guideline_manifest.json"
    last_updated = None
    if manifest.exists():
        last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(manifest.stat().st_mtime))
    return GuidelineStatus(
        total_pdfs=total,
        by_source=by_src,
        manifest_path=str(manifest.relative_to(BASE_DIR)) if manifest.exists() else "data/guideline_manifest.json",
        last_updated=last_updated,
    )


# ---------- Root ----------
@app.get("/", tags=["system"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "query": f"POST {API_PREFIX}/query",
        "glucose": f"POST {API_PREFIX}/glucose",
        "soap": f"POST {API_PREFIX}/soap/generate",
    }
