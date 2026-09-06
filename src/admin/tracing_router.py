"""Admin Tracing API — 30d retention, pagination 10, RAGAS on-demand"""
from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

try:
    from src.auth.dependencies import require_admin
    from src.observability.tracing_db import (
        list_traces, get_trace, list_spans, list_chunks, get_ragas, upsert_ragas, delete_expired_traces
    )
    from src.observability.tracing_db import TRACING_DB_PATH
    from src.reviews.evaluator import evaluate_rag
    from src.config import TRACING_PAGE_SIZE
except ImportError:
    from auth.dependencies import require_admin  # type: ignore
    from observability.tracing_db import list_traces, get_trace, list_spans, list_chunks, get_ragas, upsert_ragas, delete_expired_traces  # type: ignore
    from observability.tracing_db import TRACING_DB_PATH  # type: ignore
    from reviews.evaluator import evaluate_rag  # type: ignore
    from config import TRACING_PAGE_SIZE  # type: ignore

router = APIRouter(prefix="/v1/admin", tags=["admin-tracing"])

class RagasTriggerRequest(BaseModel):
    pass

@router.get("/traces")
def admin_list_traces(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=10, description="max 10 per spec"),
    q: Optional[str] = None,
    tone: Optional[str] = None,
    status: Optional[str] = None,
    is_low_confidence: Optional[bool] = None,
    user_id: Optional[str] = None,
    current_user=Depends(require_admin)
):
    # cleanup expired on each list (cheap)
    try:
        delete_expired_traces()
    except Exception:
        pass
    offset = (page - 1) * limit
    rows, total = list_traces(limit=limit, offset=offset, q=q, tone=tone, status=status, is_low=is_low_confidence, user_id=user_id)
    # enrich with ragas flag
    for r in rows:
        rag = get_ragas(r["id"])
        r["has_ragas"] = rag is not None
        if rag:
            r["ragas_summary"] = {"confidence": rag.get("confidence"), "failed": rag.get("failed_metrics")}
    return {"traces": rows, "total": total, "page": page, "limit": limit, "total_pages": (total + limit -1)//limit if limit else 1}

@router.get("/traces/{trace_id}")
def admin_get_trace(trace_id: str, current_user=Depends(require_admin)):
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    spans = list_spans(trace_id)
    chunks = list_chunks(trace_id)
    ragas = get_ragas(trace_id)
    # also fetch review if exists
    review = None
    if trace.get("review_id"):
        try:
            from src.reviews.service import get_review
            review = get_review(trace["review_id"])
        except Exception:
            try:
                from reviews.service import get_review  # type: ignore
                review = get_review(trace["review_id"])
            except Exception:
                pass
    return {"trace": trace, "spans": spans, "chunks": chunks, "ragas": ragas, "review": review}

@router.post("/traces/{trace_id}/ragas")
def admin_trigger_ragas(trace_id: str, current_user=Depends(require_admin)):
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    # check existing
    existing = get_ragas(trace_id)
    # need query, answer, chunks
    query = trace.get("query")
    answer = trace.get("answer")
    if not query or not answer:
        raise HTTPException(status_code=400, detail="Trace missing query/answer")
    chunks = list_chunks(trace_id)
    # convert chunks to evaluator format: list of dict with source_id, content
    ctxs = [{"source_id": c.get("source_id"), "content": c.get("content_snippet") or c.get("content_full") or ""} for c in chunks]
    if not ctxs:
        # fallback to empty
        ctxs = []
    try:
        result = evaluate_rag(query, answer, ctxs)
        # map to ragas table
        metrics = result.get("metrics", {})
        # add fluency if missing
        if "fluency" not in metrics:
            metrics["fluency"] = metrics.get("context_precision", 3.0)
        # extract comments
        comments = result.get("comments", {})
        # build failed
        failed = result.get("failed_metrics", [])
        upsert_ragas(trace_id, {
            "faithfulness": metrics.get("faithfulness"),
            "context_precision": metrics.get("context_precision"),
            "context_recall": metrics.get("context_recall"),
            "answer_relevance": metrics.get("answer_relevance"),
            "fluency": metrics.get("fluency"),
            "faithfulness_comment": comments.get("faithfulness",""),
            "context_precision_comment": comments.get("context_precision",""),
            "context_recall_comment": comments.get("context_recall",""),
            "answer_relevance_comment": comments.get("answer_relevance",""),
            "fluency_comment": comments.get("fluency",""),
            "failed_metrics": failed,
            "confidence": result.get("confidence"),
        }, result.get("raw", {}), evaluator_model=result.get("evaluator_model") or "gpt-4o-mini")
        # also update trace is_low flag
        from src.observability.tracing_db import update_trace
        update_trace(trace_id, is_low_confidence=1 if result.get("is_low_confidence") else 0, routed_role=result.get("routed_role"))
        return get_ragas(trace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAGAS failed: {e}") from e

@router.delete("/traces/cleanup")
def admin_cleanup(current_user=Depends(require_admin)):
    n = delete_expired_traces()
    return {"deleted": n, "retention_days": 30}

@router.get("/traces/stats/summary")
def admin_stats(current_user=Depends(require_admin)):
    from src.observability.tracing_db import _conn
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) as c FROM traces").fetchone()["c"]
    low = conn.execute("SELECT COUNT(*) as c FROM traces WHERE is_low_confidence=1").fetchone()["c"]
    avg_lat = conn.execute("SELECT AVG(total_latency_ms) as a FROM traces").fetchone()["a"]
    by_tone = conn.execute("SELECT tone, COUNT(*) as c FROM traces GROUP BY tone").fetchall()
    conn.close()
    return {"total": total, "low_confidence": low, "avg_latency_ms": avg_lat, "by_tone": [dict(r) for r in by_tone]}
