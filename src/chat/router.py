"""Chat service router — POST /v1/query, POST /v1/query/stream (SSE)."""
from __future__ import annotations

import json as _json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

try:
    from src.shared.config import API_PREFIX
    from src.chat.schemas import QueryRequest, QueryResponse, CitationItem, AuditTrail
    from src.auth.dependencies import get_current_user
    from src.auth.db import add_query_history
    from src.reviews.evaluator import evaluate_rag
    from src.reviews.service import create_review_request
    AUTH_ENABLED = True
except ImportError:
    from shared.config import API_PREFIX  # type: ignore
    from chat.schemas import QueryRequest, QueryResponse, CitationItem, AuditTrail  # type: ignore
    from auth.dependencies import get_current_user  # type: ignore
    from auth.db import add_query_history  # type: ignore
    from reviews.evaluator import evaluate_rag  # type: ignore
    from reviews.service import create_review_request  # type: ignore
    AUTH_ENABLED = True

router = APIRouter(prefix=API_PREFIX, tags=["chat"])


@router.post("/query")
def rag_query(req: QueryRequest, current_user=Depends(get_current_user)):  # noqa: C901
    # Safeguard gate (diabetes scope): hard early-return BEFORE any RAG/LLM call.
    try:
        try:
            from src.diabetes.scope_guard import is_out_of_scope, safeguard_response
        except ImportError:
            from diabetes.scope_guard import is_out_of_scope, safeguard_response  # type: ignore
        if req.query and is_out_of_scope(req.query):
            _safeguard = safeguard_response(req.query)
            return {
                "answer": _safeguard,
                "cited_sources": [],
                "contexts": [],
                "audit": None,
                "langsmith": None,
                "cache_hit": False,
                "timings": {},
                "status": "answered",
                "review_id": None,
                "evaluation": None,
                "is_low_confidence": False,
                "effective_user_id": (current_user["id"] if AUTH_ENABLED and current_user else req.user_id),
                "trace_id": None,
                "safeguard": True,
            }
    except HTTPException:
        raise
    except Exception:
        pass
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
        from src.chat.rag_pipeline import rag_chat
    except ImportError:
        from chat.rag_pipeline import rag_chat  # type: ignore
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
                    from src.chat.generator import detect_tone_and_temp
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
                    from src.shared.observability.tracing_db import update_trace as _ut
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
                    from src.shared.observability.tracing_db import update_trace as _ut2
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


@router.post("/query/stream")
def rag_query_stream(req: QueryRequest, current_user=Depends(get_current_user)):
    """
    SSE streaming variant of /v1/query — POST with auth, streams via text/event-stream.
    Events:
      - event: metadata  data: {trace_id, prompt_version, tone, contexts, cache_hit}
      - event: token     data: {delta: str}  (multiple)
      - event: done      data: {answer, formatted_answer, cited_sources, contexts, trace_id, evaluation, status, ...}
      - event: error     data: {error: str}
    Frontend should use fetch + ReadableStream to parse SSE (EventSource cannot POST).
    """
    effective_user_id = current_user["id"] if AUTH_ENABLED and current_user else req.user_id
    if AUTH_ENABLED and current_user:
        if req.user_id and req.user_id not in ("default_user", effective_user_id) and current_user["role"] == "user":
            raise HTTPException(status_code=403, detail="user_id mismatch with authenticated user")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")
    try:
        from src.chat.rag_pipeline import rag_chat_stream  # type: ignore
    except ImportError:
        from chat.rag_pipeline import rag_chat_stream  # type: ignore

    username = current_user.get("username") if isinstance(current_user, dict) else None

    def event_generator():
        try:
            for evt in rag_chat_stream(req.query, top_k=req.top_k, user_id=effective_user_id, username=username):
                ev = evt.get("event", "message")
                data = evt.get("data", {})
                # SSE wire format
                yield f"event: {ev}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = _json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
