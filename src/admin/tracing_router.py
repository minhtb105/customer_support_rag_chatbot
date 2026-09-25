"""Admin Tracing API — 30d retention, pagination 10, RAGAS on-demand"""
from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

try:
    from src.auth.dependencies import require_admin
    from src.shared.observability.tracing_db import (
        list_traces, get_trace, list_spans, list_chunks, get_ragas, upsert_ragas, delete_expired_traces
    )
    from src.shared.observability.tracing_db import TRACING_DB_PATH
    from src.reviews.evaluator import evaluate_rag
    from src.shared.config import TRACING_PAGE_SIZE, EMBEDDING_PROVIDER, EMBEDDING_MODEL
except ImportError:
    from auth.dependencies import require_admin  # type: ignore
    from shared.observability.tracing_db import list_traces, get_trace, list_spans, list_chunks, get_ragas, upsert_ragas, delete_expired_traces  # type: ignore
    from shared.observability.tracing_db import TRACING_DB_PATH  # type: ignore
    from reviews.evaluator import evaluate_rag  # type: ignore
    from shared.config import TRACING_PAGE_SIZE, EMBEDDING_PROVIDER, EMBEDDING_MODEL  # type: ignore

router = APIRouter(prefix="/v1/admin", tags=["admin-tracing"])

# --- Admin chunk viewer (collections) ---
CHUNK_STRATEGIES = ["structure", "sliding", "semantic", "hybrid_section_semantic"]
CHUNK_SNIPPET_LEN = 500
CHUNK_FULL_MAX = 4000
_COLLECTIONS_CACHE: dict = {}  # (provider, strategy) -> (timestamp, {"count": int, "datasets": list})
_COLLECTIONS_TTL_S = 60


def _get_chroma_collection(strategy: str):
    """Open Chroma collection read-only via PersistentClient (no embeddings, no makedirs)."""
    try:
        from src.chat.retriever import _resolve_db_dir
    except ImportError:
        from chat.retriever import _resolve_db_dir  # type: ignore
    import os
    import chromadb
    db_dir = _resolve_db_dir(strategy)
    if not os.path.isdir(db_dir):
        return None
    try:
        client = chromadb.PersistentClient(path=db_dir)
        return client.get_collection("langchain")
    except Exception:
        return None


def _collection_overview(strategy: str) -> dict:
    import time as _time
    provider = EMBEDDING_PROVIDER
    key = (provider, strategy)
    now = _time.time()
    cached = _COLLECTIONS_CACHE.get(key)
    if cached and (now - cached[0]) < _COLLECTIONS_TTL_S:
        data = cached[1]
        return {"strategy": strategy, "exists": True, "count": data["count"], "datasets": data["datasets"]}
    coll = _get_chroma_collection(strategy)
    if coll is None:
        return {"strategy": strategy, "exists": False, "count": 0, "datasets": []}
    try:
        count = coll.count()
    except Exception:
        count = 0
    datasets: list = []
    try:
        res = coll.get(include=["metadatas"])
        seen = set()
        for m in (res.get("metadatas") or []):
            ds = (m or {}).get("dataset")
            if ds and ds not in seen:
                seen.add(ds)
                datasets.append(ds)
        datasets.sort()
    except Exception:
        datasets = []
    _COLLECTIONS_CACHE[key] = (now, {"count": count, "datasets": datasets})
    return {"strategy": strategy, "exists": True, "count": count, "datasets": datasets}


def _serialize_chunk(vector_id: str, doc: str, meta: dict) -> dict:
    try:
        from src.chat.retriever import deserialize_metadata, EMBEDDING_MODEL as _EMB_MODEL
    except ImportError:
        from chat.retriever import deserialize_metadata, EMBEDDING_MODEL as _EMB_MODEL  # type: ignore
    m = deserialize_metadata(dict(meta or {}))
    full = (doc or "")[:CHUNK_FULL_MAX]
    snippet = (doc or "")[:CHUNK_SNIPPET_LEN]
    section = m.get("section_path")
    if section is not None and not isinstance(section, list):
        section = [section]
    pages = m.get("page_numbers")
    if pages is not None and not isinstance(pages, list):
        pages = [pages]
    return {
        "id": vector_id,
        "vector_id": vector_id,
        "content_snippet": snippet,
        "content_full": full,
        "source_id": m.get("source_id"),
        "file_name": m.get("file_name") or m.get("source_id"),
        "dataset": m.get("dataset"),
        "section_path": section,
        "page_numbers": pages,
        "chunk_index": m.get("chunk_index"),
        "chunk_hash": m.get("chunk_hash"),
        "chunking_strategy": m.get("chunking_strategy"),
        "embedding_model": m.get("embedding_model") or _EMB_MODEL,
        "updated_at": m.get("updated_at"),
    }

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
    # Exclude memory rows from RAGAS ctxs (memory would inflate faithfulness/precision).
    _MEMORY_DATASETS = {"short_term_memory", "episodic_memory", "long_term_memory"}
    chunks = [c for c in chunks if (c.get("dataset") or "") not in _MEMORY_DATASETS]
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
        from src.shared.observability.tracing_db import update_trace
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
    from src.shared.observability.tracing_db import _conn
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) as c FROM traces").fetchone()["c"]
    low = conn.execute("SELECT COUNT(*) as c FROM traces WHERE is_low_confidence=1").fetchone()["c"]
    avg_lat = conn.execute("SELECT AVG(total_latency_ms) as a FROM traces").fetchone()["a"]
    by_tone = conn.execute("SELECT tone, COUNT(*) as c FROM traces GROUP BY tone").fetchall()
    conn.close()
    return {"total": total, "low_confidence": low, "avg_latency_ms": avg_lat, "by_tone": [dict(r) for r in by_tone]}


@router.get("/collections")
def admin_list_collections(current_user=Depends(require_admin)):
    return {"collections": [_collection_overview(s) for s in CHUNK_STRATEGIES], "provider": EMBEDDING_PROVIDER}


@router.get("/collections/{strategy}/chunks")
def admin_list_collection_chunks(
    strategy: str,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=10, description="max 10 per spec"),
    dataset: Optional[str] = None,
    q: Optional[str] = None,
    current_user=Depends(require_admin),
):
    if strategy not in CHUNK_STRATEGIES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy}")
    qq = (q or "").strip()
    ds = (dataset or "").strip()
    where = {"dataset": ds} if ds and ds.lower() != "all" else None
    # NOTE: where_document $contains is substring, case-sensitive (Chroma default). Advanced search -> v2.
    where_document = {"$contains": qq} if qq else None
    coll = _get_chroma_collection(strategy)
    if coll is None:
        return {"chunks": [], "total": 0, "page": page, "limit": limit, "total_pages": 0,
                "collection": strategy, "dataset": ds or "all", "q": qq}
    offset = (page - 1) * limit
    try:
        paged = coll.get(where=where, where_document=where_document, limit=limit, offset=offset,
                         include=["documents", "metadatas"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma get failed: {e}") from e
    try:
        counted = coll.get(where=where, where_document=where_document, include=[])
        total = len(counted.get("ids") or [])
    except Exception:
        total = len(paged.get("ids") or [])
    ids = paged.get("ids") or []
    docs = paged.get("documents") or []
    metas = paged.get("metadatas") or []
    chunks = [_serialize_chunk(ids[i], docs[i] if i < len(docs) else "", metas[i] if i < len(metas) else {})
              for i in range(len(ids))]
    total_pages = (total + limit - 1) // limit if limit and total else 0
    return {"chunks": chunks, "total": total, "page": page, "limit": limit, "total_pages": total_pages,
            "collection": strategy, "dataset": ds or "all", "q": qq}


# --- Admin long-term memory facts per-user (read-only, no singletons) ---
MEMORY_FACT_TYPES = {"medication", "symptom", "condition", "allergy", "lifestyle", "general"}
MEMORY_SNIPPET_LEN = 500
MEMORY_FULL_MAX = 4000


def _get_memory_collection():
    """Open long-term memory Chroma collection directly via PersistentClient.

    Never use get_long_term_memory()/LongTermMemory() here (global singleton
    swaps user + constructor loads embedding model on every request).
    """
    import os
    try:
        from src.shared.config import BASE_DIR, EMBEDDING_PROVIDER as _prov
    except ImportError:
        from shared.config import BASE_DIR, EMBEDDING_PROVIDER as _prov  # type: ignore
    sub = "embeddings/memory_db_openai" if str(_prov or "").lower() == "openai" else "embeddings/memory_db"
    db_dir = str(BASE_DIR / sub)
    if not os.path.isdir(db_dir):
        return None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_dir)
        return client.get_collection("langchain")
    except Exception:
        return None


def _serialize_memory_fact(vector_id: str, doc: str, meta: dict) -> dict:
    m = dict(meta or {})
    text = doc or ""
    raw_entities = m.get("entities", "")
    if isinstance(raw_entities, list):
        entities = [str(e).strip() for e in raw_entities if str(e).strip()]
    elif isinstance(raw_entities, str):
        entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
    else:
        entities = []
    ts = m.get("timestamp")
    weight = None
    try:
        if ts is not None:
            try:
                from src.shared.memory.long_term import TemporalWeighting
            except ImportError:
                from shared.memory.long_term import TemporalWeighting  # type: ignore
            weight = TemporalWeighting(90).calculate_weight(float(ts))
    except Exception:
        weight = None
    return {
        "fact_id": m.get("fact_id") or vector_id,
        "text_snippet": text[:MEMORY_SNIPPET_LEN],
        "text_full": text[:MEMORY_FULL_MAX],
        "fact_type": m.get("fact_type"),
        "entities": entities,
        "confidence": m.get("confidence"),
        "source": m.get("source"),
        "timestamp": ts,
        "temporal_weight": weight,
    }


@router.get("/memory/facts")
def admin_list_memory_facts(
    user_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=10, description="max 10 per spec"),
    fact_type: str = Query("all"),
    q: Optional[str] = None,
    current_user=Depends(require_admin),
):
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=422, detail="user_id is required")
    ft = (fact_type or "all").strip().lower()
    qq = (q or "").strip()
    if ft != "all" and ft in MEMORY_FACT_TYPES:
        where = {"$and": [{"user_id": uid}, {"fact_type": ft}]}
    else:
        where = {"user_id": uid}
        ft = "all"
    # NOTE: where_document $contains is substring, case-sensitive (Chroma default).
    where_document = {"$contains": qq} if qq else None
    coll = _get_memory_collection()
    if coll is None:
        return {"facts": [], "total": 0, "page": page, "limit": limit, "total_pages": 0,
                "user_id": uid, "fact_type": ft, "q": qq}
    offset = (page - 1) * limit
    try:
        paged = coll.get(where=where, where_document=where_document, limit=limit, offset=offset,
                         include=["documents", "metadatas"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma get failed: {e}") from e
    try:
        counted = coll.get(where=where, where_document=where_document, include=[])
        total = len(counted.get("ids") or [])
    except Exception:
        total = len(paged.get("ids") or [])
    ids = paged.get("ids") or []
    docs = paged.get("documents") or []
    metas = paged.get("metadatas") or []
    facts = [_serialize_memory_fact(ids[i], docs[i] if i < len(docs) else "", metas[i] if i < len(metas) else {})
             for i in range(len(ids))]
    total_pages = (total + limit - 1) // limit if limit and total else 0
    return {"facts": facts, "total": total, "page": page, "limit": limit, "total_pages": total_pages,
            "user_id": uid, "fact_type": ft, "q": qq}
