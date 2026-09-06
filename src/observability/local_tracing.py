"""Local tracing engine — ContextVar based, no LangSmith"""
from __future__ import annotations
import contextvars
import functools
import time
import uuid
from typing import Optional, Dict, Any, Callable
import hashlib

try:
    from src.observability.tracing_db import create_trace, update_trace, create_span, end_span, add_span_chunks
except ImportError:
    from observability.tracing_db import create_trace, update_trace, create_span, end_span, add_span_chunks  # type: ignore

_trace_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("current_trace", default=None)
_span_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_span", default=None)

def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:length]

def start_trace(user_id: str, username: Optional[str], query: str, tone: Optional[str]=None, model: Optional[str]=None, embedding_model: Optional[str]=None, chunking_strategy: Optional[str]=None, top_k: Optional[int]=None) -> str:
    trace_id = create_trace(user_id=user_id, username=username, query=query, tone=tone, model=model, embedding_model=embedding_model, chunking_strategy=chunking_strategy, top_k=top_k)
    _trace_ctx.set({"trace_id": trace_id, "user_id": user_id})
    return trace_id

def get_current_trace_id() -> Optional[str]:
    ctx = _trace_ctx.get()
    return ctx["trace_id"] if ctx else None

def end_trace(trace_id: str, answer: Optional[str]=None, status: Optional[str]=None, total_latency_ms: Optional[float]=None, prompt_version: Optional[str]=None, is_low: Optional[bool]=None, routed_role: Optional[str]=None, review_id: Optional[str]=None):
    fields: Dict[str, Any] = {}
    if answer is not None:
        fields["answer"] = answer
    if status:
        fields["status"] = status
    if total_latency_ms is not None:
        fields["total_latency_ms"] = total_latency_ms
    if prompt_version:
        fields["prompt_version"] = prompt_version
    if is_low is not None:
        fields["is_low_confidence"] = 1 if is_low else 0
    if routed_role:
        fields["routed_role"] = routed_role
    if review_id:
        fields["review_id"] = review_id
    if fields:
        update_trace(trace_id, **fields)
    _trace_ctx.set(None)

def start_span(name: str, inputs: Optional[Dict]=None) -> str:
    trace_id = get_current_trace_id()
    if not trace_id:
        # create detached span id but not persisted
        return f"noop_{uuid.uuid4().hex[:8]}"
    parent = _span_ctx.get()
    sid = create_span(trace_id=trace_id, name=name, parent_id=parent, inputs=inputs)
    _span_ctx.set(sid)
    return sid

def finish_span(span_id: str, outputs: Optional[Dict]=None, metadata: Optional[Dict]=None):
    if span_id.startswith("noop"):
        return
    end_span(span_id, outputs=outputs, metadata=metadata)
    # reset to parent? For simplicity clear
    _span_ctx.set(None)

def trace_span(name: str):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            sid = start_span(name, inputs={"args": str(args)[:500]})
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                dur = (time.perf_counter()-t0)*1000
                # try to capture output snippet
                out = {"result": str(result)[:1000]} if result is not None else {}
                finish_span(sid, outputs=out, metadata={"duration_ms": dur})
                return result
            except Exception as e:
                finish_span(sid, outputs={"error": str(e)}, metadata={"duration_ms": (time.perf_counter()-t0)*1000})
                raise
        return wrapper
    return decorator
