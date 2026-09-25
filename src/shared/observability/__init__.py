"""
Observability package — re-exports canonical local_tracing engine.
Legacy LangSmith shim (tracing.py / feedback.py) removed 2026-09-07.
Use `from src.shared.observability.local_tracing import ...` directly.
"""
from .local_tracing import short_hash, get_current_trace_id, start_trace, end_trace, start_span, finish_span, trace_span

__all__ = [
    "short_hash",
    "get_current_trace_id",
    "start_trace",
    "end_trace",
    "start_span",
    "finish_span",
    "trace_span",
]
