"""
Local tracing re-export — compatibility shim.
All callers should import from observability.local_tracing directly.
We keep short_hash and no-op helpers for backward compat.
"""
from src.observability.local_tracing import short_hash, get_current_trace_id, start_trace, end_trace, start_span, finish_span, trace_span

def add_trace_metadata(**kwargs):
    # No-op shim: metadata now goes via span metadata; keeping for compat with old generator calls
    # We try to attach to current span if exists
    try:
        tid = get_current_trace_id()
        if tid:
            # create a metadata span?
            pass
    except Exception:
        pass

def add_trace_outputs(**kwargs):
    pass

def get_current_trace_info():
    tid = get_current_trace_id()
    return {"run_id": tid, "url": None, "enabled": True}

def setup_langsmith():
    return False

def is_tracing_enabled():
    return True

def get_client():
    return None

def current_run_tree():
    return None

def log_evaluation_summary(*args, **kwargs):
    return False

def resolve_trace_url(*args, **kwargs):
    return None
