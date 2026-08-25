"""
LangSmith tracing setup and helpers.

Central place for:
- syncing LangSmith env vars so SDKs auto-instrument
- lazily creating a shared langsmith.Client
- attaching runtime metadata to the current run tree
- exposing the current trace id / URL for feedback linking
"""
import hashlib
import logging
import os
import uuid
from functools import lru_cache
from typing import Any, Dict, Optional

try:
    from config import (
        LANGSMITH_ENDPOINT,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        LANGSMITH_TRACING,
    )
except ImportError:  # pragma: no cover - allows running as "src." package too
    from src.config import (
        LANGSMITH_ENDPOINT,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        LANGSMITH_TRACING,
    )

logger = logging.getLogger(__name__)

_setup_done = False


def setup_langsmith() -> bool:
    """
    Sync LangSmith settings into os.environ (both new LANGSMITH_* and legacy
    LANGCHAIN_* names) so that langsmith + wrapped SDKs pick them up.

    Safe to call multiple times. Returns True when tracing is fully enabled.
    """
    global _setup_done
    enabled = is_tracing_enabled()

    if not enabled:
        # Make sure nothing half-configured tries to phone home.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
        _setup_done = True
        logger.info("LangSmith tracing disabled (missing API key or disabled by env).")
        return False

    project = LANGSMITH_PROJECT or "customer-support-rag"
    endpoint = LANGSMITH_ENDPOINT or "https://api.smith.langchain.com"

    for key, value in {
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGSMITH_API_KEY": LANGSMITH_API_KEY,
        "LANGCHAIN_API_KEY": LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": project,
        "LANGCHAIN_PROJECT": project,
        "LANGSMITH_ENDPOINT": endpoint,
        "LANGCHAIN_ENDPOINT": endpoint,
    }.items():
        if value is not None:
            os.environ[key] = value

    _setup_done = True
    logger.info("LangSmith tracing enabled -> project='%s' endpoint='%s'", project, endpoint)
    return True


def is_tracing_enabled() -> bool:
    """Tracing requires both the flag and an API key."""
    return bool(LANGSMITH_TRACING) and bool(LANGSMITH_API_KEY)


@lru_cache(maxsize=1)
def get_client():
    """Shared langsmith.Client singleton, or None when tracing is unavailable."""
    if not LANGSMITH_API_KEY:
        return None
    try:
        from langsmith import Client

        return Client(
            api_key=LANGSMITH_API_KEY,
            api_url=LANGSMITH_ENDPOINT or None,
        )
    except Exception as exc:  # never crash the app because of telemetry
        logger.warning("Could not create LangSmith client: %s", exc)
        return None


def current_run_tree():
    """Return the active RunTree (inside a @traceable function) or None."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        return get_current_run_tree()
    except Exception:
        return None


def add_trace_metadata(**metadata: Any) -> None:
    """Attach metadata to the currently active run (no-op outside a trace)."""
    rt = current_run_tree()
    if rt is None:
        return
    try:
        rt.metadata.update(metadata)
    except Exception as exc:
        logger.debug("Failed to attach trace metadata: %s", exc)


def add_trace_outputs(**outputs: Any) -> None:
    """Merge extra keys into the active run's outputs (no-op outside a trace)."""
    rt = current_run_tree()
    if rt is None:
        return
    try:
        if isinstance(rt.outputs, dict):
            rt.outputs.update(outputs)
        else:
            rt.outputs = {**(rt.outputs or {}), **outputs}
    except Exception as exc:
        logger.debug("Failed to attach trace outputs: %s", exc)


def get_current_trace_info() -> Dict[str, Optional[str]]:
    """
    Return {"run_id", "url", "enabled"} for the active run.
    Used to link UI feedback buttons back to the exact trace.
    The url is best-effort while the run is still in flight; use
    resolve_trace_url() afterwards for a guaranteed link.
    """
    rt = current_run_tree()
    info: Dict[str, Optional[str]] = {
        "run_id": None,
        "url": None,
        "enabled": is_tracing_enabled(),
    }
    if rt is not None:
        info["run_id"] = str(rt.id)
        try:
            info["url"] = rt.get_url()  # langsmith >= 0.7
        except AttributeError:  # pragma: no cover - older SDKs
            try:
                info["url"] = rt.get_run_url()
            except Exception:
                pass
        except Exception:
            pass
    return info


def resolve_trace_url(run_id: Optional[str]) -> Optional[str]:
    """
    Resolve a stable LangSmith UI URL for a finished run.
    Safe to call after the trace completed (e.g. when rendering the UI).
    """
    if not run_id:
        return None
    client = get_client()
    if client is None:
        return None
    try:
        parsed = uuid.UUID(str(run_id))
    except ValueError:
        return None
    try:
        run = client.read_run(parsed)
        return client.get_run_url(run=run)
    except Exception as exc:
        logger.debug("Could not resolve trace URL for %s: %s", run_id, exc)
        return None


def short_hash(text: str, length: int = 8) -> str:
    """Stable short hash used as a lightweight prompt version marker."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:length]


def log_evaluation_summary(name: str, metrics: Dict[str, Dict[str, float]], k: int) -> bool:
    """
    Push an aggregate retrieval-evaluation snapshot into LangSmith as its own
    top-level run so it shows up in dashboards/monitoring.
    """
    client = get_client()
    if client is None:
        return False
    try:
        client.create_run(
            name=name,
            run_type="chain",
            inputs={"k": k},
            outputs={"metrics": metrics},
            project_name=LANGSMITH_PROJECT or None,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to log evaluation summary to LangSmith: %s", exc)
        return False


# Configure on import so every consumer (generator, pipeline, app) inherits it.
setup_langsmith()
