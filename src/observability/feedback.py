"""
User feedback collection on LangSmith runs.

Wires thumbs-up/down (and optional comments) from the UI back to the exact
trace that produced an answer via client.create_feedback().
"""
import logging
import uuid
from typing import Any, Dict, Optional

try:
    from .tracing import get_client
except ImportError:  # pragma: no cover - direct script execution fallback
    from tracing import get_client

logger = logging.getLogger(__name__)


def submit_feedback(
    run_id: Optional[str],
    key: str = "user_feedback",
    score: Optional[float] = None,
    value: Any = None,
    comment: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Attach feedback to a LangSmith run.

    Args:
        run_id: id of the traced root run (see tracing.get_current_trace_info()).
        key: feedback category, e.g. "user_feedback", "correctness".
        score: numeric score (1 = thumbs up, 0 = thumbs down).
        value: alternative free-form value (e.g. "positive").
        comment: optional reviewer comment.
        extra: optional metadata dict stored alongside the feedback.

    Returns True when the feedback reached LangSmith.
    """
    if not run_id:
        logger.debug("submit_feedback skipped: no run_id (tracing disabled?).")
        return False

    client = get_client()
    if client is None:
        return False

    try:
        parsed_id = uuid.UUID(str(run_id))
    except ValueError:
        logger.warning("submit_feedback skipped: invalid run_id %r", run_id)
        return False

    payload: Dict[str, Any] = {}
    if score is not None:
        payload["score"] = float(score)
    if value is not None:
        payload["value"] = value
    if comment:
        payload["comment"] = comment
    if extra:
        payload.update(extra)

    try:
        client.create_feedback(parsed_id, key=key, **payload)
        logger.info("Feedback '%s' submitted for run %s.", key, run_id)
        return True
    except Exception as exc:
        logger.warning("Failed to submit feedback for run %s: %s", run_id, exc)
        return False


def submit_thumbs_up(run_id: Optional[str], comment: Optional[str] = None) -> bool:
    """Convenience wrapper: positive user feedback."""
    return submit_feedback(
        run_id, key="user_feedback", score=1.0, value="positive", comment=comment
    )


def submit_thumbs_down(run_id: Optional[str], comment: Optional[str] = None) -> bool:
    """Convenience wrapper: negative user feedback."""
    return submit_feedback(
        run_id, key="user_feedback", score=0.0, value="negative", comment=comment
    )
