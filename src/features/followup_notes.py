"""Shared notes+episodic dual-write helper (D6).

Single implementation used by both POST /v1/glucose/followup and
POST /v1/triage — avoids copy-paste drift. Fail-open, never raises.
Appended notes are truncated to ~500 chars (tail kept).

Triage marker (D1): triage-sourced writes append a trailing
" [AI Triage]" tag (body truncated to ~470 first so the tag survives).
Detection is segment-exact on " | " split, NOT substring.
FQG followup default source=None stays untagged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

NOTES_KEEP_CHARS = 500
TRIAGE_TAG_SEGMENT = "[AI Triage]"
TRIAGE_TAG = f" | {TRIAGE_TAG_SEGMENT}"
TRIAGE_BODY_KEEP = 470

_SEGMENT_SPLIT_RE = re.compile(r" \| ")


def has_triage_marker(notes: str | None) -> bool:
    """Segment-exact check: any ' | '-separated segment == '[AI Triage]'."""
    if not notes:
        return False
    return any(seg.strip() == TRIAGE_TAG_SEGMENT for seg in _SEGMENT_SPLIT_RE.split(str(notes)))


def save_followup_notes(
    user_id: str,
    text: str,
    related_log_id: Optional[int] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Update notes of related/latest log + episodic buffer. Never raises."""
    saved_note = False
    saved_episodic = False
    target_id = related_log_id
    try:
        try:
            from src.features import glucose_tracker as _gt
        except ImportError:
            import features.glucose_tracker as _gt  # type: ignore
        if target_id is None:
            try:
                _recent = _gt.get_logs(user_id, limit=1)
                if _recent:
                    target_id = _recent[0]["id"]
            except Exception:
                target_id = None
        if target_id is not None:
            try:
                from src.features.base_tracker import get_conn as _get_conn
            except ImportError:
                from features.base_tracker import get_conn as _get_conn  # type: ignore
            conn = _get_conn(_gt.GLUCOSE_DB_PATH)
            try:
                row = conn.execute(
                    "SELECT notes FROM glucose_logs WHERE id=? AND user_id=?",
                    (target_id, user_id),
                ).fetchone()
                if row is not None:
                    try:
                        prev = row["notes"]
                    except Exception:
                        prev = row[0]
                    new_notes = f"{prev} | {text}" if prev else text
                    if source == "triage":
                        if len(new_notes) > TRIAGE_BODY_KEEP:
                            new_notes = new_notes[-TRIAGE_BODY_KEEP:]
                        if not has_triage_marker(new_notes):
                            new_notes = f"{new_notes}{TRIAGE_TAG}"
                    elif len(new_notes) > NOTES_KEEP_CHARS:
                        new_notes = new_notes[-NOTES_KEEP_CHARS:]
                    conn.execute(
                        "UPDATE glucose_logs SET notes=? WHERE id=? AND user_id=?",
                        (new_notes, target_id, user_id),
                    )
                    conn.commit()
                    saved_note = True
            finally:
                conn.close()
    except Exception:
        pass
    try:
        try:
            from src.memory.episodic import get_episodic_memory
        except ImportError:
            from memory.episodic import get_episodic_memory  # type: ignore
        get_episodic_memory().add_message(
            "user", text, metadata={"user_id": user_id, "related_log_id": target_id}
        )
        saved_episodic = True
    except Exception:
        pass
    return {
        "saved_note": saved_note,
        "saved_episodic": saved_episodic,
        "related_log_id": target_id,
    }
