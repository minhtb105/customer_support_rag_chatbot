"""Triage event log (monitor source): every POST /v1/triage call appends one line.

Fail-open, never raises. Excerpt capped at 200 chars (PII-care).
Runtime artifact: logs/triage_events.jsonl (gitignored, never committed).
Tests call these pure helpers directly, never loop POST /v1/triage.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EXCERPT_KEEP = 200
TAIL_READ_CAP = 2000

try:
    from src.shared.config import BASE_DIR as _BASE_DIR
except ImportError:  # pragma: no cover
    _BASE_DIR = Path(__file__).resolve().parents[2]  # type: ignore


def _events_path() -> Path:
    override = os.getenv("TRIAGE_EVENTS_PATH")
    if override:
        return Path(override)
    return Path(_BASE_DIR) / "logs" / "triage_events.jsonl"


def append_triage_event(
    user_id: Optional[str] = None,
    message: str = "",
    emergency: bool = False,
    red_flag_type: Optional[str] = None,
    specialty: Optional[str] = None,
    solver_used: Optional[str] = None,
) -> bool:
    """Append one event line. Fail-open -> bool, never raises."""
    try:
        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "excerpt": (message or "")[:EXCERPT_KEEP],
                "emergency": bool(emergency),
                "red_flag_type": red_flag_type,
                "specialty": specialty,
                "solver_used": solver_used,
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception:
        return False


def _tail_lines(cap: int = TAIL_READ_CAP) -> List[str]:
    try:
        path = _events_path()
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return [ln for ln in lines[-cap:] if ln.strip()]
    except Exception:
        return []


def list_triage_events(
    page: int = 1,
    limit: int = 20,
    specialty: Optional[str] = None,
    emergency: Optional[bool] = None,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    """Read tail (cap 2000), filter, paginate. Pure read, never raises."""
    try:
        items: List[Dict[str, Any]] = []
        for ln in _tail_lines():
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            if specialty and (ev.get("specialty") or "") != specialty:
                continue
            if emergency is not None and bool(ev.get("emergency")) is not emergency:
                continue
            if q and q not in str(ev.get("excerpt") or ""):
                continue
            items.append(ev)
        items.reverse()  # newest first
        total = len(items)
        page = max(1, int(page or 1))
        limit = max(1, min(int(limit or 20), 100))
        start = (page - 1) * limit
        return {
            "events": items[start : start + limit],
            "total": total,
            "page": page,
            "limit": limit,
        }
    except Exception:
        return {"events": [], "total": 0, "page": 1, "limit": 20}
