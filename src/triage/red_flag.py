"""Red-flag interceptor for Smart Triage (diabetes demo).

Pure keyword gate, INDEPENDENT from scope_guard/LLM. Runs FIRST at the
POST /v1/triage call level: on hit the endpoint audit-appends and returns
immediately WITHOUT importing/calling triage_nlu, slot_generator, matcher,
scope_guard, or any LLM.

Fail-safe: false-positive accepted, miss is not. Bare "mắt mờ / mat mo"
alone is NOT a flag (it is a routine complication symptom used in the
triage acceptance case); group (a) fires on severe hypo/DKA signs only.
"""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from src.shared.config import BASE_DIR as _BASE_DIR
except ImportError:  # pragma: no cover
    _BASE_DIR = Path(__file__).resolve().parents[2]  # type: ignore


def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


# 3 diabetes red-flag groups: (keywords matched against lowered raw text
# AND unaccented text; multi-word unaccented keys avoid "mu"-style noise).
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "hypo_severe",
        (
            "lơ mơ", "lo mo", "vã mồ hôi", "va mo hoi",
            "hơi thở mùi", "hoi tho mui", "xeton", "acetone", "cetone",
            "bất tỉnh", "bat tinh", "hôn mê", "hon me", "li bì", "li bi",
            "co giật", "co giat",
        ),
    ),
    (
        "foot_infection",
        (
            "hoại tử", "hoai tu", "sưng đỏ", "sung do", "chảy mủ", "chay mu",
            "vết loét", "vet loet", "loét chân", "loet chan",
            "sốt cao", "sot cao", "mưng mủ", "mung mu",
            "nhiễm trùng", "nhiem trung", "có mủ", "co mu",
        ),
    ),
    (
        "cardio_stroke",
        (
            "đau ngực", "dau nguc", "nhồi máu", "nhoi mau",
            "méo miệng", "meo mieng", "yếu nửa người", "yeu nua nguoi",
            "đột quỵ", "dot quy", "tai biến", "tai bien",
            "liệt nửa", "liet nua",
        ),
    ),
)

EMERGENCY_MESSAGE = (
    "🚨 CẢNH BÁO CẤP CỨU: dấu hiệu nguy hiểm, BỎ QUA đặt lịch. "
    "Gọi 115 ngay lập tức hoặc đến cơ sở y tế gần nhất. "
    "Không tự lái xe, không chờ chat trả lời."
)


def check_red_flag(text: str) -> Dict[str, Any]:
    """Pure check. Returns {emergency, red_flag_type, message} — no other keys."""
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    for group, keywords in _GROUPS:
        for kw in keywords:
            if kw in lowered or _unaccent(kw) in plain:
                return {"emergency": True, "red_flag_type": group, "message": EMERGENCY_MESSAGE}
    return {"emergency": False, "red_flag_type": None, "message": ""}


def _audit_path() -> Path:
    override = os.getenv("TRIAGE_AUDIT_PATH")
    if override:
        return Path(override)
    return Path(_BASE_DIR) / "logs" / "triage_redflag.jsonl"


def append_red_flag_audit(
    red_flag_type: str, user_id: Optional[str] = None, excerpt: str = ""
) -> bool:
    """Append one jsonl audit line (fail-open -> bool, never raises)."""
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "red_flag_type": red_flag_type,
                "user_id": user_id,
                "excerpt": (excerpt or "")[:200],
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception:
        return False
