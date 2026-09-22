"""
Hypertension tracker — larger than diabetes (14.3M, 13% controlled).
Implements BP classification per AHA/ACC 2017/2025.

Stores in metadata/vitals.db table bp_logs (unified DB) via BaseTracker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

try:
    from src.config import BP_THRESHOLDS_MMHG, VITALS_DB_PATH
    from src.features.base_tracker import get_conn as _base_get_conn, init_table, fetch_logs, classification_counts, calc_streak, calc_logs_per_week
except ImportError:  # pragma: no cover
    from config import BP_THRESHOLDS_MMHG, VITALS_DB_PATH  # type: ignore
    from features.base_tracker import get_conn as _base_get_conn, init_table, fetch_logs, classification_counts, calc_streak, calc_logs_per_week  # type: ignore

def _get_conn():
    return _base_get_conn(VITALS_DB_PATH)

def init_bp_db():
    init_table(
        VITALS_DB_PATH,
        """
        CREATE TABLE IF NOT EXISTS bp_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            systolic INTEGER NOT NULL,
            diastolic INTEGER NOT NULL,
            measured_at TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'random',
            notes TEXT,
            classification TEXT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_bp_user_time ON bp_logs(user_id, measured_at)",
    )

def classify_bp(systolic: int, diastolic: int) -> Tuple[str, str]:
    t = BP_THRESHOLDS_MMHG
    if systolic >= t["crisis_sys"] or diastolic >= t["crisis_dia"]:
        return "crisis", f"Crisis (≥{t['crisis_sys']}/{t['crisis_dia']} mmHg) — seek emergency care immediately."
    if systolic >= t["stage2_sys_min"] or diastolic >= t["stage2_dia_min"]:
        return "stage2", "Stage 2 hypertension (≥140/90) — not at target, clinician review needed."
    if (t["stage1_sys_min"] <= systolic <= t["stage1_sys_max"]) or (t["stage1_dia_min"] <= diastolic <= t["stage1_dia_max"]):
        return "stage1", "Stage 1 hypertension (130-139/80-89) — lifestyle + possible meds."
    if t["elevated_sys_min"] <= systolic <= t["elevated_sys_max"] and diastolic < t["elevated_dia_max"]:
        return "elevated", "Elevated (120-129/<80) — lifestyle counseling."
    if systolic <= t["normal_sys_max"] and diastolic <= t["normal_dia_max"]:
        return "normal", "Normal (<120/80) — at target."
    if systolic >= 130 or diastolic >= 80:
        return "stage1", "Stage 1 range — monitor."
    return "normal", "Within target."

def _should_escalate_bp(logs: List[Dict[str, Any]]) -> bool:
    if not logs:
        return False
    if any(l["classification"] == "crisis" for l in logs[:3]):
        return True
    if len(logs) >= 3 and all(l["classification"] == "stage2" for l in logs[:3]):
        return True
    return False

def add_bp_log(user_id: str, systolic: int, diastolic: int, measured_at: Optional[datetime] = None,
               context: str = "random", notes: Optional[str] = None) -> Dict[str, Any]:
    init_bp_db()
    measured_at = measured_at or datetime.now(timezone.utc)
    if not (50 <= systolic <= 300 and 30 <= diastolic <= 200):
        raise ValueError(f"BP out of range: {systolic}/{diastolic}")
    classification, message = classify_bp(systolic, diastolic)
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO bp_logs (user_id, systolic, diastolic, measured_at, context, notes, classification, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (user_id, systolic, diastolic, measured_at.isoformat(), context, notes, classification, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {"id": row_id, "user_id": user_id, "systolic": systolic, "diastolic": diastolic,
            "measured_at": measured_at, "context": context, "notes": notes,
            "classification": classification, "message": message}

def get_bp_logs(user_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
    init_bp_db()
    return fetch_logs(VITALS_DB_PATH, "bp_logs", user_id, limit, days)

def get_bp_stats(user_id: str) -> Dict[str, Any]:
    logs = get_bp_logs(user_id, limit=1000)
    total = len(logs)
    if total == 0:
        return {"user_id": user_id, "total_logs": 0, "avg_sys": None, "avg_dia": None,
                "last_7_days_avg": None, "streak_days": 0, "logs_per_week": 0.0,
                "classification_counts": {}, "at_target_rate": 0.0}
    avg_sys = sum(l["systolic"] for l in logs) / total
    avg_dia = sum(l["diastolic"] for l in logs) / total
    seven = [l for l in logs if l["measured_at"] >= (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()]
    last7 = None
    if seven:
        last7 = {"sys": round(sum(l["systolic"] for l in seven)/len(seven),1), "dia": round(sum(l["diastolic"] for l in seven)/len(seven),1)}
    counts = classification_counts(logs)
    at_target = counts.get("normal",0)/total if total else 0
    streak = calc_streak(logs)
    lpw = calc_logs_per_week(logs)
    return {"user_id": user_id, "total_logs": total, "avg_sys": round(avg_sys,1), "avg_dia": round(avg_dia,1),
            "last_7_days_avg": last7, "streak_days": streak, "logs_per_week": lpw,
            "classification_counts": counts, "at_target_rate": round(at_target,2)}

def should_escalate_bp(user_id: str) -> bool:
    logs = get_bp_logs(user_id, limit=3)
    return _should_escalate_bp(logs)
