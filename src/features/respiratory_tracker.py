"""
Asthma & COPD tracker — inhaler technique is the differentiator (Bach Mai 2016).
Stores peak flow, GOLD, CAT, inhaler steps.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

try:
    from src.config import RESPIRATORY_THRESHOLDS, VITALS_DB_PATH
except ImportError:
    from config import RESPIRATORY_THRESHOLDS, VITALS_DB_PATH  # type: ignore

def _get_conn() -> sqlite3.Connection:
    VITALS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VITALS_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_respiratory_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS respiratory_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            peak_flow_percent INTEGER,
            personal_best INTEGER,
            gold_stage TEXT,
            cat_score INTEGER,
            inhaler_correct BOOLEAN,
            inhaler_steps_correct INTEGER,
            inhaler_steps_total INTEGER,
            measured_at TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'random',
            notes TEXT,
            classification TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resp_user_time ON respiratory_logs(user_id, measured_at)")
    conn.commit()
    conn.close()

def classify_peak_flow(percent: Optional[int]) -> Tuple[str,str]:
    if percent is None:
        return "unknown", "No peak flow provided."
    t = RESPIRATORY_THRESHOLDS
    if percent >= t["peak_flow_green_min"]:
        return "green", f"Green zone (≥{t['peak_flow_green_min']}%) — Go, controlled."
    if percent >= t["peak_flow_yellow_min"]:
        return "yellow", f"Yellow zone (50-79%) — Caution, follow action plan."
    return "red", f"Red zone (<50%) — Medical alert, seek care."

def classify_gold(cat_score: Optional[int]) -> str:
    if cat_score is None:
        return "unknown"
    if cat_score <= 10:
        return "GOLD_1_mild"
    if cat_score <= 20:
        return "GOLD_2_moderate"
    if cat_score <= 30:
        return "GOLD_3_severe"
    return "GOLD_4_very_severe"

def _classify_inhaler(correct: Optional[bool], steps_correct: Optional[int], steps_total: Optional[int]) -> Tuple[str,str]:
    if correct is None and steps_correct is None:
        return "unknown", "No inhaler data."
    if correct is True:
        return "correct", "Inhaler technique correct."
    if steps_correct is not None and steps_total:
        rate = steps_correct/steps_total if steps_total else 0
        if rate >= 0.9:
            return "correct", f"Inhaler {steps_correct}/{steps_total} correct (≥90%)."
        if rate >= 0.7:
            return "partial", f"Inhaler {steps_correct}/{steps_total} partially correct — review video."
        return "incorrect", f"Inhaler {steps_correct}/{steps_total} incorrect — technique training needed."
    if correct is False:
        return "incorrect", "Inhaler technique incorrect — video check recommended."
    return "unknown", "No inhaler data."

def add_respiratory_log(user_id: str, peak_flow_percent: Optional[int] = None, personal_best: Optional[int] = None,
                        cat_score: Optional[int] = None, inhaler_correct: Optional[bool] = None,
                        inhaler_steps_correct: Optional[int] = None, inhaler_steps_total: Optional[int] = None,
                        measured_at: Optional[datetime] = None, context: str = "random", notes: Optional[str] = None) -> Dict[str, Any]:
    init_respiratory_db()
    measured_at = measured_at or datetime.utcnow()
    peak_zone, peak_msg = classify_peak_flow(peak_flow_percent)
    gold = classify_gold(cat_score)
    inhaler_cls, inhaler_msg = _classify_inhaler(inhaler_correct, inhaler_steps_correct, inhaler_steps_total)
    # overall classification: worst of peak and inhaler
    if peak_zone == "red" or inhaler_cls == "incorrect":
        classification = "red"
        message = f"{peak_msg} {inhaler_msg} — Action needed."
    elif peak_zone == "yellow" or inhaler_cls == "partial":
        classification = "yellow"
        message = f"{peak_msg} {inhaler_msg}"
    elif peak_zone == "green" and inhaler_cls in ("correct","unknown"):
        classification = "green"
        message = f"{peak_msg} {inhaler_msg}"
    else:
        classification = peak_zone
        message = f"{peak_msg} {inhaler_msg}"
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO respiratory_logs (user_id, peak_flow_percent, personal_best, gold_stage, cat_score, inhaler_correct, inhaler_steps_correct, inhaler_steps_total, measured_at, context, notes, classification, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, peak_flow_percent, personal_best, gold, cat_score, inhaler_correct, inhaler_steps_correct, inhaler_steps_total, measured_at.isoformat(), context, notes, classification, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {"id": row_id, "user_id": user_id, "peak_flow_percent": peak_flow_percent, "gold_stage": gold,
            "classification": classification, "message": message, "measured_at": measured_at, "context": context, "notes": notes}

def get_respiratory_logs(user_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
    init_respiratory_db()
    conn = _get_conn()
    q = "SELECT * FROM respiratory_logs WHERE user_id=? "
    params: List[Any] = [user_id]
    if days is not None:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        q += "AND measured_at >= ? "
        params.append(since)
    q += "ORDER BY measured_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [{k:r[k] for k in r.keys()} for r in rows]

def get_respiratory_stats(user_id: str) -> Dict[str, Any]:
    logs = get_respiratory_logs(user_id, limit=1000)
    total=len(logs)
    if total==0:
        return {"user_id": user_id, "total_logs":0, "avg_peak_flow":None, "red_rate":0.0, "incorrect_inhaler_rate":0.0, "classification_counts":{}}
    vals=[l["peak_flow_percent"] for l in logs if l["peak_flow_percent"] is not None]
    avg = round(sum(vals)/len(vals),1) if vals else None
    counts={}
    for l in logs:
        counts[l["classification"]] = counts.get(l["classification"],0)+1
    red_rate = counts.get("red",0)/total
    incorrect = sum(1 for l in logs if l["classification"]=="red" or l["inhaler_correct"]==0)
    return {"user_id": user_id, "total_logs": total, "avg_peak_flow": avg, "red_rate": round(red_rate,2),
            "incorrect_inhaler_rate": round(incorrect/total,2) if total else 0, "classification_counts": counts}

def should_escalate_respiratory(user_id: str) -> bool:
    logs=get_respiratory_logs(user_id, limit=3)
    return any(l["classification"]=="red" for l in logs)
