"""
Mental health tracker — 15M people, 29% treated, 1k psychiatrists. Largest treatment gap.
Full CRUD with PII redact + crisis safety (hotline, no self-diagnosis).

Stores PHQ-9, GAD-7, crisis flags. PII redact before storage.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

try:
    from src.config import MENTAL_HEALTH_THRESHOLDS, VITALS_DB_PATH, PII_REDACT_FIELDS
except ImportError:
    from config import MENTAL_HEALTH_THRESHOLDS, VITALS_DB_PATH, PII_REDACT_FIELDS  # type: ignore

# ---------- PII redact ----------
def redact_pii(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    redacted = text
    # phone VN: 09..., +84...
    redacted = re.sub(r"(\+84|0)\d{9,10}", "[REDACTED_PHONE]", redacted)
    # email
    redacted = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", redacted)
    # simple name pattern: not perfect, but catch explicit PII fields
    for field in PII_REDACT_FIELDS:
        # if user writes "name: Nguyen Van A", redact value
        redacted = re.sub(rf"{field}\s*[:=]\s*\S+", f"{field}:[REDACTED]", redacted, flags=re.IGNORECASE)
    # CMND/CCCD 9-12 digits
    redacted = re.sub(r"\b\d{9,12}\b", "[REDACTED_ID]", redacted)
    return redacted

def _get_conn() -> sqlite3.Connection:
    VITALS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VITALS_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_mood_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            phq9_score INTEGER,
            gad7_score INTEGER,
            mood_notes TEXT,
            crisis_flag BOOLEAN,
            crisis_keywords TEXT,
            measured_at TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'random',
            classification TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_user_time ON mood_logs(user_id, measured_at)")
    conn.commit()
    conn.close()

def classify_phq9(score: Optional[int]) -> Tuple[str,str]:
    if score is None:
        return "unknown", "No PHQ-9 score."
    t=MENTAL_HEALTH_THRESHOLDS
    if score <= t["phq9_minimal_max"]:
        return "minimal", f"PHQ-9 {score} — Minimal depression."
    if score <= t["phq9_mild_max"]:
        return "mild", f"PHQ-9 {score} — Mild."
    if score <= t["phq9_moderate_max"]:
        return "moderate", f"PHQ-9 {score} — Moderate."
    if score <= t["phq9_moderately_severe_max"]:
        return "moderately_severe", f"PHQ-9 {score} — Moderately severe."
    return "severe", f"PHQ-9 {score} — Severe."

def classify_gad7(score: Optional[int]) -> Tuple[str,str]:
    if score is None:
        return "unknown", "No GAD-7 score."
    t=MENTAL_HEALTH_THRESHOLDS
    if score <= t["gad7_minimal_max"]:
        return "minimal", f"GAD-7 {score} — Minimal anxiety."
    if score <= t["gad7_mild_max"]:
        return "mild", f"GAD-7 {score} — Mild."
    if score <= t["gad7_moderate_max"]:
        return "moderate", f"GAD-7 {score} — Moderate."
    return "severe", f"GAD-7 {score} — Severe."

def contains_crisis_keywords(text: Optional[str]) -> Tuple[bool, List[str]]:
    if not text:
        return False, []
    low=text.lower()
    found=[]
    for kw in MENTAL_HEALTH_THRESHOLDS["crisis_keywords"]:
        if kw.lower() in low:
            found.append(kw)
    return (len(found)>0, found)

def classify_mood(phq9: Optional[int], gad7: Optional[int], notes: Optional[str]) -> Tuple[str,str,bool,List[str]]:
    phq_cls,_ = classify_phq9(phq9)
    gad_cls,_ = classify_gad7(gad7)
    crisis, kws = contains_crisis_keywords(notes)
    # crisis overrides
    if crisis:
        return "crisis", f"Crisis keywords detected {kws} — Provide hotline {MENTAL_HEALTH_THRESHOLDS['crisis_hotline_vn']} and refer to human professional immediately. Do not self-diagnose.", True, kws
    if phq_cls in ("severe","moderately_severe") or gad_cls=="severe":
        return "severe", "Severe — recommend professional evaluation within days.", False, []
    if phq_cls=="moderate" or gad_cls=="moderate":
        return "moderate", "Moderate — consider professional support and follow-up.", False, []
    if phq_cls=="mild" or gad_cls=="mild":
        return "mild", "Mild — self-care + monitor.", False, []
    if phq_cls=="minimal" or gad_cls=="minimal":
        return "minimal", "Minimal — maintain wellness.", False, []
    return "unknown", "No scores.", False, []

def add_mood_log(user_id: str, phq9_score: Optional[int] = None, gad7_score: Optional[int] = None,
                 mood_notes: Optional[str] = None, measured_at: Optional[datetime] = None,
                 context: str = "random") -> Dict[str, Any]:
    init_mood_db()
    measured_at = measured_at or datetime.utcnow()
    # PII redact before storage
    redacted_notes = redact_pii(mood_notes)
    crisis, kws = contains_crisis_keywords(mood_notes)  # detect before redact to catch keywords
    # Also check redacted? keep original detection
    classification, message, is_crisis, found = classify_mood(phq9_score, gad7_score, mood_notes)
    # validate scores
    if phq9_score is not None and not (0 <= phq9_score <= 27):
        raise ValueError("PHQ-9 must be 0-27")
    if gad7_score is not None and not (0 <= gad7_score <= 21):
        raise ValueError("GAD-7 must be 0-21")
    conn=_get_conn()
    cur=conn.execute(
        "INSERT INTO mood_logs (user_id, phq9_score, gad7_score, mood_notes, crisis_flag, crisis_keywords, measured_at, context, classification, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (user_id, phq9_score, gad7_score, redacted_notes, int(is_crisis), ",".join(found), measured_at.isoformat(), context, classification, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id=cur.lastrowid
    conn.close()
    return {"id": row_id, "user_id": user_id, "phq9_score": phq9_score, "gad7_score": gad7_score,
            "mood_notes": redacted_notes, "original_notes_had_pii": redacted_notes != mood_notes,
            "classification": classification, "message": message, "crisis_flag": is_crisis, "crisis_keywords": found,
            "measured_at": measured_at, "context": context}

def get_mood_logs(user_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
    init_mood_db()
    conn=_get_conn()
    q="SELECT * FROM mood_logs WHERE user_id=? "
    params: List[Any]=[user_id]
    if days is not None:
        since=(datetime.utcnow()-timedelta(days=days)).isoformat()
        q+="AND measured_at >= ? "
        params.append(since)
    q+="ORDER BY measured_at DESC LIMIT ?"
    params.append(limit)
    rows=conn.execute(q, params).fetchall()
    conn.close()
    return [{k:r[k] for k in r.keys()} for r in rows]

def get_mood_stats(user_id: str) -> Dict[str, Any]:
    logs=get_mood_logs(user_id, limit=1000)
    total=len(logs)
    if total==0:
        return {"user_id": user_id, "total_logs":0, "avg_phq9":None, "avg_gad7":None, "crisis_count":0, "classification_counts":{}}
    phqs=[l["phq9_score"] for l in logs if l["phq9_score"] is not None]
    gads=[l["gad7_score"] for l in logs if l["gad7_score"] is not None]
    avg_phq=round(sum(phqs)/len(phqs),1) if phqs else None
    avg_gad=round(sum(gads)/len(gads),1) if gads else None
    crisis=sum(1 for l in logs if l["crisis_flag"])
    counts={}
    for l in logs:
        counts[l["classification"]]=counts.get(l["classification"],0)+1
    return {"user_id": user_id, "total_logs": total, "avg_phq9": avg_phq, "avg_gad7": avg_gad, "crisis_count": crisis, "classification_counts": counts}

def should_escalate_mood(user_id: str) -> bool:
    logs=get_mood_logs(user_id, limit=3)
    return any(l["crisis_flag"] for l in logs) or any(l["classification"]=="severe" for l in logs[:1])
