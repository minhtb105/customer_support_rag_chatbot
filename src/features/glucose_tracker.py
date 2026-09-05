"""
Hướng A — Trợ lý tuân thủ tự theo dõi đường huyết.

- Lưu log vào SQLite (metadata/glucose_logs.db)
- Phân loại ngưỡng WHO/ADA (fasting, postprandial)
- Thống kê, streak, logs_per_week (KPI: ≥3 lần/tuần)
- Gợi ý RAG: chỉ escalate khi critical / 3 lần high liên tiếp
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

try:
    from src.config import GLUCOSE_THRESHOLDS_MGDL, GLUCOSE_DB_PATH
except ImportError:
    from config import GLUCOSE_THRESHOLDS_MGDL, GLUCOSE_DB_PATH  # type: ignore

# ---------- DB ----------

def _get_conn() -> sqlite3.Connection:
    GLUCOSE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(GLUCOSE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_glucose_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS glucose_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            value_mgdl REAL NOT NULL,
            measured_at TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'random',
            notes TEXT,
            classification TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_glucose_user_time ON glucose_logs(user_id, measured_at)")
    conn.commit()
    conn.close()


# ---------- Classification ----------

def classify_glucose(value_mgdl: float, context: str) -> Tuple[str, str]:
    """
    Returns (classification, message).
    classification in: low | normal | elevated | high | critical
    """
    t = GLUCOSE_THRESHOLDS_MGDL
    if value_mgdl < t["hypoglycemia"]:
        return "low", f"Hạ đường huyết (<{t['hypoglycemia']} mg/dL) — cần xử trí ngay, ăn nhẹ và theo dõi."
    if value_mgdl >= t["random_critical_high"]:
        return "critical", f"Rất cao (≥{t['random_critical_high']} mg/dL) — nguy cơ cấp cứu, liên hệ bác sĩ ngay."
    if context == "fasting":
        if value_mgdl < t["fasting_normal_max"]:
            return "normal", "Bình thường lúc đói."
        elif value_mgdl <= t["fasting_prediabetes_max"]:
            return "elevated", "Tiền đái tháo đường lúc đói (100-125) — cần theo dõi sát."
        else:
            return "high", "Cao lúc đói (≥126) — vượt ngưỡng chẩn đoán ĐTĐ."
    elif context == "post_meal_2h":
        if value_mgdl < t["postprandial_normal_max"]:
            return "normal", "Bình thường sau ăn 2h."
        elif value_mgdl <= t["postprandial_prediabetes_max"]:
            return "elevated", "Tiền ĐTĐ sau ăn (140-199) — cần điều chỉnh chế độ."
        else:
            return "high", "Cao sau ăn 2h (≥200) — vượt ngưỡng."
    else:
        # random / pre_meal / bedtime: dùng ngưỡng chung
        if value_mgdl < 140:
            return "normal", "Trong ngưỡng tham chiếu."
        elif value_mgdl < 200:
            return "elevated", "Tăng nhẹ — theo dõi thêm."
        else:
            return "high", "Cao — cần chú ý."


def should_escalate_to_doctor(user_id: str, lookback: int = 3) -> bool:
    """Escalate nếu 1 critical hoặc `lookback` lần high liên tiếp gần nhất."""
    logs = get_logs(user_id, limit=lookback)
    if not logs:
        return False
    if any(l["classification"] == "critical" or l["classification"] == "low" for l in logs):
        return True
    if len(logs) >= lookback and all(l["classification"] == "high" for l in logs[:lookback]):
        return True
    return False


# ---------- CRUD ----------

def add_log(user_id: str, value_mgdl: float, measured_at: Optional[datetime] = None,
            context: str = "random", notes: Optional[str] = None) -> Dict[str, Any]:
    init_glucose_db()
    measured_at = measured_at or datetime.utcnow()
    classification, message = classify_glucose(value_mgdl, context)
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO glucose_logs (user_id, value_mgdl, measured_at, context, notes, classification, created_at) VALUES (?,?,?,?,?,?,?)",
        (user_id, value_mgdl, measured_at.isoformat(), context, notes, classification, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {
        "id": row_id,
        "user_id": user_id,
        "value_mgdl": value_mgdl,
        "measured_at": measured_at,
        "context": context,
        "notes": notes,
        "classification": classification,
        "message": message,
    }


def get_logs(user_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
    init_glucose_db()
    conn = _get_conn()
    q = "SELECT * FROM glucose_logs WHERE user_id=? "
    params: List[Any] = [user_id]
    if days is not None:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        q += "AND measured_at >= ? "
        params.append(since)
    q += "ORDER BY measured_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "value_mgdl": r["value_mgdl"],
            "measured_at": r["measured_at"],
            "context": r["context"],
            "notes": r["notes"],
            "classification": r["classification"],
        })
    return out


def get_stats(user_id: str) -> Dict[str, Any]:
    logs = get_logs(user_id, limit=1000)
    total = len(logs)
    if total == 0:
        return {
            "user_id": user_id,
            "total_logs": 0,
            "avg_mgdl": None,
            "last_7_days_avg": None,
            "streak_days": 0,
            "logs_per_week": 0.0,
            "classification_counts": {},
        }
    vals = [l["value_mgdl"] for l in logs]
    avg = sum(vals) / len(vals)
    # last 7 days
    seven = [l for l in logs if l["measured_at"] >= (datetime.utcnow() - timedelta(days=7)).isoformat()]
    last7_avg = sum(l["value_mgdl"] for l in seven) / len(seven) if seven else None
    # classification counts
    counts: Dict[str, int] = {}
    for l in logs:
        c = l.get("classification", "unknown")
        counts[c] = counts.get(c, 0) + 1
    # streak_days: số ngày liên tiếp có ít nhất 1 log (tính từ hôm nay ngược lại)
    dates = sorted({l["measured_at"][:10] for l in logs}, reverse=True)
    streak = 0
    cur = datetime.utcnow().date()
    date_set = set(dates)
    while cur.isoformat() in date_set:
        streak += 1
        cur -= timedelta(days=1)
    # logs_per_week: trung bình 4 tuần gần nhất
    last28 = [l for l in logs if l["measured_at"] >= (datetime.utcnow() - timedelta(days=28)).isoformat()]
    logs_per_week = len(last28) / 4.0 if last28 else (total / max(1, (datetime.utcnow() - datetime.fromisoformat(logs[-1]["measured_at"])).days / 7))
    return {
        "user_id": user_id,
        "total_logs": total,
        "avg_mgdl": round(avg, 1),
        "last_7_days_avg": round(last7_avg, 1) if last7_avg else None,
        "streak_days": streak,
        "logs_per_week": round(logs_per_week, 2),
        "classification_counts": counts,
    }


def needs_attention_summary(user_id: str) -> str:
    stats = get_stats(user_id)
    escalate = should_escalate_to_doctor(user_id)
    logs = get_logs(user_id, limit=7)
    if not logs:
        return "Chưa có dữ liệu đường huyết — hãy nhập lần đo đầu tiên."
    last = logs[0]
    base = f"Lần đo gần nhất: {last['value_mgdl']} mg/dL ({last['classification']}) lúc {last['measured_at'][:16]} [{last['context']}]. "
    base += f"Trung bình 7 ngày: {stats['last_7_days_avg'] or '—'} mg/dL. Tần suất: {stats['logs_per_week']}/tuần. "
    if escalate:
        base += "⚠️ Khuyến nghị liên hệ bác sĩ (vượt ngưỡng critical hoặc 3 lần high liên tiếp)."
    elif stats["logs_per_week"] < 3:
        base += "Gợi ý: duy trì ≥3 lần/tuần để theo dõi sát (KPI Hướng A)."
    return base
