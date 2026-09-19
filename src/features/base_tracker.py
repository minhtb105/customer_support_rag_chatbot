"""
BaseTracker — shared SQLite CRUD helpers for 4 vitals trackers (C1 deduplication).

Before: 4 files duplicated _get_conn / get_logs / get_stats streak/loops (~90 lines each).
After: single source of truth, each tracker only defines schema + classification + disease-specific stats.

Usage:
    from src.features.base_tracker import get_conn, init_table, fetch_logs, classification_counts, calc_streak, calc_logs_per_week
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.config import VITALS_DB_PATH, GLUCOSE_DB_PATH
except ImportError:  # pragma: no cover
    from config import VITALS_DB_PATH, GLUCOSE_DB_PATH  # type: ignore


# ---------- low-level ----------

def get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_table(db_path: Path, ddl: str, index_sql: Optional[str] = None) -> None:
    conn = get_conn(db_path)
    conn.execute(ddl)
    if index_sql:
        conn.execute(index_sql)
    conn.commit()
    conn.close()


def fetch_logs(db_path: Path, table: str, user_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generic SELECT * FROM {table} WHERE user_id=? [AND measured_at >= ?] ORDER BY measured_at DESC LIMIT ?"""
    # Caller should have called init_table already, but ensure table exists lazily
    q = f"SELECT * FROM {table} WHERE user_id=? "
    params: List[Any] = [user_id]
    if days is not None:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        q += "AND measured_at >= ? "
        params.append(since)
    q += "ORDER BY measured_at DESC LIMIT ?"
    params.append(limit)
    conn = get_conn(db_path)
    try:
        rows = conn.execute(q, params).fetchall()
    except sqlite3.OperationalError:
        # table not yet created (first call before init) -> return empty
        return []
    finally:
        conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]


# ---------- stats helpers (C1/C2 dedup) ----------

def classification_counts(logs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for l in logs:
        c = l.get("classification", "unknown")
        counts[c] = counts.get(c, 0) + 1
    return counts


def calc_streak(logs: List[Dict[str, Any]]) -> int:
    """Số ngày liên tiếp có ít nhất 1 log, tính từ hôm nay ngược lại."""
    if not logs:
        return 0
    dates = sorted({str(l["measured_at"])[:10] for l in logs}, reverse=True)
    date_set = set(dates)
    streak = 0
    cur = datetime.utcnow().date()
    while cur.isoformat() in date_set:
        streak += 1
        cur -= timedelta(days=1)
    return streak


def calc_logs_per_week(logs: List[Dict[str, Any]]) -> float:
    """Trung bình 4 tuần gần nhất; fallback nếu ít dữ liệu."""
    if not logs:
        return 0.0
    last28 = [l for l in logs if str(l["measured_at"]) >= (datetime.utcnow() - timedelta(days=28)).isoformat()]
    if last28:
        return round(len(last28) / 4.0, 2)
    # fallback: total / weeks since earliest log
    try:
        earliest = datetime.fromisoformat(str(logs[-1]["measured_at"]))
        weeks = max(1, (datetime.utcnow() - earliest).days / 7)
        return round(len(logs) / weeks, 2)
    except Exception:
        return round(float(len(logs)), 2)
