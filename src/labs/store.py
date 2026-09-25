"""SQLite store cho lab_reports + lab_observations (Nura B1).

Reuse base_tracker pattern (get_conn/init_table) — khong reinvent boilerplate.
DB rieng metadata/labs.db (khong nhoi vao glucose_logs.db 57MB).
Module-namespace patchable: tests monkeypatch src.labs.store.LAB_DB_PATH.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    from src.shared.config import LAB_DB_PATH as _DEFAULT_LAB_DB_PATH
    from src.vitals.base_tracker import get_conn, init_table
except ImportError:  # pragma: no cover
    from shared.config import LAB_DB_PATH as _DEFAULT_LAB_DB_PATH  # type: ignore
    from vitals.base_tracker import get_conn, init_table  # type: ignore

LAB_DB_PATH = Path(os.getenv("LAB_DB_PATH", str(_DEFAULT_LAB_DB_PATH)))

REPORT_DDL = """
CREATE TABLE IF NOT EXISTS lab_reports (
    report_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    sampled_at TEXT NOT NULL,
    facility TEXT,
    visit_no TEXT
)
"""
REPORT_INDEX = "CREATE INDEX IF NOT EXISTS idx_lab_reports_user_time ON lab_reports(user_id, sampled_at)"

OBS_DDL = """
CREATE TABLE IF NOT EXISTS lab_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    loinc TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    ref_low REAL,
    ref_high REAL,
    flag TEXT
)
"""
OBS_INDEX = "CREATE INDEX IF NOT EXISTS idx_lab_obs_report ON lab_observations(report_id)"


def _db_path(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(LAB_DB_PATH)


def _ensure_db(target=None) -> Path:
    """Tao 2 tables dung tren DB dich (precedent seed_synthetic fix)."""
    db = _db_path(target)
    init_table(db, REPORT_DDL, REPORT_INDEX)
    init_table(db, OBS_DDL, OBS_INDEX)
    return db


def create_report(report_id: str, user_id: str, sampled_at: str,
                  facility: str = "", visit_no: str = "", db_path=None) -> None:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO lab_reports (report_id, user_id, sampled_at, facility, visit_no)"
            " VALUES (?,?,?,?,?)", (report_id, user_id, sampled_at, facility, visit_no))
        conn.commit()
    finally:
        conn.close()


def add_observation(report_id: str, loinc: str, name: str, value: float,
                    unit: str, ref_low: float | None = None,
                    ref_high: float | None = None, flag: str = "",
                    db_path=None) -> int:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        cur = conn.execute(
            "INSERT INTO lab_observations (report_id, loinc, name, value, unit, ref_low, ref_high, flag)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (report_id, loinc, name, value, unit, ref_low, ref_high, flag))
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def get_reports(user_id: str, db_path=None) -> list[dict[str, Any]]:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        rows = conn.execute(
            "SELECT * FROM lab_reports WHERE user_id=? ORDER BY sampled_at DESC",
            (user_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_report(report_id: str, db_path=None) -> dict[str, Any] | None:
    """1 report theo id (read-only; None khi khong thay)."""
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        row = conn.execute(
            "SELECT * FROM lab_reports WHERE report_id=?",
            (report_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return dict(row) if row is not None else None


def get_observations(report_id: str, db_path=None) -> list[dict[str, Any]]:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        rows = conn.execute(
            "SELECT * FROM lab_observations WHERE report_id=? ORDER BY loinc",
            (report_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_history(user_id: str, loinc: str, db_path=None) -> list[dict[str, Any]]:
    """Lich su 1 chi so theo user (join reports + observations, sort time ASC)."""
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        rows = conn.execute(
            "SELECT o.*, r.sampled_at FROM lab_observations o"
            " JOIN lab_reports r ON r.report_id=o.report_id"
            " WHERE r.user_id=? AND o.loinc=? ORDER BY r.sampled_at ASC",
            (user_id, loinc)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def count_reports(db_path=None) -> int:
    db = _ensure_db(db_path)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT COUNT(*) FROM lab_reports").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def count_observations(db_path=None) -> int:
    db = _ensure_db(db_path)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT COUNT(*) FROM lab_observations").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()
