"""Learn store — learner_profile + lesson_progress (Micro-Curriculum).

2 tables (D2 pin): profile is 1 row/user, progress is N rows/user — merging
them would cause nullable-sprawl + update anomalies (precedent: labs 2 tables).
Separate file metadata/learn.db (LEARN_DB_PATH), never inside glucose/labs DBs.
Module-namespace patchable: tests monkeypatch src.learn.store.LEARN_DB_PATH.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.shared.config import LEARN_DB_PATH as _DEFAULT_LEARN_DB_PATH
    from src.vitals.base_tracker import get_conn, init_table
except ImportError:  # pragma: no cover
    from shared.config import LEARN_DB_PATH as _DEFAULT_LEARN_DB_PATH  # type: ignore
    from vitals.base_tracker import get_conn, init_table  # type: ignore

LEARN_DB_PATH = Path(os.getenv("LEARN_DB_PATH", str(_DEFAULT_LEARN_DB_PATH)))

PROFILE_DDL = """
CREATE TABLE IF NOT EXISTS learner_profile (
    user_id TEXT PRIMARY KEY,
    age INTEGER,
    meds TEXT DEFAULT '',
    has_meter INTEGER DEFAULT 0,
    day1_date TEXT DEFAULT '',
    mastery TEXT DEFAULT '{}',
    engagement TEXT DEFAULT 'normal',
    updated_at TEXT DEFAULT ''
)
"""

PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS lesson_progress (
    user_id TEXT NOT NULL,
    lesson_id TEXT NOT NULL,
    status TEXT DEFAULT 'not_started',
    attempts INTEGER DEFAULT 0,
    last_correct INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT '',
    PRIMARY KEY (user_id, lesson_id)
)
"""
PROGRESS_INDEX = "CREATE INDEX IF NOT EXISTS idx_learn_progress_user ON lesson_progress(user_id)"


def _db_path(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(LEARN_DB_PATH)


def _ensure_db(target=None) -> Path:
    """Create both tables on the target DB (precedent: labs _ensure_db)."""
    db = _db_path(target)
    init_table(db, PROFILE_DDL)
    init_table(db, PROGRESS_DDL, PROGRESS_INDEX)
    return db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_profile(user_id: str, db_path=None) -> dict[str, Any]:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        row = conn.execute(
            "SELECT * FROM learner_profile WHERE user_id=?", (user_id,)).fetchone()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    if row is None:
        return {}
    d = dict(row)
    try:
        d["mastery"] = json.loads(d.get("mastery") or "{}")
    except Exception:
        d["mastery"] = {}
    return d


def upsert_profile(user_id: str, patch: dict[str, Any], db_path=None) -> dict[str, Any]:
    db = _ensure_db(db_path)
    cur = get_profile(user_id, db_path=db)
    merged: dict[str, Any] = {
        "age": patch.get("age", cur.get("age")),
        "meds": patch.get("meds", cur.get("meds", "")),
        "has_meter": patch.get("has_meter", cur.get("has_meter", 0)),
        "day1_date": patch.get("day1_date", cur.get("day1_date", "")),
        "mastery": patch.get("mastery", cur.get("mastery", {})),
        "engagement": patch.get("engagement", cur.get("engagement", "normal")),
    }
    if isinstance(merged["mastery"], dict):
        mastery_s = json.dumps(merged["mastery"], ensure_ascii=False)
    else:
        mastery_s = str(merged["mastery"] or "{}")
    conn = get_conn(db)
    try:
        conn.execute(
            "INSERT INTO learner_profile (user_id, age, meds, has_meter, day1_date, mastery, engagement, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(user_id) DO UPDATE SET age=excluded.age, meds=excluded.meds,"
            " has_meter=excluded.has_meter, day1_date=excluded.day1_date,"
            " mastery=excluded.mastery, engagement=excluded.engagement,"
            " updated_at=excluded.updated_at",
            (user_id, merged["age"], merged["meds"], merged["has_meter"],
             merged["day1_date"], mastery_s, merged["engagement"], _now()))
        conn.commit()
    finally:
        conn.close()
    return get_profile(user_id, db_path=db)


def get_progress(user_id: str, db_path=None) -> list[dict[str, Any]]:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        rows = conn.execute(
            "SELECT * FROM lesson_progress WHERE user_id=? ORDER BY lesson_id",
            (user_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def record_quiz(user_id: str, lesson_id: str, correct: bool, db_path=None) -> dict[str, Any]:
    db = _ensure_db(db_path)
    conn = get_conn(db)
    try:
        row = conn.execute(
            "SELECT attempts FROM lesson_progress WHERE user_id=? AND lesson_id=?",
            (user_id, lesson_id)).fetchone()
        attempts = int(row["attempts"]) + 1 if row is not None else 1
        conn.execute(
            "INSERT INTO lesson_progress (user_id, lesson_id, status, attempts, last_correct, updated_at)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(user_id, lesson_id) DO UPDATE SET status='done',"
            " attempts=excluded.attempts, last_correct=excluded.last_correct,"
            " updated_at=excluded.updated_at",
            (user_id, lesson_id, "done", attempts, 1 if correct else 0, _now()))
        conn.commit()
    finally:
        conn.close()
    rows = get_progress(user_id, db_path=db)
    return next((r for r in rows if r["lesson_id"] == lesson_id), {})
