"""Auth DB — SQLite metadata/auth.db : users, refresh_tokens, notifications"""
from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from src.config import AUTH_DB_PATH
except ImportError:
    from config import AUTH_DB_PATH  # type: ignore

def _get_conn() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL for concurrency
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn

def init_auth_db():
    conn = _get_conn()
    # users
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL CHECK(role IN ('user','doctor','pharmacist','specialist','admin')),
            is_active INTEGER DEFAULT 1,
            is_verified INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
    # refresh_tokens
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            revoked INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
    # review_requests — HILT queue
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_requests (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            draft_answer TEXT NOT NULL,
            contexts_json TEXT NOT NULL,
            confidence_json TEXT,
            confidence REAL,
            confidence_reason TEXT,
            failed_metrics TEXT,
            routed_role TEXT,
            status TEXT CHECK(status IN ('pending','approved','rejected','revised')) DEFAULT 'pending',
            requester_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            assigned_expert_id TEXT REFERENCES users(id),
            disease TEXT,
            final_answer TEXT,
            expert_notes TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            langsmith_run_id TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON review_requests(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_review_requester ON review_requests(requester_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_review_routed ON review_requests(routed_role)")
    # notifications — for polling
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            review_id TEXT REFERENCES review_requests(id) ON DELETE SET NULL,
            is_read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, is_read)")
    # query_history — optional denormalized cache for approved reviews linked to query
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            status TEXT DEFAULT 'answered',
            review_id TEXT REFERENCES review_requests(id),
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qh_user ON query_history(user_id)")
    conn.commit()
    conn.close()

# ---------- users helpers ----------
def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None

def list_users(limit: int = 100, offset: int = 0, role: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_conn()
    if role:
        rows = conn.execute("SELECT * FROM users WHERE role=? ORDER BY created_at DESC LIMIT ? OFFSET ?", (role, limit, offset)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_user(username: str, email: Optional[str], hashed_password: str, full_name: Optional[str], role: str, is_verified: bool = True) -> Dict[str, Any]:
    uid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    # expert created by admin defaults to unverified if caller wants
    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (id, username, email, hashed_password, full_name, role, is_active, is_verified, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (uid, username, email, hashed_password, full_name, role, 1, 1 if is_verified else 0, now)
    )
    conn.commit()
    conn.close()
    return get_user_by_id(uid)  # type: ignore

def update_user(user_id: str, **fields) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_user_by_id(user_id)
    allowed = {"email", "full_name", "role", "is_active", "is_verified", "hashed_password", "last_login"}
    sets = []
    params: List[Any] = []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            params.append(v)
    if not sets:
        return get_user_by_id(user_id)
    params.append(user_id)
    conn = _get_conn()
    conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", params)
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)

def delete_user(user_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

# ---------- refresh_tokens ----------
def store_refresh_token(token: str, user_id: str, expires_at: datetime):
    conn = _get_conn()
    conn.execute("INSERT INTO refresh_tokens (token, user_id, expires_at, revoked, created_at) VALUES (?,?,?,?,?)",
                 (token, user_id, expires_at.isoformat(), 0, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM refresh_tokens WHERE token=?", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None

def revoke_refresh_token(token: str):
    conn = _get_conn()
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE token=?", (token,))
    conn.commit()
    conn.close()

def revoke_all_user_tokens(user_id: str):
    conn = _get_conn()
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

# ---------- notifications ----------
def create_notification(user_id: str, type: str, title: str, body: str = "", review_id: Optional[str] = None) -> str:
    nid = uuid.uuid4().hex
    conn = _get_conn()
    conn.execute("INSERT INTO notifications (id, user_id, type, title, body, review_id, is_read, created_at) VALUES (?,?,?,?,?,?,?,?)",
                 (nid, user_id, type, title, body, review_id, 0, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return nid

def list_notifications(user_id: str, limit: int = 50, unread_only: bool = False) -> List[Dict[str, Any]]:
    conn = _get_conn()
    if unread_only:
        rows = conn.execute("SELECT * FROM notifications WHERE user_id=? AND is_read=0 ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def count_unread(user_id: str) -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as c FROM notifications WHERE user_id=? AND is_read=0", (user_id,)).fetchone()
    conn.close()
    return int(row["c"]) if row else 0

def mark_notification_read(notif_id: str, user_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("UPDATE notifications SET is_read=1 WHERE id=? AND user_id=?", (notif_id, user_id))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

def mark_all_read(user_id: str):
    conn = _get_conn()
    conn.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

# ---------- query_history ----------
def add_query_history(user_id: str, query: str, answer: str, status: str = "answered", review_id: Optional[str] = None, confidence: Optional[float] = None) -> str:
    qid = uuid.uuid4().hex
    conn = _get_conn()
    conn.execute("INSERT INTO query_history (id, user_id, query, answer, status, review_id, confidence, created_at) VALUES (?,?,?,?,?,?,?,?)",
                 (qid, user_id, query, answer, status, review_id, confidence, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return qid

def list_query_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM query_history WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_query_history_by_review(review_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM query_history WHERE review_id=? ORDER BY created_at DESC LIMIT 1", (review_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_query_history_answer(review_id: str, final_answer: str):
    conn = _get_conn()
    conn.execute("UPDATE query_history SET answer=?, status='approved' WHERE review_id=?", (final_answer, review_id))
    conn.commit()
    conn.close()

# init on import
try:
    init_auth_db()
except Exception:
    pass
