"""Monitoring DB — SQLite metadata/monitoring.db"""
from __future__ import annotations
import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from src.config import BASE_DIR
except ImportError:
    from config import BASE_DIR  # type: ignore

MONITORING_DB_PATH = BASE_DIR / "metadata" / "monitoring.db"

# Allowlist for crawler SSRF protection
ALLOWED_MONITOR_DOMAINS = {
    "iris.who.int",
    "diabetesjournals.org",
    "ada-journals.cld.bz",
    "www.ahajournals.org",
    "ahajournals.org",
    "goldcopd.org",
    "ginasthma.org",
    "api.fda.gov",
    "www.fda.gov",
    "moh.gov.vn",
    "dav.gov.vn",
    "thuvienphapluat.vn",
    "kcb.vn",
    "www.who.int",
}

def _conn() -> sqlite3.Connection:
    MONITORING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(MONITORING_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return c

def init_monitoring_db():
    conn = _conn()
    # monitored_sources
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitored_sources (
            id TEXT PRIMARY KEY,
            source_key TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            check_interval TEXT NOT NULL,
            risk_tier TEXT,
            last_checked_at TEXT,
            last_etag TEXT,
            last_modified TEXT,
            last_hash TEXT,
            enabled INTEGER DEFAULT 1,
            config_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # guideline_versions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guideline_versions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            version_label TEXT,
            publication_date TEXT,
            fetched_at TEXT NOT NULL,
            sha256 TEXT,
            etag TEXT,
            staging_path TEXT,
            corpus_path TEXT,
            status TEXT CHECK(status IN ('pending_review','approved','rejected','superseded','archived')) DEFAULT 'pending_review',
            change_summary_json TEXT,
            diff_text TEXT,
            reviewer_id TEXT,
            reviewed_at TEXT,
            review_notes TEXT,
            supersedes_id TEXT,
            indexed_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gv_source ON guideline_versions(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gv_status ON guideline_versions(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gv_fetched ON guideline_versions(fetched_at)")
    # safety_alerts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS safety_alerts (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT CHECK(severity IN ('critical','high','medium','low')) DEFAULT 'medium',
            drug_name TEXT,
            alert_title TEXT NOT NULL,
            alert_url TEXT,
            published_date TEXT,
            fetched_at TEXT NOT NULL,
            raw_json TEXT,
            ai_summary TEXT,
            ai_risk_score REAL,
            status TEXT CHECK(status IN ('pending_review','approved','dismissed','archived')) DEFAULT 'pending_review',
            assigned_role TEXT DEFAULT 'pharmacist',
            reviewer_id TEXT,
            reviewed_at TEXT,
            review_notes TEXT,
            is_notified INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sa_status ON safety_alerts(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sa_severity ON safety_alerts(severity)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sa_source ON safety_alerts(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sa_fetched ON safety_alerts(fetched_at)")
    # monitor_runs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitor_runs (
            id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            found_new INTEGER DEFAULT 0,
            status TEXT CHECK(status IN ('running','success','failed')) DEFAULT 'running',
            error TEXT,
            trace_id TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_source ON monitor_runs(source_key, started_at)")
    conn.commit()
    # seed monitored_sources if empty
    cnt = conn.execute("SELECT COUNT(*) as c FROM monitored_sources").fetchone()["c"]
    if cnt == 0:
        now = datetime.utcnow().isoformat()
        seeds = [
            ("ada_soc", "ADA Standards of Care (Diabetes)", "https://diabetesjournals.org/care/issue/47/Supplement_1", "monthly", "high"),
            ("aha_acc_htn", "AHA/ACC Hypertension Guideline", "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065", "monthly", "high"),
            ("gold", "GOLD COPD Report", "https://goldcopd.org/2024-gold-report/", "monthly", "high"),
            ("gina", "GINA Asthma Strategy", "https://ginasthma.org/2024-gina-main-report/", "monthly", "high"),
            ("who_mhgap", "WHO mhGAP Intervention Guide", "https://iris.who.int/handle/10665/250239", "monthly", "medium"),
            ("who_diabetes", "WHO Diabetes Classification/HEARTS-D", "https://iris.who.int/handle/10665/325182", "monthly", "high"),
            ("byt_diabetes", "BYT — QĐ Đái tháo đường (BYT)", "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-3319-QD-BYT-2017-huong-dan-chan-doan-dieu-tri-dai-thao-duong-356066.aspx", "weekly", "high"),
            ("fda_recall", "FDA Drug Recalls (openFDA enforcement)", "https://api.fda.gov/drug/enforcement.json", "daily", "critical"),
            ("dav_thuhoi", "DAV — Cục Quản lý Dược Thu hồi", "https://dav.gov.vn", "weekly", "critical"),
            ("moh_canhbao", "MOH — Cảnh báo Dược", "https://moh.gov.vn", "weekly", "high"),
        ]
        for key, name, url, interval, tier in seeds:
            conn.execute(
                "INSERT INTO monitored_sources (id, source_key, display_name, base_url, check_interval, risk_tier, enabled, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, key, name, url, interval, tier, 1, now),
            )
        conn.commit()
    conn.close()

# ---------- monitored_sources helpers ----------
def list_sources(enabled_only: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    if enabled_only:
        rows = conn.execute("SELECT * FROM monitored_sources WHERE enabled=1 ORDER BY source_key").fetchall()
    else:
        rows = conn.execute("SELECT * FROM monitored_sources ORDER BY source_key").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_source(source_key: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM monitored_sources WHERE source_key=?", (source_key,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_source_check(source_key: str, etag: Optional[str] = None, last_modified: Optional[str] = None, last_hash: Optional[str] = None):
    conn = _conn()
    now = datetime.utcnow().isoformat()
    # build dynamic update
    fields = {"last_checked_at": now}
    if etag is not None:
        fields["last_etag"] = etag
    if last_modified is not None:
        fields["last_modified"] = last_modified
    if last_hash is not None:
        fields["last_hash"] = last_hash
    sets = ", ".join([f"{k}=?" for k in fields])
    vals = list(fields.values()) + [source_key]
    conn.execute(f"UPDATE monitored_sources SET {sets} WHERE source_key=?", vals)
    conn.commit()
    conn.close()

# ---------- guideline_versions ----------
def create_guideline_version(
    source: str,
    title: str,
    url: str,
    version_label: Optional[str] = None,
    publication_date: Optional[str] = None,
    sha256: Optional[str] = None,
    etag: Optional[str] = None,
    staging_path: Optional[str] = None,
    change_summary_json: Optional[str] = None,
    diff_text: Optional[str] = None,
    supersedes_id: Optional[str] = None,
) -> Dict[str, Any]:
    gid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute(
        """INSERT INTO guideline_versions
        (id, source, title, url, version_label, publication_date, fetched_at, sha256, etag, staging_path, status, change_summary_json, diff_text, supersedes_id, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (gid, source, title, url, version_label, publication_date, now, sha256, etag, staging_path, "pending_review", change_summary_json, diff_text, supersedes_id, now),
    )
    conn.commit()
    conn.close()
    return get_guideline_version(gid)  # type: ignore

def get_guideline_version(gid: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM guideline_versions WHERE id=?", (gid,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for k in ("change_summary_json",):
        if d.get(k):
            try:
                d[k+"_parsed"] = json.loads(d[k])
            except Exception:
                pass
    return d

def list_guideline_versions(
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], int]:
    conn = _conn()
    where = ["1=1"]
    params: List[Any] = []
    if status:
        where.append("status=?")
        params.append(status)
    if source:
        where.append("source=?")
        params.append(source)
    where_str = " AND ".join(where)
    rows = conn.execute(f"SELECT * FROM guideline_versions WHERE {where_str} ORDER BY fetched_at DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
    total = conn.execute(f"SELECT COUNT(*) as c FROM guideline_versions WHERE {where_str}", params).fetchone()["c"]
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("change_summary_json"):
            try:
                d["change_summary_parsed"] = json.loads(d["change_summary_json"])
            except Exception:
                pass
        out.append(d)
    return out, total

def update_guideline_status(gid: str, status: str, reviewer_id: Optional[str] = None, review_notes: Optional[str] = None, corpus_path: Optional[str] = None, indexed_at: Optional[str] = None):
    conn = _conn()
    now = datetime.utcnow().isoformat()
    fields: Dict[str, Any] = {"status": status, "reviewed_at": now}
    if reviewer_id:
        fields["reviewer_id"] = reviewer_id
    if review_notes is not None:
        fields["review_notes"] = review_notes
    if corpus_path:
        fields["corpus_path"] = corpus_path
    if indexed_at:
        fields["indexed_at"] = indexed_at
    sets = ", ".join([f"{k}=?" for k in fields])
    vals = list(fields.values()) + [gid]
    conn.execute(f"UPDATE guideline_versions SET {sets} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return get_guideline_version(gid)

def delete_superseded_expired(days: int = 30) -> int:
    """Delete superseded guideline_versions older than days and remove files."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    conn = _conn()
    rows = conn.execute("SELECT * FROM guideline_versions WHERE status='superseded' AND reviewed_at < ?", (cutoff,)).fetchall()
    count = 0
    for r in rows:
        d = dict(r)
        # remove files if exist
        for pkey in ("staging_path", "corpus_path"):
            p = d.get(pkey)
            if p:
                try:
                    pp = Path(p)
                    if pp.exists():
                        if pp.is_file():
                            pp.unlink()
                        elif pp.is_dir():
                            import shutil
                            shutil.rmtree(pp, ignore_errors=True)
                except Exception:
                    pass
        conn.execute("DELETE FROM guideline_versions WHERE id=?", (d["id"],))
        count += 1
    # also handle archived
    conn.commit()
    conn.close()
    return count

# ---------- safety_alerts ----------
def create_safety_alert(
    source: str,
    alert_type: str,
    alert_title: str,
    severity: str = "medium",
    drug_name: Optional[str] = None,
    alert_url: Optional[str] = None,
    published_date: Optional[str] = None,
    raw_json: Optional[str] = None,
    ai_summary: Optional[str] = None,
    ai_risk_score: Optional[float] = None,
    assigned_role: str = "pharmacist",
) -> Dict[str, Any]:
    # deduplicate by alert_url + title within 7 days
    conn = _conn()
    # check existing url
    if alert_url:
        existing = conn.execute("SELECT id FROM safety_alerts WHERE alert_url=? AND created_at > datetime('now','-7 days')", (alert_url,)).fetchone()
        if existing:
            conn.close()
            return get_safety_alert(existing["id"])  # type: ignore
    # also check title+drug duplicate
    if drug_name and alert_title:
        dup = conn.execute("SELECT id FROM safety_alerts WHERE drug_name=? AND alert_title=? AND created_at > datetime('now','-7 days')", (drug_name, alert_title)).fetchone()
        if dup:
            conn.close()
            return get_safety_alert(dup["id"])  # type: ignore
    aid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    published = published_date or now
    conn.execute(
        """INSERT INTO safety_alerts
        (id, source, alert_type, severity, drug_name, alert_title, alert_url, published_date, fetched_at, raw_json, ai_summary, ai_risk_score, status, assigned_role, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (aid, source, alert_type, severity, drug_name, alert_title, alert_url, published, now, raw_json, ai_summary, ai_risk_score, "pending_review", assigned_role, now),
    )
    conn.commit()
    conn.close()
    return get_safety_alert(aid)  # type: ignore

def get_safety_alert(aid: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM safety_alerts WHERE id=?", (aid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def list_safety_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], int]:
    conn = _conn()
    where = ["1=1"]
    params: List[Any] = []
    if status:
        where.append("status=?")
        params.append(status)
    if severity:
        where.append("severity=?")
        params.append(severity)
    if source:
        where.append("source=?")
        params.append(source)
    where_str = " AND ".join(where)
    rows = conn.execute(f"SELECT * FROM safety_alerts WHERE {where_str} ORDER BY fetched_at DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
    total = conn.execute(f"SELECT COUNT(*) as c FROM safety_alerts WHERE {where_str}", params).fetchone()["c"]
    conn.close()
    return [dict(r) for r in rows], total

def update_safety_status(aid: str, status: str, reviewer_id: Optional[str] = None, review_notes: Optional[str] = None):
    conn = _conn()
    now = datetime.utcnow().isoformat()
    fields: Dict[str, Any] = {"status": status, "reviewed_at": now}
    if reviewer_id:
        fields["reviewer_id"] = reviewer_id
    if review_notes is not None:
        fields["review_notes"] = review_notes
    sets = ", ".join([f"{k}=?" for k in fields])
    vals = list(fields.values()) + [aid]
    conn.execute(f"UPDATE safety_alerts SET {sets} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return get_safety_alert(aid)

def mark_safety_notified(aid: str):
    conn = _conn()
    conn.execute("UPDATE safety_alerts SET is_notified=1 WHERE id=?", (aid,))
    conn.commit()
    conn.close()

# ---------- monitor_runs ----------
def create_run(source_key: str, trace_id: Optional[str] = None) -> str:
    rid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute("INSERT INTO monitor_runs (id, source_key, started_at, status, trace_id) VALUES (?,?,?, ?,?)", (rid, source_key, now, "running", trace_id))
    conn.commit()
    conn.close()
    return rid

def finish_run(run_id: str, found_new: int = 0, status: str = "success", error: Optional[str] = None):
    conn = _conn()
    now = datetime.utcnow().isoformat()
    conn.execute("UPDATE monitor_runs SET finished_at=?, found_new=?, status=?, error=? WHERE id=?", (now, found_new, status, error, run_id))
    conn.commit()
    conn.close()

def list_runs(source_key: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    conn = _conn()
    if source_key:
        rows = conn.execute("SELECT * FROM monitor_runs WHERE source_key=? ORDER BY started_at DESC LIMIT ?", (source_key, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM monitor_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_stats() -> Dict[str, Any]:
    conn = _conn()
    pending_g = conn.execute("SELECT COUNT(*) as c FROM guideline_versions WHERE status='pending_review'").fetchone()["c"]
    pending_s = conn.execute("SELECT COUNT(*) as c FROM safety_alerts WHERE status='pending_review'").fetchone()["c"]
    total_g = conn.execute("SELECT COUNT(*) as c FROM guideline_versions").fetchone()["c"]
    total_s = conn.execute("SELECT COUNT(*) as c FROM safety_alerts").fetchone()["c"]
    conn.close()
    return {"pending_guidelines": pending_g, "pending_alerts": pending_s, "total_guidelines": total_g, "total_alerts": total_s}

# init on import
try:
    init_monitoring_db()
except Exception:
    pass
