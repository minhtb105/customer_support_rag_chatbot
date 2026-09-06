"""Tracing DB — SQLite metadata/tracing.db"""
from __future__ import annotations
import sqlite3
import uuid
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from src.config import BASE_DIR
except ImportError:
    from config import BASE_DIR  # type: ignore

TRACING_DB_PATH = BASE_DIR / "metadata" / "tracing.db"

def _conn() -> sqlite3.Connection:
    TRACING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(TRACING_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return c

def init_tracing_db():
    conn = _conn()
    # traces
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            username TEXT,
            query TEXT NOT NULL,
            answer TEXT,
            status TEXT CHECK(status IN ('pending_review','answered','failed')) DEFAULT 'answered',
            tone TEXT,
            prompt_version TEXT,
            prompt_id TEXT,
            model TEXT,
            embedding_model TEXT,
            chunking_strategy TEXT,
            top_k INTEGER,
            vector_hits INTEGER,
            bm25_hits INTEGER,
            total_latency_ms REAL,
            is_low_confidence BOOLEAN,
            routed_role TEXT,
            review_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_user ON traces(user_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)")
    # spans
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            id TEXT PRIMARY KEY,
            trace_id TEXT REFERENCES traces(id) ON DELETE CASCADE,
            parent_id TEXT,
            name TEXT NOT NULL,
            start_at TEXT,
            end_at TEXT,
            duration_ms REAL,
            inputs_json TEXT,
            outputs_json TEXT,
            metadata_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)")
    # span_chunks
    conn.execute("""
        CREATE TABLE IF NOT EXISTS span_chunks (
            id TEXT PRIMARY KEY,
            span_id TEXT REFERENCES spans(id) ON DELETE CASCADE,
            trace_id TEXT REFERENCES traces(id) ON DELETE CASCADE,
            rank INTEGER,
            source_id TEXT,
            file_name TEXT,
            dataset TEXT,
            section_path TEXT,
            page_numbers TEXT,
            chunk_index INTEGER,
            chunk_hash TEXT,
            chunking_strategy TEXT,
            embedding_model TEXT,
            updated_at TEXT,
            score REAL,
            content_snippet TEXT,
            content_full TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_span ON span_chunks(span_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_trace ON span_chunks(trace_id)")
    # prompts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            tone TEXT NOT NULL,
            version TEXT NOT NULL,
            text TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT,
            is_active BOOLEAN DEFAULT 0,
            parent_version TEXT,
            status TEXT CHECK(status IN ('draft','pending_approval','active','archived')) DEFAULT 'draft'
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_tone_version ON prompts(tone, version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompts(tone, is_active)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_status ON prompts(status)")
    # ragas_evaluations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ragas_evaluations (
            id TEXT PRIMARY KEY,
            trace_id TEXT UNIQUE REFERENCES traces(id) ON DELETE CASCADE,
            faithfulness REAL, context_precision REAL, context_recall REAL, answer_relevance REAL, fluency REAL,
            faithfulness_comment TEXT, precision_comment TEXT, recall_comment TEXT, relevance_comment TEXT, fluency_comment TEXT,
            failed_metrics TEXT,
            confidence REAL,
            raw_json TEXT,
            evaluated_at TEXT,
            evaluator_model TEXT,
            is_pending BOOLEAN DEFAULT 0
        )
    """)
    # feedback
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            trace_id TEXT REFERENCES traces(id) ON DELETE CASCADE,
            user_id TEXT,
            key TEXT, score REAL, value TEXT, comment TEXT, created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

# ---------- traces ----------
def create_trace(user_id: str, username: Optional[str], query: str, tone: Optional[str], model: Optional[str], embedding_model: Optional[str], chunking_strategy: Optional[str], top_k: Optional[int]) -> str:
    tid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute("INSERT INTO traces (id, user_id, username, query, tone, model, embedding_model, chunking_strategy, top_k, created_at, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (tid, user_id, username, query, tone, model, embedding_model, chunking_strategy, top_k, now, "answered"))
    conn.commit()
    conn.close()
    return tid

def update_trace(trace_id: str, **fields):
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [trace_id]
    conn = _conn()
    conn.execute(f"UPDATE traces SET {sets} WHERE id=?", vals)
    conn.commit()
    conn.close()

def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def list_traces(limit: int = 10, offset: int = 0, user_id: Optional[str] = None, tone: Optional[str] = None, status: Optional[str] = None, q: Optional[str] = None, is_low: Optional[bool] = None) -> List[Dict[str, Any]]:
    # enforce 30 days retention filter
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    conn = _conn()
    where = ["created_at >= ?"]
    params: List[Any] = [cutoff]
    if user_id:
        where.append("user_id=?")
        params.append(user_id)
    if tone:
        where.append("tone=?")
        params.append(tone)
    if status:
        where.append("status=?")
        params.append(status)
    if q:
        where.append("(query LIKE ? OR answer LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if is_low is not None:
        where.append("is_low_confidence=?")
        params.append(1 if is_low else 0)
    where_str = " AND ".join(where)
    rows = conn.execute(f"SELECT * FROM traces WHERE {where_str} ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
    # total count
    total_row = conn.execute(f"SELECT COUNT(*) as c FROM traces WHERE {where_str}", params).fetchone()
    total = total_row["c"] if total_row else 0
    conn.close()
    return [dict(r) for r in rows], total

def delete_expired_traces() -> int:
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    conn = _conn()
    cur = conn.execute("DELETE FROM traces WHERE created_at < ?", (cutoff,))
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n

# ---------- spans ----------
def create_span(trace_id: str, name: str, parent_id: Optional[str] = None, inputs: Optional[Dict]=None) -> str:
    sid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute("INSERT INTO spans (id, trace_id, parent_id, name, start_at, inputs_json) VALUES (?,?,?,?,?,?)",
                 (sid, trace_id, parent_id, name, now, json.dumps(inputs, ensure_ascii=False) if inputs else None))
    conn.commit()
    conn.close()
    return sid

def end_span(span_id: str, outputs: Optional[Dict]=None, metadata: Optional[Dict]=None):
    now = datetime.utcnow().isoformat()
    conn = _conn()
    row = conn.execute("SELECT start_at FROM spans WHERE id=?", (span_id,)).fetchone()
    duration = None
    if row and row["start_at"]:
        try:
            start = datetime.fromisoformat(row["start_at"])
            end = datetime.fromisoformat(now)
            duration = (end - start).total_seconds()*1000
        except Exception:
            pass
    conn.execute("UPDATE spans SET end_at=?, duration_ms=?, outputs_json=?, metadata_json=? WHERE id=?",
                 (now, duration, json.dumps(outputs, ensure_ascii=False) if outputs else None, json.dumps(metadata, ensure_ascii=False) if metadata else None, span_id))
    conn.commit()
    conn.close()
    return duration

def list_spans(trace_id: str) -> List[Dict[str, Any]]:
    conn = _conn()
    rows = conn.execute("SELECT * FROM spans WHERE trace_id=? ORDER BY start_at ASC", (trace_id,)).fetchall()
    conn.close()
    out=[]
    for r in rows:
        d=dict(r)
        for k in ["inputs_json","outputs_json","metadata_json"]:
            if d.get(k):
                try:
                    d[k+"_parsed"] = json.loads(d[k])
                except Exception:
                    pass
        out.append(d)
    return out

# ---------- span_chunks ----------
def add_span_chunks(span_id: str, trace_id: str, chunks: List[Dict[str, Any]]):
    if not chunks:
        return
    conn = _conn()
    for idx, c in enumerate(chunks):
        cid = uuid.uuid4().hex
        # page_numbers -> first page
        pages = c.get("page_numbers") or []
        if isinstance(pages, str):
            try:
                pages = json.loads(pages)
            except Exception:
                pages = []
        first_page = pages[0] if pages else None
        # snippet 500
        snippet = (c.get("content") or "")[:500]
        full = (c.get("content") or "")[:500]  # only snippet as per spec, but keep same
        conn.execute("""INSERT INTO span_chunks
            (id, span_id, trace_id, rank, source_id, file_name, dataset, section_path, page_numbers, chunk_index, chunk_hash, chunking_strategy, embedding_model, updated_at, score, content_snippet, content_full)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, span_id, trace_id, idx, c.get("source_id"), c.get("file_name"), c.get("dataset"), json.dumps(c.get("section_path"), ensure_ascii=False) if c.get("section_path") else None, json.dumps(pages, ensure_ascii=False), c.get("chunk_index"), c.get("chunk_hash"), c.get("chunking_strategy"), c.get("embedding_model"), c.get("updated_at"), c.get("score"), snippet, full))
    conn.commit()
    conn.close()

def list_chunks(trace_id: str, span_id: Optional[str]=None) -> List[Dict[str, Any]]:
    conn = _conn()
    if span_id:
        rows = conn.execute("SELECT * FROM span_chunks WHERE span_id=? ORDER BY rank ASC", (span_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM span_chunks WHERE trace_id=? ORDER BY rank ASC", (trace_id,)).fetchall()
    conn.close()
    out=[]
    for r in rows:
        d=dict(r)
        # parse jsons
        for k in ["section_path","page_numbers"]:
            if d.get(k):
                try:
                    d[k+"_parsed"] = json.loads(d[k])
                except Exception:
                    pass
        out.append(d)
    return out

# ---------- prompts ----------
def list_prompts(tone: Optional[str]=None, status: Optional[str]=None, limit=100) -> List[Dict[str, Any]]:
    conn = _conn()
    where=[]
    params=[]
    if tone:
        where.append("tone=?")
        params.append(tone)
    if status:
        where.append("status=?")
        params.append(status)
    where_str = " AND ".join(where) if where else "1=1"
    rows = conn.execute(f"SELECT * FROM prompts WHERE {where_str} ORDER BY created_at DESC LIMIT ?", params+[limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_active_prompt(tone: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM prompts WHERE tone=? AND is_active=1 ORDER BY created_at DESC LIMIT 1", (tone,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_prompt_by_version(tone: str, version: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM prompts WHERE tone=? AND version=?", (tone, version)).fetchone()
    conn.close()
    return dict(row) if row else None

def create_prompt(tone: str, text: str, description: Optional[str], created_by: Optional[str], status: str = "pending_approval") -> Dict[str, Any]:
    import hashlib
    version = hashlib.sha256(text.encode()).hexdigest()[:8]
    pid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    # check duplicate version for tone
    existing = conn.execute("SELECT id FROM prompts WHERE tone=? AND version=?", (tone, version)).fetchone()
    if existing:
        # return existing
        conn.close()
        return get_prompt_by_version(tone, version)  # type: ignore
    # insert as pending_approval, not active
    conn.execute("INSERT INTO prompts (id, tone, version, text, description, created_at, created_by, is_active, parent_version, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (pid, tone, version, text, description, now, created_by, 0, None, status))
    conn.commit()
    conn.close()
    return get_prompt_by_version(tone, version)  # type: ignore

def approve_prompt(tone: str, version: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    # deactivate all for tone
    conn.execute("UPDATE prompts SET is_active=0, status='archived' WHERE tone=? AND is_active=1", (tone,))
    # activate target
    conn.execute("UPDATE prompts SET is_active=1, status='active' WHERE tone=? AND version=?", (tone, version))
    conn.commit()
    conn.close()
    return get_prompt_by_version(tone, version)

def reject_prompt(tone: str, version: str):
    conn = _conn()
    conn.execute("UPDATE prompts SET status='archived', is_active=0 WHERE tone=? AND version=?", (tone, version))
    conn.commit()
    conn.close()

# ---------- ragas ----------
def upsert_ragas(trace_id: str, metrics: Dict[str, Any], raw: Dict[str,Any], evaluator_model: Optional[str]=None):
    import hashlib
    rid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    conn = _conn()
    # check existing
    existing = conn.execute("SELECT id FROM ragas_evaluations WHERE trace_id=?", (trace_id,)).fetchone()
    if existing:
        conn.execute("""UPDATE ragas_evaluations SET faithfulness=?, context_precision=?, context_recall=?, answer_relevance=?, fluency=?,
            faithfulness_comment=?, precision_comment=?, recall_comment=?, relevance_comment=?, fluency_comment=?,
            failed_metrics=?, confidence=?, raw_json=?, evaluated_at=?, evaluator_model=? WHERE trace_id=?""",
            (metrics.get("faithfulness"), metrics.get("context_precision"), metrics.get("context_recall"), metrics.get("answer_relevance"), metrics.get("fluency"),
             metrics.get("faithfulness_comment") or "", metrics.get("context_precision_comment") or "", metrics.get("context_recall_comment") or "", metrics.get("answer_relevance_comment") or "", metrics.get("fluency_comment") or "",
             json.dumps(metrics.get("failed_metrics") or []), metrics.get("confidence"), json.dumps(raw, ensure_ascii=False), now, evaluator_model, trace_id))
    else:
        conn.execute("""INSERT INTO ragas_evaluations
            (id, trace_id, faithfulness, context_precision, context_recall, answer_relevance, fluency, faithfulness_comment, precision_comment, recall_comment, relevance_comment, fluency_comment, failed_metrics, confidence, raw_json, evaluated_at, evaluator_model)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, trace_id, metrics.get("faithfulness"), metrics.get("context_precision"), metrics.get("context_recall"), metrics.get("answer_relevance"), metrics.get("fluency"),
             metrics.get("faithfulness_comment") or "", metrics.get("context_precision_comment") or "", metrics.get("context_recall_comment") or "", metrics.get("answer_relevance_comment") or "", metrics.get("fluency_comment") or "",
             json.dumps(metrics.get("failed_metrics") or []), metrics.get("confidence"), json.dumps(raw, ensure_ascii=False), now, evaluator_model))
    conn.commit()
    conn.close()

def get_ragas(trace_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM ragas_evaluations WHERE trace_id=?", (trace_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d=dict(row)
    try:
        d["failed_metrics"] = json.loads(d.get("failed_metrics") or "[]")
        d["raw"] = json.loads(d.get("raw_json") or "{}")
    except Exception:
        pass
    return d

# init on import
try:
    init_tracing_db()
except Exception:
    pass
