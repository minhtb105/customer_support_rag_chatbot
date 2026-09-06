"""Review service — DB ops + notifications"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from src.auth.db import _get_conn, create_notification, add_query_history, update_query_history_answer
    from src.config import VALID_ROLES
except ImportError:
    from auth.db import _get_conn, create_notification, add_query_history, update_query_history_answer  # type: ignore
    from config import VALID_ROLES  # type: ignore

def create_review_request(
    query: str,
    draft_answer: str,
    contexts: List[Dict[str, Any]],
    evaluation: Dict[str, Any],
    requester_id: str,
    disease: Optional[str] = None,
    langsmith_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    rid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    confidence = evaluation.get("confidence")
    reason = evaluation.get("comments", {})
    failed = evaluation.get("failed_metrics", [])
    routed = evaluation.get("routed_role", "doctor")
    contexts_json = json.dumps(contexts, ensure_ascii=False)
    conf_json = json.dumps(evaluation, ensure_ascii=False)
    conn = _get_conn()
    conn.execute(
        """INSERT INTO review_requests
        (id, query, draft_answer, contexts_json, confidence_json, confidence, confidence_reason, failed_metrics, routed_role, status, requester_id, disease, created_at, langsmith_run_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rid, query, draft_answer, contexts_json, conf_json, confidence, json.dumps(reason, ensure_ascii=False), json.dumps(failed, ensure_ascii=False), routed, "pending", requester_id, disease, now, langsmith_run_id)
    )
    conn.commit()
    conn.close()
    # also create query_history as pending
    add_query_history(user_id=requester_id, query=query, answer=draft_answer, status="pending_review", review_id=rid, confidence=confidence)
    # notify experts of routed role + admins
    # find users with that role (verified)
    conn2 = _get_conn()
    experts = conn2.execute("SELECT id FROM users WHERE role=? AND is_active=1 AND is_verified=1", (routed,)).fetchall()
    admins = conn2.execute("SELECT id FROM users WHERE role='admin' AND is_active=1").fetchall()
    conn2.close()
    notify_ids = set([r["id"] for r in experts] + [r["id"] for r in admins])
    # also if no expert of routed role, notify all expert roles
    if not experts:
        conn3 = _get_conn()
        fallback = conn3.execute("SELECT id FROM users WHERE role IN ('doctor','pharmacist','specialist') AND is_active=1 AND is_verified=1").fetchall()
        conn3.close()
        notify_ids.update([r["id"] for r in fallback])
    for uid in notify_ids:
        create_notification(uid, type="review_pending", title=f"Mới: cần duyệt ({routed})", body=f"Query: {query[:80]}... | metrics fail: {', '.join(failed)}", review_id=rid)
    # notify requester that pending
    create_notification(requester_id, type="review_pending_user", title="Câu hỏi đang chờ duyệt chuyên gia", body=f"Câu hỏi của bạn đang được chuyển tới {routed} do AI chưa tự tin (failed: {', '.join(failed)})", review_id=rid)
    return get_review(rid)  # type: ignore

def get_review(review_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM review_requests WHERE id=?", (review_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    # parse jsons
    try:
        d["contexts"] = json.loads(d.get("contexts_json") or "[]")
        d["evaluation"] = json.loads(d.get("confidence_json") or "{}")
        d["failed_metrics"] = json.loads(d.get("failed_metrics") or "[]")
    except Exception:
        d["contexts"] = []
        d["evaluation"] = {}
        d["failed_metrics"] = []
    return d

def list_reviews(
    status: Optional[str] = None,
    routed_role: Optional[str] = None,
    requester_id: Optional[str] = None,
    assigned_expert_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    q = "SELECT * FROM review_requests WHERE 1=1"
    params: List[Any] = []
    if status:
        q += " AND status=?"
        params.append(status)
    if routed_role:
        q += " AND routed_role=?"
        params.append(routed_role)
    if requester_id:
        q += " AND requester_id=?"
        params.append(requester_id)
    if assigned_expert_id:
        q += " AND assigned_expert_id=?"
        params.append(assigned_expert_id)
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(q, params).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["contexts"] = json.loads(d.get("contexts_json") or "[]")
            d["evaluation"] = json.loads(d.get("confidence_json") or "{}")
            d["failed_metrics"] = json.loads(d.get("failed_metrics") or "[]")
        except Exception:
            pass
        out.append(d)
    return out

def decide_review(review_id: str, decision: str, expert_id: str, final_answer: Optional[str] = None, expert_notes: Optional[str] = None) -> Dict[str, Any]:
    # decision: approved | rejected | revised
    if decision not in ("approved", "rejected", "revised"):
        raise ValueError("Invalid decision")
    rev = get_review(review_id)
    if not rev:
        raise ValueError("Review not found")
    if rev["status"] != "pending":
        raise ValueError(f"Review already {rev['status']}")
    # determine final_answer
    if decision == "approved":
        final = rev["draft_answer"]
    elif decision == "revised":
        if not final_answer:
            raise ValueError("final_answer required for revised")
        final = final_answer
    else:  # rejected
        final = final_answer or "Câu hỏi đã bị từ chối bởi chuyên gia. Vui lòng cung cấp thêm thông tin hoặc liên hệ trực tiếp."
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute("UPDATE review_requests SET status=?, assigned_expert_id=?, final_answer=?, expert_notes=?, reviewed_at=? WHERE id=?",
                 (decision if decision != "approved" else "approved", expert_id, final, expert_notes, now, review_id))
    conn.commit()
    conn.close()
    # update query_history
    update_query_history_answer(review_id, final)
    # notifications
    requester_id = rev["requester_id"]
    create_notification(requester_id, type="review_decided", title=f"Câu trả lời đã được duyệt ({decision})", body=f"Chuyên gia đã {decision}: {(final or '')[:120]}", review_id=review_id)
    # also notify admin? optional
    return get_review(review_id)  # type: ignore

def get_review_with_vitals(review_id: str) -> Optional[Dict[str, Any]]:
    """Expert xem toàn bộ vitals/logs của requester"""
    rev = get_review(review_id)
    if not rev:
        return None
    requester_id = rev["requester_id"]
    # fetch vitals from existing trackers
    vitals = {}
    try:
        from src.features.glucose_tracker import get_logs as g_logs, get_stats as g_stats
        vitals["glucose_logs"] = g_logs(requester_id, limit=20)
        vitals["glucose_stats"] = g_stats(requester_id)
    except Exception:
        vitals["glucose_logs"] = []
    try:
        from src.features.bp_tracker import get_bp_logs, get_bp_stats
        vitals["bp_logs"] = get_bp_logs(requester_id, limit=20)
        vitals["bp_stats"] = get_bp_stats(requester_id)
    except Exception:
        vitals["bp_logs"] = []
    try:
        from src.features.respiratory_tracker import get_respiratory_logs, get_respiratory_stats
        vitals["respiratory_logs"] = get_respiratory_logs(requester_id, limit=20)
        vitals["respiratory_stats"] = get_respiratory_stats(requester_id)
    except Exception:
        vitals["respiratory_logs"] = []
    try:
        from src.features.mood_tracker import get_mood_logs, get_mood_stats
        vitals["mood_logs"] = get_mood_logs(requester_id, limit=20)
        vitals["mood_stats"] = get_mood_stats(requester_id)
    except Exception:
        vitals["mood_logs"] = []
    rev["vitals"] = vitals
    # also fetch user info
    try:
        from src.auth.db import get_user_by_id
        rev["requester"] = get_user_by_id(requester_id)
    except Exception:
        rev["requester"] = None
    return rev
