"""Monitor Service — promote guideline to corpus, notifications, cleanup"""
from __future__ import annotations
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from src.config import BASE_DIR, STAGING_DIR, PDF_DIR, MONITOR_SUPERSEDED_RETENTION_DAYS
    from src.monitors.db import (
        get_guideline_version, get_safety_alert,
        update_guideline_status, update_safety_status,
        list_guideline_versions, delete_superseded_expired,
        get_source,
    )
    from src.auth.db import create_notification, _get_conn as auth_conn
except ImportError:
    from config import BASE_DIR, STAGING_DIR, PDF_DIR, MONITOR_SUPERSEDED_RETENTION_DAYS  # type: ignore
    from monitors.db import get_guideline_version, get_safety_alert, update_guideline_status, update_safety_status, list_guideline_versions, delete_superseded_expired, get_source  # type: ignore
    from auth.db import create_notification, _get_conn as auth_conn  # type: ignore

def _notify_roles(roles: list[str], title: str, body: str, review_id: Optional[str] = None):
    """Create in-app notifications for all users of given roles + admins."""
    try:
        conn = auth_conn()
        # collect user ids for roles
        placeholders = ",".join(["?"] * len(roles))
        experts = conn.execute(f"SELECT id FROM users WHERE role IN ({placeholders}) AND is_active=1 AND is_verified=1", roles).fetchall()
        admins = conn.execute("SELECT id FROM users WHERE role='admin' AND is_active=1").fetchall()
        conn.close()
        notify_ids = set([r["id"] for r in experts] + [r["id"] for r in admins])
        # fallback: if no expert of that role, notify all expert roles
        if not experts:
            conn2 = auth_conn()
            fallback = conn2.execute("SELECT id FROM users WHERE role IN ('doctor','pharmacist','specialist') AND is_active=1 AND is_verified=1").fetchall()
            conn2.close()
            notify_ids.update([r["id"] for r in fallback])
        for uid in notify_ids:
            create_notification(uid, type="monitor_pending", title=title, body=body[:300], review_id=review_id)
    except Exception as e:
        print(f"[notify] failed: {e}")

def notify_guideline_pending(gv: Dict[str, Any], summary: Dict[str, Any]):
    title = f"[Guideline] Cần duyệt: {gv['source']} — {gv['title'][:40]}"
    body = summary.get("tom_tat_tieng_viet", "") or gv.get("title", "")
    # guideline -> specialist + doctor (both per requirement)
    _notify_roles(["specialist", "doctor"], title, body, review_id=gv["id"])

def notify_safety_pending(alert: Dict[str, Any], summary: Dict[str, Any]):
    title = f"[An toàn thuốc] Cần duyệt ({alert['severity']}): {alert['drug_name'] or alert['alert_title'][:30]}"
    # handle ai_summary may be json string
    try:
        s = json.loads(alert.get("ai_summary") or "{}")
        body = s.get("tom_tat_vi") or s.get("tieu_de_vi") or alert["alert_title"]
    except Exception:
        body = alert["alert_title"]
    # safety -> pharmacist per HILT_ROUTING
    _notify_roles(["pharmacist"], title, body, review_id=alert["id"])
    # if critical, also ensure admin notified (already in _notify_roles)

# ---------- Promote guideline ----------
def promote_guideline_to_corpus(gid: str, reviewer_id: str, approve: bool = True, review_notes: Optional[str] = None) -> Dict[str, Any]:
    gv = get_guideline_version(gid)
    if not gv:
        raise ValueError("Guideline version not found")
    if gv["status"] != "pending_review":
        raise ValueError(f"Already {gv['status']}")

    if not approve:
        # reject
        updated = update_guideline_status(gid, "rejected", reviewer_id=reviewer_id, review_notes=review_notes)
        # keep staging for 7 days then cleanup manually? For now keep
        return updated  # type: ignore

    # approve → move staging → corpus
    staging_path = gv.get("staging_path")
    if not staging_path or not Path(staging_path).exists():
        # if no file (paywall placeholder), just mark approved but no index
        updated = update_guideline_status(gid, "approved", reviewer_id=reviewer_id, review_notes=review_notes or "approved without file (paywall/manual)")
        return updated  # type: ignore

    src_path = Path(staging_path)
    # Determine corpus destination: PDF_DIR/<source>/filename
    source = gv["source"]
    # Map source to directory (reuse GUIDELINE_SOURCE_DIRS)
    try:
        from src.config import GUIDELINE_SOURCE_DIRS
        dest_dir = GUIDELINE_SOURCE_DIRS.get(source) or (PDF_DIR / source)
    except ImportError:
        from config import GUIDELINE_SOURCE_DIRS  # type: ignore
        dest_dir = GUIDELINE_SOURCE_DIRS.get(source) or (PDF_DIR / source)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    # If dest exists, version it
    if dest_path.exists():
        # archive old with timestamp
        backup_name = dest_path.stem + f"_superseded_{datetime.utcnow().strftime('%Y%m%d')}" + dest_path.suffix
        try:
            shutil.move(str(dest_path), str(dest_dir / backup_name))
        except Exception:
            pass
    # Move file
    try:
        shutil.move(str(src_path), str(dest_path))
        # cleanup empty staging parent if empty
        try:
            if src_path.parent.exists() and not any(src_path.parent.iterdir()):
                src_path.parent.rmdir()
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Move failed: {e}")

    # Mark superseded for previous approved of same source
    rows, _ = list_guideline_versions(source=source, status="approved", limit=20)
    for r in rows:
        if r["id"] != gid:
            # mark as superseded
            update_guideline_status(r["id"], "superseded", reviewer_id=reviewer_id, review_notes=f"Superseded by {gid} on {datetime.utcnow().isoformat()}")

    # Update current as approved with corpus_path and indexed_at after reindex
    update_guideline_status(gid, "approved", reviewer_id=reviewer_id, review_notes=review_notes, corpus_path=str(dest_path))

    # Trigger reindex for this file only (with version citation)
    indexed = False
    try:
        from src.monitors.indexer_helper import reindex_single_pdf

        version_label = gv.get("version_label")
        publication_date = gv.get("publication_date")
        reindex_single_pdf(str(dest_path), version_label=version_label, publication_date=publication_date)
        indexed = True
    except Exception as e:
        print(f"[promote] reindex failed {dest_path}: {e}")
        indexed = False

    # Final update indexed_at
    indexed_at = datetime.utcnow().isoformat() if indexed else None
    final = update_guideline_status(gid, "approved", reviewer_id=reviewer_id, review_notes=review_notes, corpus_path=str(dest_path), indexed_at=indexed_at)
    # Also invalidate cache if available
    try:
        from src.rag_pipeline import cache
        # clear cache entries that might use old guideline? For simplicity clear all
        # cache doesn't have clear, but we can reset stats or recreate
        # We'll try to clear internal dict if exists
        if hasattr(cache, "_exact"):
            try:
                cache._exact.clear()
            except Exception:
                pass
    except Exception:
        pass

    # Notify requester? For now notify admins/specialists that approved
    try:
        # find requester? guideline doesn't have requester_id, so notify admins
        _notify_roles(["specialist", "doctor"], f"[Guideline] Đã duyệt: {gv['title'][:40]}", f"Đã đưa vào corpus: {dest_path.name} (indexed={indexed})", review_id=gid)
    except Exception:
        pass

    return final  # type: ignore

def decide_safety_alert(aid: str, reviewer_id: str, decision: str, review_notes: Optional[str] = None) -> Dict[str, Any]:
    """decision: approved|dismissed"""
    alert = get_safety_alert(aid)
    if not alert:
        raise ValueError("Alert not found")
    if alert["status"] != "pending_review":
        raise ValueError(f"Already {alert['status']}")
    if decision not in ("approved", "dismissed"):
        raise ValueError("decision must be approved|dismissed")
    status = "approved" if decision == "approved" else "dismissed"
    updated = update_safety_status(aid, status, reviewer_id=reviewer_id, review_notes=review_notes)
    # notify admins/pharmacists
    try:
        _notify_roles(["pharmacist"], f"[An toàn thuốc] Đã {decision}: {alert['alert_title'][:40]}", review_notes or alert["alert_title"], review_id=aid)
    except Exception:
        pass
    return updated  # type: ignore

def cleanup_superseded() -> int:
    return delete_superseded_expired(days=MONITOR_SUPERSEDED_RETENTION_DAYS)
