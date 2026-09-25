import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.auth.security import create_access_token, hash_password
from src.auth.db import create_user, get_user_by_username, _get_conn
from src.shared.config import BASE_DIR
from src.monitors.db import (
    create_guideline_version,
    create_safety_alert,
    list_guideline_versions,
    list_safety_alerts,
    get_stats,
    init_monitoring_db,
)
from src.monitors.utils import is_url_allowed
from src.monitors.summarizer import summarize_guideline_diff, summarize_safety_alert


@pytest.fixture(scope="module")
def users():
    # cleanup
    for uname in ["mon_test_admin", "mon_test_spec", "mon_test_pharm", "mon_test_user"]:
        u = get_user_by_username(uname)
        if u:
            conn = _get_conn()
            conn.execute("DELETE FROM users WHERE username=?", (uname,))
            conn.commit()
            conn.close()
    admin = create_user("mon_test_admin", "admin_mon@test.com", hash_password("test123"), "Admin Mon", "admin", is_verified=True)
    spec = create_user("mon_test_spec", "spec_mon@test.com", hash_password("test123"), "Spec Mon", "specialist", is_verified=True)
    pharm = create_user("mon_test_pharm", "pharm_mon@test.com", hash_password("test123"), "Pharm Mon", "pharmacist", is_verified=True)
    user = create_user("mon_test_user", "user_mon@test.com", hash_password("test123"), "User Mon", "user", is_verified=True)
    yield {"admin": admin, "spec": spec, "pharm": pharm, "user": user}
    # teardown
    conn = _get_conn()
    conn.execute("DELETE FROM users WHERE username IN (?,?,?,?)", ("mon_test_admin", "mon_test_spec", "mon_test_pharm", "mon_test_user"))
    conn.commit()
    conn.close()
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    mconn.execute("DELETE FROM guideline_versions WHERE source='ada' AND version_label='manual-required'")
    mconn.execute("DELETE FROM safety_alerts WHERE alert_title='Test safety for API'")
    mconn.execute("DELETE FROM guideline_versions WHERE title LIKE 'Test%'")
    mconn.execute("DELETE FROM safety_alerts WHERE drug_name IN ('TestDrug','Metformin')")
    mconn.commit()
    mconn.close()


def token_for(u):
    return create_access_token({"sub": u["id"], "username": u["username"], "role": u["role"]})


def test_ssrf_allowlist():
    assert is_url_allowed("https://iris.who.int/handle/10665/325182") is True
    assert is_url_allowed("https://api.fda.gov/drug/enforcement.json?search=test") is True
    assert is_url_allowed("http://127.0.0.1/admin") is False
    assert is_url_allowed("https://evil.com/malicious") is False


def test_summarizer_fallback_guideline_vi():
    res = summarize_guideline_diff("gold", "GOLD 2024", "old text", "new text about COPD GOLD 2025 update", version_hint="2025")
    assert "tom_tat_tieng_viet" in res
    assert isinstance(res["tom_tat_tieng_viet"], str)
    # fallback should contain VI
    assert len(res["tom_tat_tieng_viet"]) > 0


def test_summarizer_fallback_safety_vi():
    res = summarize_safety_alert({"alert_title": "Test recall", "drug_name": "Metformin", "reason_for_recall": "contamination"}, "FDA")
    assert "tieu_de_vi" in res
    assert "tom_tat_vi" in res
    assert res.get("nguon") == "FDA"


def test_guideline_db_create_list():
    init_monitoring_db()
    gv = create_guideline_version(source="gold", title="Test GOLD DB", url="https://goldcopd.org/test", version_label="test-db-1", sha256="abc999", staging_path=None, change_summary_json=json.dumps({"tom_tat_tieng_viet": "test"}))
    assert gv["status"] == "pending_review"
    rows, total = list_guideline_versions(status="pending_review", source="gold", limit=10)
    assert total >= 1
    assert any(r["id"] == gv["id"] for r in rows)
    # cleanup
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    mconn.execute("DELETE FROM guideline_versions WHERE id=?", (gv["id"],))
    mconn.commit()
    mconn.close()


def test_safety_db_dedup():
    sa = create_safety_alert(source="FDA", alert_type="recall", alert_title="Test dup safety", severity="high", drug_name="DupDrug", alert_url="https://api.fda.gov/test_dup_url_123", raw_json=json.dumps({"test": 1}))
    sa2 = create_safety_alert(source="FDA", alert_type="recall", alert_title="Test dup safety", severity="high", drug_name="DupDrug", alert_url="https://api.fda.gov/test_dup_url_123", raw_json=json.dumps({"test": 1}))
    assert sa["id"] == sa2["id"]
    # cleanup
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    mconn.execute("DELETE FROM safety_alerts WHERE id=?", (sa["id"],))
    mconn.commit()
    mconn.close()


def test_api_monitors_status(users):
    from src.api.main import app

    client = TestClient(app)
    resp = client.get("/v1/monitors/status", headers={"Authorization": f"Bearer {token_for(users['admin'])}"})
    assert resp.status_code == 200
    assert "stats" in resp.json()
    assert "sources" in resp.json()


def test_api_guideline_trigger_and_decision(users):
    from src.api.main import app

    client = TestClient(app)
    # trigger with mocked download failure -> paywall placeholder
    with patch("src.monitors.guideline_fetcher.download_file_safe", return_value=False):
        with patch("src.monitors.guideline_fetcher.head_with_etag", return_value={"etag": "test-etag-xyz", "last_modified": "2025-01-01", "status": 200}):
            resp = client.post("/v1/monitors/check/guidelines?source_key=ada_soc&force=true", headers={"Authorization": f'Bearer {token_for(users["spec"])}'})
            assert resp.status_code == 200
            assert resp.json()["result"]["found_new"] is True

    resp = client.get("/v1/monitors/guidelines?status=pending_review", headers={"Authorization": f'Bearer {token_for(users["spec"])}'})
    assert resp.status_code == 200
    items = [i for i in resp.json()["items"] if i["source"] == "ada"]
    assert len(items) >= 1
    gid = items[0]["id"]

    # user should be forbidden
    resp2 = client.post(f"/v1/monitors/guidelines/{gid}/decision", json={"decision": "rejected", "notes": "test"}, headers={"Authorization": f'Bearer {token_for(users["user"])}'})
    assert resp2.status_code == 403

    # specialist can reject
    resp2 = client.post(f"/v1/monitors/guidelines/{gid}/decision", json={"decision": "rejected", "notes": "test reject by spec"}, headers={"Authorization": f'Bearer {token_for(users["spec"])}'})
    assert resp2.status_code == 200
    assert resp2.json()["result"]["status"] == "rejected"

    # doctor also allowed (create doctor for test)
    doc = get_user_by_username("mon_test_doctor_tmp")
    if doc:
        conn = _get_conn()
        conn.execute("DELETE FROM users WHERE username=?", ("mon_test_doctor_tmp",))
        conn.commit()
        conn.close()
    doc = create_user("mon_test_doctor_tmp", "doc_tmp@test.com", hash_password("test123"), "Doc", "doctor", is_verified=True)
    # try approve already rejected should fail 400
    resp2 = client.post(f"/v1/monitors/guidelines/{gid}/decision", json={"decision": "approved"}, headers={"Authorization": f'Bearer {token_for(doc)}'})
    assert resp2.status_code == 400
    # cleanup doc
    conn = _get_conn()
    conn.execute("DELETE FROM users WHERE username=?", ("mon_test_doctor_tmp",))
    conn.commit()
    conn.close()


def test_api_safety_approve_roles(users):
    from src.api.main import app

    client = TestClient(app)
    sa = create_safety_alert(source="FDA", alert_type="recall", alert_title="Test safety for API role", severity="critical", drug_name="TestDrug", alert_url=f"https://api.fda.gov/test_api_role_{users['admin']['id'][:6]}", raw_json=json.dumps({"test": 1}), ai_summary=json.dumps({"tom_tat_vi": "test"}))
    aid = sa["id"]
    # spec should be forbidden for safety
    resp2 = client.post(f"/v1/monitors/alerts/{aid}/decision", json={"decision": "approved", "notes": "ok"}, headers={"Authorization": f'Bearer {token_for(users["spec"])}'})
    assert resp2.status_code == 403
    # pharm can approve
    resp2 = client.post(f"/v1/monitors/alerts/{aid}/decision", json={"decision": "approved", "notes": "ok pharm"}, headers={"Authorization": f'Bearer {token_for(users["pharm"])}'})
    assert resp2.status_code == 200
    assert resp2.json()["result"]["status"] == "approved"
    # cleanup
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    mconn.execute("DELETE FROM safety_alerts WHERE id=?", (aid,))
    mconn.commit()
    mconn.close()


def test_promote_flow_staging_to_corpus(users):
    from src.monitors.db import create_guideline_version
    from src.monitors.service import promote_guideline_to_corpus
    from src.shared.config import STAGING_DIR

    staging = STAGING_DIR / "gold" / "test_pytest_promote"
    staging.mkdir(parents=True, exist_ok=True)
    pdf_path = staging / "pytest_gold_test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content for pytest promote" + b"0" * 2000)
    gv = create_guideline_version(source="gold", title="GOLD Pytest Promote", url="https://goldcopd.org/pytest", version_label="pytest-1", sha256="fakehash_pytest", staging_path=str(pdf_path), change_summary_json=json.dumps({"tom_tat_tieng_viet": "test"}))
    # mock reindex
    with patch("src.monitors.indexer_helper.reindex_single_pdf") as mock_reindex:
        mock_reindex.return_value = None
        res = promote_guideline_to_corpus(gv["id"], reviewer_id=users["spec"]["id"], approve=True, review_notes="pytest approve")
        assert res["status"] == "approved"
        assert Path(res["corpus_path"]).name == "pytest_gold_test.pdf"
        assert Path(res["corpus_path"]).exists()
        # cleanup corpus
        Path(res["corpus_path"]).unlink(missing_ok=True)
    # cleanup db
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    mconn.execute("DELETE FROM guideline_versions WHERE id=?", (gv["id"],))
    mconn.commit()
    mconn.close()
    # ensure staging cleaned (moved)
    assert not pdf_path.exists()


def test_superseded_cleanup():
    from src.monitors.db import create_guideline_version, update_guideline_status
    from datetime import datetime, timedelta, timezone

    gv = create_guideline_version(source="gold", title="Superseded Test", url="https://goldcopd.org/superseded", version_label="sup-1", sha256="hash-sup")
    # mark superseded with old reviewed_at
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    mconn.execute("UPDATE guideline_versions SET status='superseded', reviewed_at=? WHERE id=?", (old_date, gv["id"]))
    mconn.commit()
    mconn.close()
    from src.monitors.service import cleanup_superseded

    deleted = cleanup_superseded()
    assert deleted >= 1
    # ensure deleted
    mconn = sqlite3.connect(str(BASE_DIR / "metadata" / "monitoring.db"))
    row = mconn.execute("SELECT * FROM guideline_versions WHERE id=?", (gv["id"],)).fetchone()
    mconn.close()
    assert row is None
