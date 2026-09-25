"""Lab HITL B3: tier-routing + review rows + event-log + pharmacist account.

Isolation: monkeypatch src.labs.store.LAB_DB_PATH (tmp) + TRIAGE_EVENTS_PATH
(tmp) + delenv OPENAI_API_KEY (precedent test_labs_ask). Reviews live in the
session tmp auth DB (conftest client fixture) — count before/after per test.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_lab_data as seed  # noqa: E402

PIDS = ["syn_patient_911", "syn_patient_912", "syn_patient_913"]

Q_T0 = "Phiếu xét nghiệm của tôi thế nào?"
Q_T1 = "Tôi muốn đặt lịch khám tuần sau, cho hỏi giờ khám ở phòng khám?"
Q_T2_DRUG = "Thuốc này uống chung có tương tác gì không?"
Q_T2_MILD = "Tôi bị tê chân nhẹ, có sao không?"
Q_T3 = "Tôi có nên đổi liều thuốc tiểu đường không?"


@pytest.fixture
def iso_hitl(monkeypatch, tmp_path):
    import src.labs.store as ls
    tmp_db = tmp_path / "labs.db"
    monkeypatch.setattr(ls, "LAB_DB_PATH", tmp_db)
    monkeypatch.setenv("TRIAGE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seed.seed_all(seed=42, reset=True, patient_ids=PIDS)
    return tmp_path


def _anon(client):
    client.cookies.clear()


def _ask(client, body):
    return client.post("/v1/labs/ask", json=body)


def _reviews():
    from src.reviews.service import list_reviews
    return list_reviews(limit=1000)


def _events(path):
    p = path / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _set_panic(pid, loinc="4548-4", value=13.5):
    import src.labs.store as ls
    reps = ls.get_reports(pid)
    assert reps, f"no reports for {pid}"
    rid = reps[0]["report_id"]
    conn = sqlite3.connect(str(ls.LAB_DB_PATH))
    try:
        conn.execute("UPDATE lab_observations SET value=? WHERE report_id=? AND loinc=?",
                     (value, rid, loinc))
        conn.commit()
    finally:
        conn.close()
    return rid


class TestTierMapping:
    def test_pure_four_tiers(self):
        from src.labs.handoff import route_tier
        assert route_tier(Q_T0)["tier"] == "T0"
        assert route_tier(Q_T1)["tier"] == "T1"
        assert route_tier(Q_T2_DRUG)["tier"] == "T2"
        assert route_tier(Q_T3)["tier"] == "T3"
        assert route_tier("bat ky", panic=True)["tier"] == "T3"

    def test_panic_always_t3_with_115(self, client, iso_hitl):
        _anon(client)
        rid = _set_panic(PIDS[0])
        r = _ask(client, {"user_id": PIDS[0], "report_id": rid,
                          "question": "Kết quả này sao rồi?"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["handoff_tier"] == "T3"
        assert "115" in (data["message"] or "")
        assert data["review_id"]  # T3 creates review row


class TestReviewRows:
    def test_t0_creates_no_row(self, client, iso_hitl):
        _anon(client)
        before = len(_reviews())
        r = _ask(client, {"user_id": PIDS[0], "question": Q_T0})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["handoff_tier"] == "T0"
        assert data["review_id"] is None
        assert data["what_happens_next"]
        assert len(_reviews()) == before

    def test_t1_routes_admin(self, client, iso_hitl):
        from src.reviews.service import get_review
        _anon(client)
        r = _ask(client, {"user_id": PIDS[0], "question": Q_T1})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["handoff_tier"] == "T1"
        assert data["review_id"]
        assert data["what_happens_next"]
        rev = get_review(data["review_id"])
        assert rev and rev["routed_role"] == "admin"

    def test_t2_routes_pharmacist_or_specialist(self, client, iso_hitl):
        from src.reviews.service import get_review
        _anon(client)
        r = _ask(client, {"user_id": PIDS[1], "question": Q_T2_DRUG})
        assert r.status_code == 200, r.text
        assert r.json()["handoff_tier"] == "T2"
        rev = get_review(r.json()["review_id"])
        assert rev and rev["routed_role"] == "pharmacist"
        r2 = _ask(client, {"user_id": PIDS[1], "question": Q_T2_MILD})
        assert r2.status_code == 200, r2.text
        assert r2.json()["handoff_tier"] == "T2"
        rev2 = get_review(r2.json()["review_id"])
        assert rev2 and rev2["routed_role"] == "specialist"

    def test_t3_routes_doctor(self, client, iso_hitl):
        from src.reviews.service import get_review
        _anon(client)
        r = _ask(client, {"user_id": PIDS[2], "question": Q_T3})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["handoff_tier"] == "T3"
        rev = get_review(data["review_id"])
        assert rev and rev["routed_role"] == "doctor"


class TestEventLogAndAuth:
    def test_event_log_appended(self, client, iso_hitl):
        _anon(client)
        _ask(client, {"user_id": PIDS[0], "question": Q_T0})
        _ask(client, {"user_id": PIDS[0], "question": Q_T1})
        evs = _events(iso_hitl)
        assert len(evs) >= 2
        assert all("lab" in str(e.get("specialty") or "") for e in evs)

    def test_anonymous_empty_401(self, client, iso_hitl):
        _anon(client)
        r = _ask(client, {"question": "Phiếu của tôi thế nào?"})
        assert r.status_code == 401, r.text

    def test_seed_has_pharmacist_account(self):
        import scripts.seed_demo_users as su
        rows = dict((u, (r, v)) for u, r, v in su.ACCOUNTS)
        assert "demo_pharmacist_01" in rows
        assert rows["demo_pharmacist_01"] == ("pharmacist", 1)
        # 3 existing accounts untouched
        assert rows["demo_patient_01"] == ("user", 1)
        assert rows["demo_doctor_01"] == ("doctor", 1)
        assert rows["demo_admin_01"] == ("admin", 1)

    def test_pharmacist_verified_passes_require_expert(self, client, admin_client):
        tok = admin_client["token"]
        import uuid
        uname = f"ph_{uuid.uuid4().hex[:6]}"
        pwd = "Ph@12345"
        r = client.post("/v1/auth/register",
                        json={"username": uname, "password": pwd, "role": "pharmacist"},
                        headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code in (200, 201), r.text
        r2 = client.post("/v1/auth/login-json", json={"username": uname, "password": pwd})
        assert r2.status_code == 200, r2.text
        assert r2.json()["user"]["role"] == "pharmacist"
        assert r2.json()["user"]["is_verified"] in (1, True)
        ptok = r2.json()["access_token"]
        # require_expert gate: GET /v1/doctor/patients (doctor/pharmacist/specialist/admin only)
        g = client.get("/v1/doctor/patients",
                       headers={"Authorization": f"Bearer {ptok}"})
        assert g.status_code == 200, g.text
