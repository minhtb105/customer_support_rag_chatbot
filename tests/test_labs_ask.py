"""Labs Q&A B2: POST /v1/labs/ask — grounded, panic, citations, auth, zero-writes.

Isolation: monkeypatch module-namespace src.labs.store.LAB_DB_PATH + tmp_path
(precedent test_lab_seed); delenv OPENAI_API_KEY for determinism.
Read-only: every ask must leave lab_reports/lab_observations counts unchanged.
Labs carve-out: NEVER scope_guard on this path (kidney/liver are legit lab Qs).
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_lab_data as seed  # noqa: E402


PIDS = ["syn_patient_901", "syn_patient_902", "syn_patient_903"]


@pytest.fixture
def iso_labs(monkeypatch, tmp_path):
    import src.labs.store as ls
    tmp_db = tmp_path / "labs.db"
    monkeypatch.setattr(ls, "LAB_DB_PATH", tmp_db)
    monkeypatch.setenv("TRIAGE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRIAGE_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_BUSY", raising=False)
    seed.seed_all(seed=42, reset=True, patient_ids=PIDS)
    return tmp_path


def _anon(client):
    client.cookies.clear()


def _counts():
    import src.labs.store as ls
    return ls.count_reports(), ls.count_observations()


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


def _ask(client, body):
    return client.post("/v1/labs/ask", json=body)


class TestGroundedAnswer:
    def test_grounded_values_and_citations(self, client, iso_labs):
        import src.labs.store as ls
        _anon(client)
        reps = ls.get_reports(PIDS[0])
        rid = reps[0]["report_id"]
        before = _counts()
        r = _ask(client, {"user_id": PIDS[0], "report_id": rid,
                          "question": "Phiếu xét nghiệm của tôi thế nào?"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["panic"] is False and data["emergency"] is False
        assert data["report_id"] == rid
        assert len(data["answer_segments"]) >= 10  # full panel
        for seg in data["answer_segments"]:
            assert seg["text"].strip()
            assert seg["citations"], "per-claim citations required"
            for c in seg["citations"]:
                assert c["report_id"] and c["loinc"] and c["source"]
        obs = ls.get_observations(rid)
        assert any(str(o["value"]) in data["answer_segments"][i]["text"]
                   for o in obs for i in range(len(data["answer_segments"])))
        assert _counts() == before  # read-only

    def test_latest_report_when_no_report_id(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"user_id": PIDS[1], "question": "HbA1c của tôi bao nhiêu?"})
        assert r.status_code == 200, r.text
        assert r.json()["report_id"]
        assert r.json()["answer_segments"]

    def test_scope_guard_carveout_kidney_liver(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"user_id": PIDS[0],
                          "question": "Chỉ số creatinine/eGFR thận và men gan AST/ALT của tôi thế nào?"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["panic"] is False
        assert data["answer_segments"]
        blob = " ".join(s["text"] for s in data["answer_segments"])
        assert "nằm ngoài phạm vi" not in blob
        assert "thiết kế riêng để theo dõi" not in blob


class TestPanic:
    def test_panic_accented(self, client, iso_labs):
        _anon(client)
        rid = _set_panic(PIDS[0])
        before = _counts()
        r = _ask(client, {"user_id": PIDS[0], "report_id": rid,
                          "question": "HbA1c của tôi có nguy hiểm không?"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["panic"] is True and data["emergency"] is True
        assert "115" in (data["message"] or "")
        assert _counts() == before

    def test_panic_unaccented(self, client, iso_labs):
        _anon(client)
        rid = _set_panic(PIDS[1], loinc="33914-3", value=10.0)
        r = _ask(client, {"user_id": PIDS[1], "report_id": rid,
                          "question": "chi so egfr cua toi co sao khong?"})
        assert r.status_code == 200, r.text
        assert r.json()["panic"] is True
        assert "115" in (r.json()["message"] or "")

    def test_panic_skips_rag_import(self, client, iso_labs, monkeypatch):
        _anon(client)
        _set_panic(PIDS[2])
        monkeypatch.setitem(sys.modules, "src.chat.rag_pipeline", None)
        r = _ask(client, {"user_id": PIDS[2], "question": "Kết quả này sao rồi?"})
        assert r.status_code == 200, r.text
        assert r.json()["panic"] is True

    def test_red_flag_first_no_auth(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"question": "Đau ngực dữ dội, méo miệng từ sáng"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["emergency"] is True
        assert "115" in (data["message"] or "")

    def test_no_false_positive_ordinary(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"user_id": PIDS[0],
                          "question": "Đường huyết 140 lúc đói có ổn không?"})
        assert r.status_code == 200, r.text
        assert r.json()["panic"] is False
        assert r.json()["emergency"] is False


class TestDeterminismAuthWrites:
    def test_no_key_double_post_equal(self, client, iso_labs, monkeypatch):
        _anon(client)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setitem(sys.modules, "openai", None)
        body = {"user_id": PIDS[0], "question": "Phiếu này nói gì về mỡ máu?"}
        r1, r2 = _ask(client, body), _ask(client, body)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

    def test_401_anonymous_empty(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"question": "Phiếu của tôi thế nào?"})
        assert r.status_code == 401, r.text

    def test_403_user_cannot_read_other(self, client, make_user, iso_labs):
        victim, vtok = make_user(role="user")
        attacker, atok = make_user(role="user")
        r = client.post("/v1/labs/ask",
                        json={"user_id": victim["id"], "question": "Phiếu này sao?"},
                        headers={"Authorization": f"Bearer {atok}"})
        assert r.status_code == 403, r.text

    def test_owner_read_own(self, client, make_user, iso_labs):
        import src.labs.store as ls
        user, tok = make_user(role="user")
        seed.seed_all(seed=7, reset=True, patient_ids=[user["id"]])
        assert ls.get_reports(user["id"])
        r = client.post("/v1/labs/ask",
                        json={"user_id": user["id"], "question": "Phiếu của tôi thế nào?"},
                        headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        assert r.json()["answer_segments"]

    def test_404_unknown_report(self, client, iso_labs):
        _anon(client)
        r = _ask(client, {"user_id": PIDS[0], "report_id": "no-such-report",
                          "question": "Phiếu này sao?"})
        assert r.status_code == 404, r.text
