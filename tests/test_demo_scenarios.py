"""Demo scenario tests KB1-KB4 (backend layer, isolated tmp DB).

Covers the exact 3-5 minute video flow:
KB1 spike 280 fasting / KB2 trend cascade 110->125->142 + followup
"banh ngot" / KB3 safeguard khang sinh / KB4 anomaly_ids + SOAP S/A/P.
"""
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def iso_glucose(monkeypatch):
    """Isolate glucose DB per test + reset episodic buffer."""
    import src.diabetes.glucose_tracker as gt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", Path(tmp.name))
    gt.init_glucose_db()
    try:
        from src.shared.memory.episodic import get_episodic_memory
        get_episodic_memory().conversation_buffer.clear()
        get_episodic_memory().summaries.clear()
    except Exception:
        pass
    yield Path(tmp.name)
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


def _iso_day(offset: int) -> str:
    base = datetime.now(timezone.utc).replace(hour=7, minute=0, second=0, microsecond=0)
    return (base - timedelta(days=offset)).isoformat()


def _push_cascade(client: TestClient, hdr: dict, uid: str):
    """Push 110/125/142 fasting on 3 distinct days; last carries 'banh ngot' note."""
    ids = []
    last = None
    for value, off, notes in (
        (110, 2, None),
        (125, 1, None),
        (142, 0, "an hai mieng banh ngot toi qua"),
    ):
        r = client.post("/v1/glucose", json={
            "user_id": uid, "value_mgdl": value, "context": "fasting",
            "measured_at": _iso_day(off), "notes": notes,
        }, headers=hdr)
        assert r.status_code == 200, r.text
        ids.append(r.json()["id"])
        last = r
    return ids, last


class TestDemoScenarios:
    def test_kb1_spike_280(self, client: TestClient, make_user, iso_glucose):
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        r = client.post("/v1/glucose", json={
            "user_id": user["id"], "value_mgdl": 280,
            "context": "fasting", "measured_at": _iso_day(0),
        }, headers=hdr)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["anomaly"]["type"] == "spike"
        assert len(data["follow_up_questions"]) > 0
        assert data["classification"] in ("high", "critical")

    def test_kb2_trend_cascade_and_followup(self, client: TestClient, make_user, iso_glucose):
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        ids, last = _push_cascade(client, hdr, user["id"])
        # the third push triggers the trend on screen
        assert last.json()["anomaly"]["type"] == "trend"
        assert len(last.json()["follow_up_questions"]) > 0
        # followup dual-write: notes + episodic buffer
        f = client.post("/v1/glucose/followup", json={
            "user_id": user["id"], "text": "an hai mieng banh ngot",
            "related_log_id": ids[-1],
        }, headers=hdr)
        assert f.status_code == 200, f.text
        assert f.json()["saved"] is True
        from src.shared.memory.episodic import get_episodic_memory
        buf = get_episodic_memory().conversation_buffer
        assert any("banh ngot" in (m.get("content") or "") for m in buf)
        g = client.get(f"/v1/glucose/{user['id']}", headers=hdr)
        note = next(l for l in g.json()["logs"] if l["id"] == ids[-1])["notes"]
        assert note and "banh ngot" in note

    def test_kb3_safeguard_no_rag(self, client: TestClient, make_user, iso_glucose, monkeypatch):
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        try:
            import src.chat.rag_pipeline as rp
            monkeypatch.setattr(rp, "rag_chat",
                                lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG must not be called")))
        except Exception:
            pass
        resp = client.post("/v1/query", json={
            "query": "ho co dom, nen uong khang sinh loai nao ket hop thuoc tieu duong?",
            "top_k": 3, "user_id": user["id"],
        }, headers=hdr)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "theo dõi đường huyết" in data["answer"]
        assert "nằm ngoài phạm vi hỗ trợ" in data["answer"]
        assert data.get("safeguard") is True

    def test_kb4_anomaly_ids_and_soap(self, client: TestClient, make_user, iso_glucose, mock_openai):
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        ids, _ = _push_cascade(client, hdr, user["id"])
        g = client.get(f"/v1/glucose/{user['id']}?limit=50", headers=hdr)
        assert g.status_code == 200, g.text
        anomaly_ids = g.json().get("anomaly_ids", [])
        for _id in ids:
            assert _id in anomaly_ids
        resp = client.post("/v1/soap/generate",
                           json={"user_id": user["id"], "days": 90}, headers=hdr)
        assert resp.status_code == 200, resp.text
        soap = resp.json()["soap"]
        assert "banh ngot" in soap["subjective"]
        assert "[Xem log #" in soap["subjective"]
        assert "HbA1c" in soap["assessment"]
        assert soap["plan"] == ""
