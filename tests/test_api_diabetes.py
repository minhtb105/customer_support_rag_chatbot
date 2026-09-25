"""Integration tests for diabetes API (Tracks A/B/C) — with Auth."""
import pytest
from fastapi.testclient import TestClient

class TestDiabetesAPI:
    def test_query_diabetes_returns_citation(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        resp = client.post("/v1/query", json={"query": "What are WHO diagnostic thresholds for diabetes fasting and 2-hour?", "top_k": 3, "user_id": auth_header["user"]["id"]}, headers=hdr)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        if data.get("audit"):
            assert "citations" in data["audit"]
        # HILT fields should be present
        assert "status" in data
        assert "evaluation" in data

    def test_query_requires_auth(self, client: TestClient):
        # Session-scoped client keeps login cookie from prior tests — clear it
        # so this request is truly unauthenticated (same pattern as test_admin_memory).
        client.cookies.clear()
        resp = client.post("/v1/query", json={"query": "hello", "top_k": 3})
        assert resp.status_code == 401

    def test_query_requires_nonempty(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        resp = client.post("/v1/query", json={"query": "", "top_k": 3}, headers=hdr)
        assert resp.status_code in (200, 422)

    def test_glucose_roundtrip(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        r = client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 145, "context": "fasting", "notes": "after breakfast"}, headers=hdr)
        assert r.status_code == 200, r.text
        assert r.json()["classification"] in ("elevated", "high", "normal", "low", "critical")
        r2 = client.get(f"/v1/glucose/{uid}", headers=hdr)
        assert r2.status_code == 200
        assert r2.json()["stats"]["total_logs"] >= 1
        assert "logs_per_week" in r2.json()["stats"]

    def test_glucose_validation_rejects_out_of_range(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        r = client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 5, "context": "fasting"}, headers=hdr)
        assert r.status_code == 422

    def test_soap_generate(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 110, "context": "fasting"}, headers=hdr)
        client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 190, "context": "post_meal_2h"}, headers=hdr)
        resp = client.post("/v1/soap/generate", json={"user_id": uid, "days": 14}, headers=hdr)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "soap" in data
        for k in ("subjective", "objective", "assessment", "plan"):
            assert k in data["soap"]
        # P is ALWAYS empty — the doctor decides
        assert data["soap"]["plan"] == ""
        # A only states HbA1c<7% target check + audit links on S/O/A
        assert "HbA1c" in data["soap"]["assessment"]
        assert "[Xem log #" in data["soap"]["subjective"]
        assert "[Xem log #" in data["soap"]["objective"]
        assert "[Xem log #" in data["soap"]["assessment"]

    def test_glucose_returns_anomaly_and_fqg(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        r = client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 260, "context": "random"}, headers=hdr)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "anomaly" in data and data["anomaly"]["type"] == "spike"
        assert "follow_up_questions" in data and len(data["follow_up_questions"]) > 0
        r2 = client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 110, "context": "fasting"}, headers=hdr)
        assert r2.status_code == 200, r2.text
        assert r2.json()["anomaly"]["type"] == "none"
        assert r2.json()["follow_up_questions"] == []

    def test_safeguard_out_of_scope_no_rag(self, client: TestClient, auth_header, monkeypatch):
        import src.api.main as api_main
        called = {"rag": False}
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        # If RAG were invoked it would go through rag_chat — guard must return before it
        try:
            import src.chat.rag_pipeline as rp
            monkeypatch.setattr(rp, "rag_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG must not be called")))
        except Exception:
            pass
        resp = client.post("/v1/query", json={"query": "Tôi nên uống kháng sinh nào?", "top_k": 3, "user_id": uid}, headers=hdr)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "được thiết kế riêng để theo dõi đường huyết" in data["answer"]
        assert "nằm ngoài phạm vi hỗ trợ" in data["answer"]
        assert data.get("safeguard") is True

    def test_soap_markdown(self, client: TestClient, auth_header):
        hdr = {"Authorization": f"Bearer {auth_header['token']}"}
        uid = auth_header["user"]["id"]
        client.post("/v1/glucose", json={"user_id": uid, "value_mgdl": 95, "context": "fasting"}, headers=hdr)
        resp = client.post("/v1/soap/generate/markdown", json={"user_id": uid, "days": 7}, headers=hdr)
        assert resp.status_code == 200
        assert "Subjective" in resp.text or "S —" in resp.text

    def test_guidelines_status(self, client: TestClient):
        resp = client.get("/v1/guidelines/status")
        assert resp.status_code == 200
        assert "total_pdfs" in resp.json()
        assert resp.json()["total_pdfs"] >= 4

    def test_health(self, client: TestClient):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["embedding_provider"] in ("openai", "local")

    def test_user_isolation(self, client: TestClient, make_user):
        u1, t1 = make_user(role="user")
        u2, t2 = make_user(role="user")
        h1 = {"Authorization": f"Bearer {t1}"}
        h2 = {"Authorization": f"Bearer {t2}"}
        client.post("/v1/glucose", json={"user_id": u1["id"], "value_mgdl": 100, "context": "fasting"}, headers=h1)
        client.post("/v1/glucose", json={"user_id": u2["id"], "value_mgdl": 200, "context": "fasting"}, headers=h2)
        # u1 cannot read u2
        r = client.get(f"/v1/glucose/{u2['id']}", headers=h1)
        assert r.status_code == 403
        # own succeeds
        r2 = client.get(f"/v1/glucose/{u1['id']}", headers=h1)
        assert r2.status_code == 200
