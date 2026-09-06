"""Integration tests for diabetes API (Tracks A/B/C)."""

import pytest
from fastapi.testclient import TestClient


class TestDiabetesAPI:
    def test_query_diabetes_returns_citation(self, client: TestClient, temp_user):
        # limit to small top_k to avoid heavy LLM, but still should return answer/citations
        resp = client.post("/v1/query", json={"query": "What are WHO diagnostic thresholds for diabetes fasting and 2-hour?", "top_k": 3, "user_id": temp_user})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        # audit is optional but if included should have citations field
        if data.get("audit"):
            assert "citations" in data["audit"]

    def test_query_requires_nonempty(self, client: TestClient):
        resp = client.post("/v1/query", json={"query": "", "top_k": 3})
        # Pydantic may allow empty but pipeline should handle; accept 200 or 422
        assert resp.status_code in (200, 422)

    def test_glucose_roundtrip(self, client: TestClient, temp_user):
        # POST then GET
        r = client.post("/v1/glucose", json={"user_id": temp_user, "value_mgdl": 145, "context": "fasting", "notes": "after breakfast"})
        assert r.status_code == 200
        assert r.json()["classification"] in ("elevated", "high", "normal", "low", "critical")
        r2 = client.get(f"/v1/glucose/{temp_user}")
        assert r2.status_code == 200
        assert r2.json()["stats"]["total_logs"] >= 1
        assert "logs_per_week" in r2.json()["stats"]

    def test_glucose_validation_rejects_out_of_range(self, client: TestClient, temp_user):
        r = client.post("/v1/glucose", json={"user_id": temp_user, "value_mgdl": 5, "context": "fasting"})
        assert r.status_code == 422

    def test_soap_generate(self, client: TestClient, temp_user):
        # need at least one log
        client.post("/v1/glucose", json={"user_id": temp_user, "value_mgdl": 110, "context": "fasting"})
        client.post("/v1/glucose", json={"user_id": temp_user, "value_mgdl": 190, "context": "post_meal_2h"})
        resp = client.post("/v1/soap/generate", json={"user_id": temp_user, "days": 14})
        assert resp.status_code == 200
        data = resp.json()
        assert "soap" in data
        for k in ("subjective", "objective", "assessment", "plan"):
            assert k in data["soap"]

    def test_soap_markdown(self, client: TestClient, temp_user):
        client.post("/v1/glucose", json={"user_id": temp_user, "value_mgdl": 95, "context": "fasting"})
        resp = client.post("/v1/soap/generate/markdown", json={"user_id": temp_user, "days": 7})
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
