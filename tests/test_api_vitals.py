"""Integration tests for vitals tracks B/C/D — BP, Respiratory, Mood, Vitals unified, SOAP disease."""
import pytest
from fastapi.testclient import TestClient

# ---------- BP ----------

def test_bp_roundtrip(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    r = client.post("/v1/bp", json={"user_id": uid, "systolic": 135, "diastolic": 85, "context": "random", "notes": "after coffee"}, headers=hdr)
    assert r.status_code == 200, r.text
    assert r.json()["classification"] in ("elevated", "stage1", "stage2", "normal", "crisis")
    r2 = client.get(f"/v1/bp/{uid}", headers=hdr)
    assert r2.status_code == 200
    assert r2.json()["stats"]["total_logs"] >= 1
    assert "at_target_rate" in r2.json()["stats"]
    assert "should_escalate" in r2.json()


def test_bp_validation_out_of_range(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    r = client.post("/v1/bp", json={"user_id": uid, "systolic": 500, "diastolic": 80}, headers=hdr)
    assert r.status_code in (400, 422)
    r2 = client.post("/v1/bp", json={"user_id": uid, "systolic": 120, "diastolic": 5}, headers=hdr)
    assert r2.status_code in (400, 422)


def test_bp_escalation_flow(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    # 3 x stage2 should escalate
    for _ in range(3):
        client.post("/v1/bp", json={"user_id": uid, "systolic": 150, "diastolic": 95}, headers=hdr)
    r = client.get(f"/v1/bp/{uid}", headers=hdr)
    assert r.json()["should_escalate"] is True


# ---------- Respiratory ----------

def test_respiratory_roundtrip(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    r = client.post("/v1/respiratory", json={"user_id": uid, "peak_flow_percent": 85, "cat_score": 8, "inhaler_correct": True}, headers=hdr)
    assert r.status_code == 200, r.text
    assert r.json()["classification"] in ("green", "yellow", "red", "unknown")
    r2 = client.get(f"/v1/respiratory/{uid}", headers=hdr)
    assert r2.status_code == 200
    assert r2.json()["stats"]["total_logs"] >= 1
    assert "red_rate" in r2.json()["stats"]


def test_respiratory_red_escalation(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    client.post("/v1/respiratory", json={"user_id": uid, "peak_flow_percent": 40, "inhaler_correct": False}, headers=hdr)
    r = client.get(f"/v1/respiratory/{uid}", headers=hdr)
    assert r.json()["should_escalate"] is True


# ---------- Mood ----------

def test_mood_roundtrip_and_pii(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    # PII phone should be redacted
    r = client.post("/v1/mood", json={"user_id": uid, "phq9_score": 12, "gad7_score": 8, "mood_notes": "feel bad phone 0912345678 email a@b.com"}, headers=hdr)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["classification"] in ("minimal", "mild", "moderate", "severe", "moderately_severe", "crisis", "unknown")
    assert data["original_notes_had_pii"] is True
    assert "[REDACTED_PHONE]" in data["mood_notes"]
    assert "[REDACTED_EMAIL]" in data["mood_notes"]
    r2 = client.get(f"/v1/mood/{uid}", headers=hdr)
    assert r2.status_code == 200
    assert r2.json()["stats"]["total_logs"] >= 1
    assert "crisis_count" in r2.json()["stats"]


def test_mood_crisis_escalation(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    r = client.post("/v1/mood", json={"user_id": uid, "phq9_score": 5, "gad7_score": 5, "mood_notes": "tôi muốn tự tử"}, headers=hdr)
    assert r.json()["crisis_flag"] is True
    assert r.json()["classification"] == "crisis"
    r2 = client.get(f"/v1/mood/{uid}", headers=hdr)
    assert r2.json()["should_escalate"] is True


def test_mood_validation(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    r = client.post("/v1/mood", json={"user_id": uid, "phq9_score": 30, "gad7_score": 5}, headers=hdr)
    assert r.status_code in (400, 422)
    r2 = client.post("/v1/mood", json={"user_id": uid, "phq9_score": 5, "gad7_score": 30}, headers=hdr)
    assert r2.status_code in (400, 422)


# ---------- Vitals unified ----------

def test_vitals_unified_each_disease(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    # diabetes via vitals
    r = client.post("/v1/vitals", json={"user_id": uid, "disease": "diabetes", "value_mgdl": 130, "context": "fasting"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["disease"] == "diabetes"
    # hypertension
    r = client.post("/v1/vitals", json={"user_id": uid, "disease": "hypertension", "systolic": 135, "diastolic": 85}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["disease"] == "hypertension"
    # respiratory
    r = client.post("/v1/vitals", json={"user_id": uid, "disease": "respiratory", "peak_flow_percent": 75, "cat_score": 12}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["disease"] == "respiratory"
    # mental
    r = client.post("/v1/vitals", json={"user_id": uid, "disease": "mental", "phq9_score": 8, "gad7_score": 7, "notes": "ok"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["disease"] == "mental"
    # invalid disease -> 422 validation (Pydantic Literal) or 400 from handler
    r = client.post("/v1/vitals", json={"user_id": uid, "disease": "unknown"}, headers=hdr)
    assert r.status_code in (400, 422)


# ---------- SOAP disease param ----------

def test_soap_hypertension(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    client.post("/v1/bp", json={"user_id": uid, "systolic": 150, "diastolic": 95}, headers=hdr)
    resp = client.post("/v1/soap/generate", json={"user_id": uid, "days": 14, "disease": "hypertension"}, headers=hdr)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "soap" in data
    # P (plan) intentionally stays "" — doctor decides (see soap_summary.py rule_based + LLM enforce).
    for k in ("subjective", "objective", "assessment"):
        assert k in data["soap"]
        assert len(data["soap"][k]) > 0
    assert "systolic" in data["soap"]["subjective"].lower() or "mmHg" in data["soap"]["objective"] or "huyết áp" in data["soap"]["subjective"].lower()


def test_soap_respiratory(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    client.post("/v1/respiratory", json={"user_id": uid, "peak_flow_percent": 60, "cat_score": 15}, headers=hdr)
    resp = client.post("/v1/soap/generate", json={"user_id": uid, "days": 14, "disease": "respiratory"}, headers=hdr)
    assert resp.status_code == 200
    assert "soap" in resp.json()


def test_soap_mental(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    client.post("/v1/mood", json={"user_id": uid, "phq9_score": 15, "gad7_score": 12, "mood_notes": "stress"}, headers=hdr)
    resp = client.post("/v1/soap/generate", json={"user_id": uid, "days": 14, "disease": "mental"}, headers=hdr)
    assert resp.status_code == 200
    assert resp.json()["disease"] == "mental"
    # markdown should contain PHQ9/GAD7
    md = client.post("/v1/soap/generate/markdown", json={"user_id": uid, "days": 14, "disease": "mental"}, headers=hdr)
    assert md.status_code == 200
    assert "PHQ" in md.text or "GAD" in md.text or "Crisis" in md.text


def test_soap_markdown_disease(client: TestClient, auth_header):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    uid = auth_header["user"]["id"]
    client.post("/v1/bp", json={"user_id": uid, "systolic": 120, "diastolic": 80}, headers=hdr)
    md = client.post("/v1/soap/generate/markdown", json={"user_id": uid, "days": 7, "disease": "hypertension"}, headers=hdr)
    assert md.status_code == 200
    assert "hypertension" in md.text.lower() or "huyết áp" in md.text.lower()


# ---------- Isolation for vitals ----------

def test_vitals_user_isolation(client: TestClient, make_user):
    u1, t1 = make_user(role="user")
    u2, t2 = make_user(role="user")
    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    client.post("/v1/bp", json={"user_id": u1["id"], "systolic": 120, "diastolic": 80}, headers=h1)
    client.post("/v1/bp", json={"user_id": u2["id"], "systolic": 150, "diastolic": 95}, headers=h2)
    # u1 cannot read u2 bp
    r = client.get(f"/v1/bp/{u2['id']}", headers=h1)
    assert r.status_code == 403
    # mood isolation
    client.post("/v1/mood", json={"user_id": u1["id"], "phq9_score": 5}, headers=h1)
    r = client.get(f"/v1/mood/{u2['id']}", headers=h1)
    assert r.status_code == 403
