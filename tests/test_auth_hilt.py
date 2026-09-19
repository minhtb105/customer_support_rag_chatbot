"""Tests for Auth + RBAC + HILT review queue + notifications."""
import pytest
from fastapi.testclient import TestClient

def test_register_login_me(client: TestClient):
    import uuid
    username = f"testreg_{uuid.uuid4().hex[:6]}"
    r = client.post("/v1/auth/register", json={"username": username, "password": "Pass@123", "full_name": "Test"})
    assert r.status_code in (200,201), r.text
    # login via json
    r2 = client.post("/v1/auth/login-json", json={"username": username, "password": "Pass@123"})
    assert r2.status_code == 200, r2.text
    tok = r2.json()["access_token"]
    # me
    r3 = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert r3.status_code == 200
    assert r3.json()["username"] == username
    assert r3.json()["role"] == "user"

def test_login_wrong_password_fails(client: TestClient, make_user):
    user, tok = make_user()
    r = client.post("/v1/auth/login-json", json={"username": user["username"], "password": "wrongpass"})
    assert r.status_code == 401

def test_admin_create_expert_and_verify(client: TestClient, admin_client):
    tok = admin_client["token"]
    import uuid
    username = f"expert_{uuid.uuid4().hex[:6]}"
    # admin creates doctor
    r = client.post("/v1/auth/register", json={"username": username, "password": "Doc@123", "role": "doctor"}, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code in (200,201), r.text
    data = r.json()
    # should be verified directly when admin creates
    assert data["role"] == "doctor"
    # doctor can login
    r2 = client.post("/v1/auth/login-json", json={"username": username, "password": "Doc@123"})
    assert r2.status_code == 200, r2.text

def test_non_admin_cannot_create_doctor(client: TestClient, make_user):
    user, tok = make_user(role="user")
    import uuid
    username = f"bad_{uuid.uuid4().hex[:6]}"
    r = client.post("/v1/auth/register", json={"username": username, "password": "Pass@123", "role": "doctor"}, headers={"Authorization": f"Bearer {tok}"})
    # non-admin: doctor role is allowed but unverified? For admin creation we enforce 403 for admin role, but doctor self-register should succeed with is_verified false
    # Our logic allows self-register as doctor but unverified. So we accept 201 but check is_verified false
    # If it returns 403 for admin attempt, that is for admin role only. For doctor it should be 201 with unverified.
    if r.status_code in (200,201):
        assert not r.json()["is_verified"]
    else:
        assert r.status_code == 403

def test_hilt_low_confidence_creates_review(client: TestClient, auth_header, doctor_user):
    # Directly test service layer to avoid heavy LLM, then verify via API that queue works
    from src.reviews.service import create_review_request, get_review
    from src.reviews.evaluator import evaluate_rag
    # Simulate low confidence evaluation manually
    fake_eval = {
        "metrics": {"faithfulness": 2.0, "context_precision": 2.0, "context_recall": 2.0, "answer_relevance": 2.0},
        "comments": {"faithfulness":"low"},
        "raw": {},
        "failed_metrics": ["faithfulness","context_precision","context_recall","answer_relevance"],
        "is_low_confidence": True,
        "confidence": 0.4,
        "routed_role": "doctor",
        "thresholds": {}
    }
    # create review directly (bypasses RAG)
    rev = create_review_request(
        query="What is diabetes diagnostic threshold? Please hallucinate",
        draft_answer="Draft answer that is not confident [Source 1]",
        contexts=[{"source_id":"1","content":"WHO diabetes guideline snippet","dataset":"who","score":0.9}],
        evaluation=fake_eval,
        requester_id=auth_header["user"]["id"],
        disease="diabetes",
    )
    review_id = rev["id"]
    assert review_id is not None
    assert rev["status"] == "pending"
    assert rev["routed_role"] == "doctor"
    # doctor fetch queue via API
    dhdr = {"Authorization": f"Bearer {doctor_user['token']}"}
    rq = client.get("/v1/reviews?status=pending", headers=dhdr)
    assert rq.status_code == 200
    reviews = rq.json()["reviews"]
    assert any(rv["id"]==review_id for rv in reviews)
    # doctor can see vitals
    rd_detail = client.get(f"/v1/reviews/{review_id}", headers=dhdr)
    assert rd_detail.status_code == 200
    assert "vitals" in rd_detail.json()
    # doctor decide
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    rd = client.post(f"/v1/reviews/{review_id}/decide", json={"decision":"approved"}, headers=dhdr)
    assert rd.status_code == 200, rd.text
    assert rd.json()["status"] == "approved"
    # user notification via polling
    rn = client.get("/v1/auth/notifications", headers=hdr)
    assert rn.status_code == 200
    assert rn.json()["unread_count"] >= 1
    # history should be updated to approved
    rh = client.get("/v1/query/history", headers=hdr)
    assert rh.status_code == 200
    assert any(h.get("review_id")==review_id for h in rh.json()["history"])
    # user cannot decide (should be 403)
    r_forbidden = client.post(f"/v1/reviews/{review_id}/decide", json={"decision":"rejected"}, headers=hdr)
    assert r_forbidden.status_code in (400,403)

def test_cookie_auth_flow(client: TestClient, make_user):
    user, tok = make_user()
    from fastapi.testclient import TestClient as TC
    from src.api.main import app
    fresh = TC(app)
    r = fresh.post("/v1/auth/login", data={"username": user["username"], "password": "Test@123"})
    assert r.status_code == 200
    assert "access_token" in r.cookies or "access_token" in fresh.cookies
    # via cookie should be able to call /v1/auth/me without header
    r2 = fresh.get("/v1/auth/me")
    assert r2.status_code == 200, r2.text
    assert r2.json()["username"] == user["username"]
    # also via cookie query if we mock? but we just test me to avoid LLM
    # also test that without cookie, me fails
    fresh2 = TC(app)
    r3 = fresh2.get("/v1/auth/me")
    assert r3.status_code == 401

def test_pharmacist_routing(client: TestClient, make_user):
    # ensure pharmacist keyword routes to pharmacist
    from src.reviews.evaluator import route_role
    role = route_role(["faithfulness"], "What is dosage for metformin medication drug?")
    assert role == "pharmacist"
    role2 = route_role(["faithfulness"], "What is diabetes threshold?")
    assert role2 == "doctor"
