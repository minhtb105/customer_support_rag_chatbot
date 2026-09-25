"""Shared pytest fixtures for diabetes RAG chatbot + Auth + HILT."""
import tempfile
import os
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with temp DB isolation for glucose + vitals + auth (4 diseases)."""
    import src.shared.config as cfg
    tmp_glucose = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp_glucose.close()
    tmp_auth = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp_auth.close()
    tmp_vitals = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp_vitals.close()
    original_glucose = cfg.GLUCOSE_DB_PATH
    original_auth = cfg.AUTH_DB_PATH
    original_vitals = cfg.VITALS_DB_PATH
    cfg.GLUCOSE_DB_PATH = Path(tmp_glucose.name)
    cfg.AUTH_DB_PATH = Path(tmp_auth.name)
    cfg.VITALS_DB_PATH = Path(tmp_vitals.name)
    # Re-init DBs (glucose legacy + unified vitals + auth)
    try:
        from src.diabetes.glucose_tracker import init_glucose_db
        init_glucose_db()
    except Exception:
        pass
    try:
        from src.vitals.bp_tracker import init_bp_db
        init_bp_db()
    except Exception:
        pass
    try:
        from src.vitals.respiratory_tracker import init_respiratory_db
        init_respiratory_db()
    except Exception:
        pass
    try:
        from src.vitals.mood_tracker import init_mood_db
        init_mood_db()
    except Exception:
        pass
    try:
        from src.auth.db import init_auth_db
        init_auth_db()
    except Exception:
        pass
    from src.api.main import app
    with TestClient(app) as c:
        yield c
    cfg.GLUCOSE_DB_PATH = original_glucose
    cfg.AUTH_DB_PATH = original_auth
    cfg.VITALS_DB_PATH = original_vitals
    try:
        os.unlink(tmp_glucose.name)
        os.unlink(tmp_auth.name)
        os.unlink(tmp_vitals.name)
    except Exception:
        pass

@pytest.fixture
def temp_user():
    return f"test_{uuid.uuid4().hex[:8]}"

@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield

# ---------- auth helpers ----------
@pytest.fixture
def make_user(client):
    """Factory to create user via register and return (user_dict, token)"""
    def _make(role="user", username=None, password="Test@123"):
        if username is None:
            username = f"u_{uuid.uuid4().hex[:6]}"
        # register (admin creation for expert needs admin token, but we allow self-register for test user)
        # For expert roles, caller should use admin fixture
        resp = client.post("/v1/auth/register", json={"username": username, "password": password, "role": role if role=="user" else None})
        # If expert, the register above will create unverified expert; for tests we need verified, so create via admin
        if resp.status_code not in (200,201):
            # try with plain
            resp = client.post("/v1/auth/register", json={"username": username, "password": password})
        # login
        r2 = client.post("/v1/auth/login-json", json={"username": username, "password": password})
        assert r2.status_code == 200, r2.text
        tok = r2.json()["access_token"]
        user = r2.json()["user"]
        return user, tok
    return _make

@pytest.fixture(scope="session")
def admin_client(client):
    """Create/admin login once per session"""
    from src.auth.db import get_user_by_username
    from src.auth.security import hash_password
    from src.auth.db import create_user
    # ensure admin exists
    admin_username = f"admin_test_{uuid.uuid4().hex[:4]}"
    pwd = "Admin@123"
    # try create via DB directly to avoid race
    try:
        if not get_user_by_username(admin_username):
            create_user(username=admin_username, email=None, hashed_password=hash_password(pwd), full_name="Admin", role="admin", is_verified=True)
    except Exception:
        pass
    r = client.post("/v1/auth/login-json", json={"username": admin_username, "password": pwd})
    assert r.status_code == 200, r.text
    tok = r.json()["access_token"]
    user = r.json()["user"]
    return {"client": client, "token": tok, "user": user, "username": admin_username, "password": pwd}

@pytest.fixture
def auth_header(make_user):
    """Default user auth header"""
    user, tok = make_user(role="user")
    return {"Authorization": f"Bearer {tok}", "user": user, "token": tok}

@pytest.fixture
def doctor_user(admin_client, client):
    """Create verified doctor via admin"""
    username = f"doc_{uuid.uuid4().hex[:6]}"
    pwd = "Doc@123"
    # admin creates doctor
    tok = admin_client["token"]
    r = client.post("/v1/auth/register", json={"username": username, "password": pwd, "role": "doctor"}, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code in (200,201), r.text
    # doctor may be verified directly by admin (register via admin auto-verified)
    # login doctor
    r2 = client.post("/v1/auth/login-json", json={"username": username, "password": pwd})
    assert r2.status_code == 200, r2.text
    return {"user": r2.json()["user"], "token": r2.json()["access_token"], "username": username}
