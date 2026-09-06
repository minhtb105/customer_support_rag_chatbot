"""Shared pytest fixtures for diabetes RAG chatbot."""

import tempfile
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with temp glucose DB isolation."""
    # Isolate glucose DB to temp file so tests don't pollute metadata/glucose_logs.db
    import src.config as cfg
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    original = cfg.GLUCOSE_DB_PATH
    cfg.GLUCOSE_DB_PATH = Path(tmp.name)
    # Re-init DB
    try:
        from src.features.glucose_tracker import init_glucose_db
        init_glucose_db()
    except Exception:
        pass
    from src.api.main import app
    with TestClient(app) as c:
        yield c
    # cleanup
    cfg.GLUCOSE_DB_PATH = original
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


@pytest.fixture
def temp_user():
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_openai(monkeypatch):
    """Mock OpenAI client to avoid real API calls in unit tests."""
    # For glucose_tracker/soap_summary that call OpenAI, we mock via env unset
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield
