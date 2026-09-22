"""Doctor triage queue (glucose-only): GET /v1/doctor/patients.

Covers: 401 no-token, 403 role=user, 200 sort
critical 280 > trend 110/125/142 distinct-day > safe 100,
per-user fail-open, glucose-only keys, no email leak.
"""
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def iso_glucose(monkeypatch):
    import src.features.glucose_tracker as gt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", Path(tmp.name))
    gt.init_glucose_db()
    yield Path(tmp.name)
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


def _iso_day(offset: int) -> str:
    base = datetime.now(timezone.utc).replace(hour=7, minute=0, second=0, microsecond=0)
    return (base - timedelta(days=offset)).isoformat()


def _push(client: TestClient, hdr: dict, uid: str, value: float, off: int, notes=None):
    r = client.post("/v1/glucose", json={
        "user_id": uid, "value_mgdl": value, "context": "fasting",
        "measured_at": _iso_day(off), "notes": notes,
    }, headers=hdr)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _seed_three(client: TestClient, make_user):
    u_crit, t_crit = make_user(role="user")
    u_trend, t_trend = make_user(role="user")
    u_safe, t_safe = make_user(role="user")
    _push(client, {"Authorization": f"Bearer {t_crit}"}, u_crit["id"], 280, 0)
    for v, off in ((110, 2), (125, 1), (142, 0)):
        _push(client, {"Authorization": f"Bearer {t_trend}"}, u_trend["id"], v, off)
    _push(client, {"Authorization": f"Bearer {t_safe}"}, u_safe["id"], 100, 0)
    return u_crit, u_trend, u_safe


class TestDoctorPatients:
    def test_401_no_token(self, client: TestClient, iso_glucose):
        r = client.get("/v1/doctor/patients")
        assert r.status_code == 401, r.text

    def test_403_user_role(self, client: TestClient, make_user, iso_glucose):
        user, tok = make_user(role="user")
        r = client.get("/v1/doctor/patients", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 403, r.text

    def test_sort_critical_trend_safe(self, client: TestClient, make_user, doctor_user, iso_glucose):
        u_crit, u_trend, u_safe = _seed_three(client, make_user)
        hdr = {"Authorization": f"Bearer {doctor_user['token']}"}
        r = client.get("/v1/doctor/patients", headers=hdr)
        assert r.status_code == 200, r.text
        patients = r.json()["patients"]
        assert [p["user_id"] for p in patients] == [u_crit["id"], u_trend["id"], u_safe["id"]]
        assert [p["level"] for p in patients] == ["critical", "trend", "safe"]

    def test_fail_open_one_user_broken(self, client: TestClient, make_user, doctor_user, iso_glucose, monkeypatch):
        u_crit, u_trend, u_safe = _seed_three(client, make_user)
        import src.features.glucose_tracker as gt
        orig = gt.get_logs

        def flaky(uid, limit=50, days=None):
            if uid == u_trend["id"]:
                raise RuntimeError("boom")
            return orig(uid, limit=limit, days=days)

        monkeypatch.setattr(gt, "get_logs", flaky)
        hdr = {"Authorization": f"Bearer {doctor_user['token']}"}
        r = client.get("/v1/doctor/patients", headers=hdr)
        assert r.status_code == 200, r.text
        patients = r.json()["patients"]
        assert len(patients) == 3
        row = next(p for p in patients if p["user_id"] == u_trend["id"])
        assert row["level"] == "safe"

    def test_glucose_only_keys_no_email(self, client: TestClient, make_user, doctor_user, iso_glucose):
        _seed_three(client, make_user)
        hdr = {"Authorization": f"Bearer {doctor_user['token']}"}
        r = client.get("/v1/doctor/patients", headers=hdr)
        assert r.status_code == 200, r.text
        assert "email" not in r.text
        allowed = {"user_id", "username", "last_value", "last_classification",
                   "last_measured_at", "level", "should_escalate", "anomaly_count"}
        for p in r.json()["patients"]:
            assert set(p.keys()) <= allowed, set(p.keys()) - allowed
