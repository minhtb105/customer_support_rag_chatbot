"""Triage monitor + trailing-tag (D1/D2): pure helpers + admin endpoint.

Scale rule: tests call pure functions, never loop POST /v1/triage.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def iso_env(monkeypatch, tmp_path):
    import src.features.glucose_tracker as gt
    tmp_db = tmp_path / "mon.db"
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
    gt.init_glucose_db()
    monkeypatch.setenv("TRIAGE_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


class TestTrailingTag:
    def test_segment_exact_true(self):
        from src.features.followup_notes import has_triage_marker
        assert has_triage_marker("note A | [AI Triage]") is True
        assert has_triage_marker("[AI Triage]") is True

    def test_substring_not_enough(self):
        from src.features.followup_notes import has_triage_marker
        assert has_triage_marker("foo [AI Triage] bar") is False
        assert has_triage_marker("x[AI Triage]") is False
        assert has_triage_marker("") is False
        assert has_triage_marker(None) is False

    def test_old_notes_untagged(self):
        from src.features.followup_notes import has_triage_marker
        # Legacy FQG/seed notes carry no marker.
        assert has_triage_marker("ngủ ngon | ăn bún chả") is False
        assert has_triage_marker("hơi mệt") is False

    def test_triage_source_tags_tail(self, iso_env):
        from src.features import glucose_tracker as gt
        from src.features.followup_notes import save_followup_notes, has_triage_marker
        rec = gt.add_log("tag_user_01", 120, context="fasting", notes="baseline notes")
        out = save_followup_notes("tag_user_01", "tê chân, mắt mờ", rec["id"], source="triage")
        assert out["saved_note"] is True
        rows = gt.get_logs("tag_user_01", limit=5)
        notes = rows[0]["notes"]
        assert notes.endswith(" [AI Triage]")
        assert has_triage_marker(notes) is True
        assert len(notes) <= 500

    def test_long_body_keeps_tag(self, iso_env):
        from src.features import glucose_tracker as gt
        from src.features.followup_notes import save_followup_notes
        rec = gt.add_log("tag_user_02", 120, context="fasting", notes="x" * 480)
        save_followup_notes("tag_user_02", "y" * 100, rec["id"], source="triage")
        notes = gt.get_logs("tag_user_02", limit=5)[0]["notes"]
        assert notes.endswith(" [AI Triage]")
        assert len(notes) <= 500

    def test_followup_default_untagged(self, iso_env):
        from src.features import glucose_tracker as gt
        from src.features.followup_notes import save_followup_notes, has_triage_marker
        rec = gt.add_log("tag_user_03", 120, context="fasting", notes="prev")
        save_followup_notes("tag_user_03", "ăn 2 miếng bánh ngọt", rec["id"])
        notes = gt.get_logs("tag_user_03", limit=5)[0]["notes"]
        assert has_triage_marker(notes) is False
        assert "[AI Triage]" not in notes


class TestTriageEventsPure:
    def test_excerpt_capped_200(self, iso_env):
        from src.features import triage_events as te
        assert te.append_triage_event(user_id="u1", message="z" * 500) is True
        got = te.list_triage_events()
        assert got["total"] == 1
        assert len(got["events"][0]["excerpt"]) == 200

    def test_filters_and_pagination(self, iso_env):
        from src.features import triage_events as te
        te.append_triage_event(user_id="u1", message="tê chân", specialty="Endocrinology_General")
        te.append_triage_event(user_id="u2", message="vã mồ hôi lơ mơ", emergency=True,
                               red_flag_type="hypo_severe")
        te.append_triage_event(user_id="u3", message="tê chân nặng", specialty="Endocrinology_Complication")
        assert te.list_triage_events(specialty="Endocrinology_Complication")["total"] == 1
        assert te.list_triage_events(emergency=True)["total"] == 1
        assert te.list_triage_events(emergency=False)["total"] == 2
        assert te.list_triage_events(q="nặng")["total"] == 1
        p1 = te.list_triage_events(page=1, limit=2)
        p2 = te.list_triage_events(page=2, limit=2)
        assert (p1["total"], p1["page"], p1["limit"]) == (3, 1, 2)
        assert len(p1["events"]) == 2 and len(p2["events"]) == 1


class TestTriageEventsEndpoint:
    def test_401_no_token(self, client: TestClient, iso_env):
        client.cookies.clear()
        r = client.get("/v1/admin/triage/events")
        assert r.status_code == 401, r.text

    def test_403_user_role(self, client: TestClient, make_user, iso_env):
        user, tok = make_user(role="user")
        r = client.get("/v1/admin/triage/events", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 403, r.text

    def test_200_admin_paged(self, client: TestClient, admin_client, iso_env):
        from src.features import triage_events as te
        te.append_triage_event(user_id="u9", message="đau ngực dữ dội", emergency=True,
                               red_flag_type="cardio_stroke")
        hdr = {"Authorization": f"Bearer {admin_client['token']}"}
        r = client.get("/v1/admin/triage/events?page=1&limit=20", headers=hdr)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] >= 1 and body["page"] == 1
        r2 = client.get("/v1/admin/triage/events?emergency=true&limit=5", headers=hdr)
        assert r2.status_code == 200 and all(e["emergency"] for e in r2.json()["events"])
        r3 = client.get("/v1/admin/triage/events?limit=200", headers=hdr)
        assert r3.status_code == 422  # max 100
