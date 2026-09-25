"""Smart Triage & Scheduling: red-flag, slots, NLU, POST /v1/triage.

Covers D5 angles: audit jsonl append, fully-unaccented path, unknown-doctor
fallback, anonymous (no user_id -> 200 + notes + zero DB writes),
missing-info -> followup_question + empty slots. No OPENAI_API_KEY here.
"""
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.seed_synthetic_data import DOCTOR_SPECS  # exact seed strings, never hand-copied


@pytest.fixture
def iso_triage(monkeypatch, tmp_path):
    import src.diabetes.glucose_tracker as gt
    tmp_db = tmp_path / "triage.db"
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
    gt.init_glucose_db()
    meta = tmp_path / "meta"
    meta.mkdir()
    names = ["BS. Nguyen Van A", "BS. Tran Thi B", "BS. Le Van C",
             "BS. Pham Thi D", "BS. Hoang Van E"]
    doctors = [
        {"doctor_id": did, "name": name, "specialty": spec, "working_hours": hours}
        for (did, spec, hours), name in zip(DOCTOR_SPECS, names)
    ]
    (meta / "synthetic_roster.json").write_text(
        json.dumps({"doctors": doctors}, ensure_ascii=False), encoding="utf-8")
    (meta / "doctors_patients.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(meta))
    monkeypatch.setenv("TRIAGE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_BUSY", raising=False)
    return tmp_path


def _roster(tmp_path):
    return json.loads((tmp_path / "meta" / "synthetic_roster.json").read_text(encoding="utf-8"))["doctors"]


def _post(client, message, user_id=None):
    body = {"message": message}
    if user_id is not None:
        body["user_id"] = user_id
    return client.post("/v1/triage", json=body)


class TestRedFlag:
    @pytest.mark.parametrize("msg,group", [
        ("Chóng mặt vã mồ hôi, lơ mơ, hơi thở mùi xeton", "hypo_severe"),
        ("Bàn chân sưng đỏ, chảy mủ, có chỗ hoại tử, sốt cao", "foot_infection"),
        ("Đau ngực dữ dội, méo miệng, yếu nửa người", "cardio_stroke"),
    ])
    def test_groups_accented(self, client, iso_triage, msg, group):
        r = _post(client, msg)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["emergency"] is True
        assert data["red_flag_type"] == group
        assert "115" in (data["message"] or "")
        assert data["recommended_doctors"] == []

    @pytest.mark.parametrize("msg,group", [
        ("va mo hoi, lo mo, hoi tho mui xeton", "hypo_severe"),
        ("ban chan sung do, chay mu, hoai tu, sot cao", "foot_infection"),
        ("dau nguc du doi, meo mieng, yeu nua nguoi", "cardio_stroke"),
    ])
    def test_groups_unaccented(self, client, iso_triage, msg, group):
        r = _post(client, msg)
        assert r.status_code == 200, r.text
        assert r.json()["emergency"] is True
        assert r.json()["red_flag_type"] == group
        assert "115" in (r.json()["message"] or "")

    def test_no_false_positive(self, client, iso_triage):
        for msg in ["đường huyết 140 lúc đói có ổn không",
                    "Tê rần 10 đầu ngón chân, mắt mờ 2 hôm"]:
            r = _post(client, msg)
            assert r.status_code == 200, r.text
            assert r.json()["emergency"] is False, msg

    def test_lone_co_mu_fires_muon_kham_safe(self, client, iso_triage):
        for msg in ["vết thương có mủ", "vet thuong co mu"]:
            r = _post(client, msg)
            assert r.status_code == 200, r.text
            assert r.json()["emergency"] is True, msg
            assert r.json()["red_flag_type"] == "foot_infection", msg
        for msg in ["muốn khám", "muon kham", "muon dat lich kham"]:
            r = _post(client, msg)
            assert r.status_code == 200, r.text
            assert r.json()["emergency"] is False, msg

    def test_short_circuit_no_nlu_call(self, client, iso_triage, monkeypatch):
        import src.triage.triage_nlu as nlu

        def _boom(_text):
            raise AssertionError("NLU must not run on emergency path")

        monkeypatch.setattr(nlu, "parse_triage", _boom)
        r = _post(client, "lơ mơ, vã mồ hôi sau tiêm insulin")
        assert r.status_code == 200, r.text
        assert r.json()["emergency"] is True
        assert r.json()["recommended_doctors"] == []

    def test_short_circuit_no_import(self, client, iso_triage, monkeypatch):
        monkeypatch.setitem(sys.modules, "src.triage.triage_nlu", None)
        monkeypatch.setitem(sys.modules, "src.scheduling.slot_generator", None)
        r = _post(client, "méo miệng, yếu nửa người từ sáng")
        assert r.status_code == 200, r.text
        assert r.json()["emergency"] is True

    def test_audit_append(self, client, iso_triage):
        r = _post(client, "sốt cao, vết loét bàn chân có mủ", user_id="anon_audit")
        assert r.json()["emergency"] is True
        lines = (iso_triage / "audit.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["red_flag_type"] == "foot_infection"
        assert entry["user_id"] == "anon_audit"


class TestSlots:
    def test_seed_strings_open_days(self):
        from src.scheduling.slot_generator import parse_working_hours
        hours = [h for _, _, h in DOCTOR_SPECS]
        assert len(hours) == 5
        for h in hours:
            parsed = parse_working_hours(h)
            assert 0 in parsed and 6 not in parsed  # Mon open, Sun closed
        sat_specs = [h for h in hours if "Sat" in h]
        assert len(sat_specs) == 2
        for h in sat_specs:
            assert 5 in parse_working_hours(h)

    def test_cross_check_availability(self, iso_triage):
        from src.scheduling.slot_generator import parse_working_hours
        from src.scheduling.synthetic_roster import query_doctor_availability
        mon = {d["doctor_id"] for d in query_doctor_availability(day="Mon")}
        sat = {d["doctor_id"] for d in query_doctor_availability(day="Sat")}
        sun = query_doctor_availability(day="Sunday")
        assert len(mon) == 5 and len(sat) == 2 and sun == []
        for did, _spec, hours in DOCTOR_SPECS:
            parsed = parse_working_hours(hours)
            assert (did in sat) == (5 in parsed)  # both sides agree on Sat open
            assert 0 in parsed and 6 not in parsed

    def test_busy_deterministic_hashlib(self):
        from src.scheduling import slot_generator as sg
        a = sg.is_slot_busy("syn_doctor_01", "2026-09-21", "09:30")
        b = sg.is_slot_busy("syn_doctor_01", "2026-09-21", "09:30")
        assert a == b

    def test_exclusion_t4_is_wednesday(self, client, iso_triage):
        r = _post(client, "tôi hay khát nhiều, tiểu nhiều, muốn khám tuần sau trừ T4")
        assert r.status_code == 200, r.text
        recs = r.json()["recommended_doctors"]
        assert recs, "expected fallback proposals"
        for d in recs:
            for s in d["slots"]:
                assert s["weekday"] != "T4", s
                assert date.fromisoformat(s["date"]).weekday() != 2, s  # T4 = Wednesday

    def test_complication_chain_single_doctor(self, client, iso_triage, monkeypatch):
        from src.scheduling import slot_generator as sg
        doctors = _roster(iso_triage)
        comp = [d for d in doctors if d["specialty"] == "Biến chứng ĐTĐ"]
        assert len(comp) == 1  # same-specialty alternative impossible by design
        victim = comp[0]["doctor_id"]
        orig = sg.is_slot_busy

        def _busy_all(doc, *a):
            return True if doc == victim else orig(doc, *a)

        monkeypatch.setattr(sg, "is_slot_busy", _busy_all)
        r = _post(client, "Tê rần 10 đầu ngón chân, mắt mờ 2 hôm")
        assert r.status_code == 200, r.text
        recs = r.json()["recommended_doctors"]
        assert recs, "chain must fall through to other specialties"
        assert all(d["doctor_id"] != victim for d in recs)

    def test_unknown_doctor_fallback(self, client, iso_triage):
        r = _post(client, "đặt lịch khám với BS Không Tồn Tại XYZ sáng mai")
        assert r.status_code == 200, r.text
        assert isinstance(r.json()["recommended_doctors"], list)


class TestEndpoint:
    def test_complication_acceptance(self, client, iso_triage):
        r = _post(client, "Tê rần 10 đầu ngón chân, mắt mờ 2 hôm")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["emergency"] is False
        assert data["suggested_specialty"] == "Endocrinology_Complication"
        assert {s["code"] for s in data["symptoms"]} >= {"neuropathy", "retinopathy"}
        comp = [d for d in data["recommended_doctors"]
                if d["specialty"] == "Biến chứng ĐTĐ" and d["slots"]]
        assert comp, "complication doctor with slots required"

    def test_unaccented_full_path(self, client, iso_triage):
        r = _post(client, "te ran 10 dau ngon chan, mat mo 2 hom")
        assert r.status_code == 200, r.text
        assert r.json()["suggested_specialty"] == "Endocrinology_Complication"

    def test_missing_info(self, client, iso_triage):
        r = _post(client, "xin chào")
        assert r.status_code == 200, r.text
        assert (r.json()["followup_question"] or "").strip() != ""
        assert r.json()["recommended_doctors"] == []

    def test_anonymous_zero_db_writes(self, client, iso_triage):
        import src.diabetes.glucose_tracker as gt
        import sqlite3
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        before = conn.execute("SELECT COUNT(*) FROM glucose_logs").fetchone()[0]
        conn.close()
        r = _post(client, "tê chân, mắt mờ, muốn khám tuần sau")
        assert r.status_code == 200, r.text
        assert (r.json()["previsit_notes"] or "").strip() != ""
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        after = conn.execute("SELECT COUNT(*) FROM glucose_logs").fetchone()[0]
        conn.close()
        assert after == before

    def test_multi_intent(self, client, make_user, iso_triage):
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        pr = client.post("/v1/glucose", json={
            "user_id": user["id"], "value_mgdl": 130, "context": "fasting",
        }, headers=hdr)
        assert pr.status_code == 200, pr.text
        real_name = _roster(iso_triage)[0]["name"]
        real_id = _roster(iso_triage)[0]["doctor_id"]
        r = _post(client, f"hỏi lịch {real_name} sáng mai, dạo này cồn ruột do thuốc cũ",
                  user_id=user["id"])
        assert r.status_code == 200, r.text
        data = r.json()
        assert "cồn ruột" in (data.get("previsit_notes") or "")
        assert data["recommended_doctors"], "booking intent must propose slots"
        assert data["recommended_doctors"][0]["doctor_id"] == real_id
        import src.diabetes.glucose_tracker as gt
        logs = gt.get_logs(user["id"], limit=1)
        assert "cồn ruột" in (logs[0].get("notes") or "")

    def test_no_key_deterministic(self, client, iso_triage):
        r1 = _post(client, "khát nhiều, tiểu nhiều, khám tuần sau trừ T5")
        r2 = _post(client, "khát nhiều, tiểu nhiều, khám tuần sau trừ T5")
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

    def test_notes_truncated(self, client, make_user, iso_triage):
        import src.diabetes.glucose_tracker as gt
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        client.post("/v1/glucose", json={
            "user_id": user["id"], "value_mgdl": 130, "context": "fasting",
        }, headers=hdr)
        long_msg = "tê chân kéo dài, " + "x" * 2000
        r = _post(client, long_msg, user_id=user["id"])
        assert r.status_code == 200, r.text
        notes = gt.get_logs(user["id"], limit=1)[0].get("notes") or ""
        assert len(notes) <= 500, len(notes)


class TestTriageAuthGate:
    """POST /v1/triage dual-write is gated on authenticated identity.

    Synthetic IDs have no auth accounts so they stay response-only; only
    real authenticated users persist to their own id.
    """
    MSG = "tê chân, mắt mờ, muốn khám tuần sau"

    def test_unauthenticated_with_user_id_zero_writes(self, client, make_user, iso_triage):
        import src.diabetes.glucose_tracker as gt
        victim, vtok = make_user(role="user")
        vh = {"Authorization": f"Bearer {vtok}"}
        pr = client.post("/v1/glucose", json={
            "user_id": victim["id"], "value_mgdl": 130, "context": "fasting",
            "notes": "baseline notes",
        }, headers=vh)
        assert pr.status_code == 200, pr.text
        client.cookies.clear()  # session client keeps login cookie; force anonymous
        r = _post(client, self.MSG, user_id=victim["id"])
        assert r.status_code == 200, r.text
        assert (r.json().get("previsit_notes") or "").strip() != ""
        notes = gt.get_logs(victim["id"], limit=1)[0].get("notes") or ""
        assert notes == "baseline notes", notes

    def test_user_role_cannot_write_to_other_user(self, client, make_user, iso_triage):
        import src.diabetes.glucose_tracker as gt
        victim, vtok = make_user(role="user")
        attacker, atok = make_user(role="user")
        vh = {"Authorization": f"Bearer {vtok}"}
        ah = {"Authorization": f"Bearer {atok}"}
        pr = client.post("/v1/glucose", json={
            "user_id": victim["id"], "value_mgdl": 130, "context": "fasting",
            "notes": "victim baseline",
        }, headers=vh)
        assert pr.status_code == 200, pr.text
        r = client.post("/v1/triage",
                        json={"message": self.MSG, "user_id": victim["id"]},
                        headers=ah)
        assert r.status_code == 200, r.text
        assert (r.json().get("previsit_notes") or "").strip() != ""
        notes = gt.get_logs(victim["id"], limit=1)[0].get("notes") or ""
        assert notes == "victim baseline", notes

    def test_own_id_dual_write(self, client, make_user, iso_triage):
        import src.diabetes.glucose_tracker as gt
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        pr = client.post("/v1/glucose", json={
            "user_id": user["id"], "value_mgdl": 130, "context": "fasting",
        }, headers=hdr)
        assert pr.status_code == 200, pr.text
        r = client.post("/v1/triage",
                        json={"message": self.MSG, "user_id": user["id"]},
                        headers=hdr)
        assert r.status_code == 200, r.text
        assert (r.json().get("previsit_notes") or "").strip() != ""
        notes = gt.get_logs(user["id"], limit=1)[0].get("notes") or ""
        assert "tê chân" in notes, notes
