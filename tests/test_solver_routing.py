"""Dynamic Solver Routing: greedy / DFS-CSP / GA + NLU extensions + /v1/triage wire.

Pins D1-D6: simulation skip-write, filters-always-apply + echo, BHYT cost
warning + never-specialty, GA schema/timing/zero-writes, slot_generator
untouched (gated by test_triage 27/27), single solvers.py module.
No OPENAI_API_KEY here: fully deterministic.
"""
import json
import time
from datetime import date
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.seed_synthetic_data import DOCTOR_SPECS  # exact seed strings, never hand-copied

NAMES = ["BS. Nguyen Van A", "BS. Tran Thi B", "BS. Le Van C",
         "BS. Pham Thi D", "BS. Hoang Van E"]


@pytest.fixture
def iso_solver(monkeypatch, tmp_path):
    import src.features.glucose_tracker as gt
    tmp_db = tmp_path / "solver.db"
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
    gt.init_glucose_db()
    meta = tmp_path / "meta"
    meta.mkdir()
    doctors = [
        {"doctor_id": did, "name": name, "specialty": spec, "working_hours": hours}
        for (did, spec, hours), name in zip(DOCTOR_SPECS, NAMES)
    ]
    (meta / "synthetic_roster.json").write_text(
        json.dumps({"doctors": doctors}, ensure_ascii=False), encoding="utf-8")
    (meta / "doctors_patients.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(meta))
    monkeypatch.setenv("TRIAGE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_BUSY", raising=False)
    return tmp_path


def _post(client, message, user_id=None, headers=None):
    body = {"message": message}
    if user_id is not None:
        body["user_id"] = user_id
    return client.post("/v1/triage", json=body, headers=headers)


class TestNLUExtensions:
    @pytest.mark.parametrize("msg", [
        "muốn khám có BHYT", "muon kham co BHYT",
        "có bảo hiểm y tế", "co bao hiem y te",
    ])
    def test_bhyt_variants(self, iso_solver, msg):
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage(msg)
        assert nlu["has_bhyt"] is True, msg
        assert "bhyt" in nlu["constraints"], msg

    @pytest.mark.parametrize("msg,tier", [
        ("đặt lịch ở Bạch Mai", "central"),
        ("dat lich o Bach Mai", "central"),
        ("khám tuyến huyện", "district"),
        ("kham tuyen huyen", "district"),
        ("muốn khám trung ương", "central"),
        ("muon kham trung uong", "central"),
    ])
    def test_branch_variants(self, iso_solver, msg, tier):
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage(msg)
        assert nlu["tier"] == tier, msg
        assert "branch" in nlu["constraints"], msg

    @pytest.mark.parametrize("msg", ["trừ 10h-11h, muốn khám", "tru 10h-11h, muon kham"])
    def test_window_variants(self, iso_solver, msg):
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage(msg)
        assert nlu["time_constraints"].get("excluded_window") == {
            "start": "10:00", "end": "11:00"}, msg
        assert "window" in nlu["constraints"], msg

    def test_doctor_set_and_simulation(self, iso_solver):
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage(f"Đặt lịch {NAMES[3]} hoặc {NAMES[4]} sáng mai")
        assert nlu["requested_doctors"] == ["syn_doctor_04", "syn_doctor_05"]
        assert nlu["requested_doctor"] == "syn_doctor_04"  # backward-compat: first
        nlu = parse_triage(f"Mô phỏng {NAMES[0]} nghỉ 3 ngày")
        assert nlu["simulation"]["is_simulation"] is True
        assert nlu["simulation"]["days_off"] == 3
        assert nlu["simulation"]["doctor_id"] == "syn_doctor_01"
        nlu = parse_triage(f"Mo phong {NAMES[1]} nghi 2 ngay")
        assert nlu["simulation"]["is_simulation"] is True
        assert nlu["simulation"]["days_off"] == 2

    def test_router_boundary_1_vs_2(self, iso_solver):
        from src.features.solvers import SchedulerOrchestrator
        from src.features.triage_nlu import parse_triage
        one = parse_triage("muốn khám có BHYT")
        assert len([c for c in one["constraints"]]) == 1
        assert SchedulerOrchestrator.route(one)[0] == "greedy"
        two = parse_triage("muốn khám có BHYT, trừ T4")
        assert len([c for c in two["constraints"]]) >= 2
        assert SchedulerOrchestrator.route(two)[0] == "dfs"


class TestAttributes:
    def test_calibration_mix(self, iso_solver):
        from src.features.solvers import doctor_attributes
        ids = [did for did, _, _ in DOCTOR_SPECS]
        attrs = [doctor_attributes(d) for d in ids]
        assert sum(1 for a in attrs if a["distance_km"] < 5) >= 2
        assert {a["accepts_bhyt"] for a in attrs} == {True, False}
        assert {a["tier"] for a in attrs} == {"central", "district"}


class TestGreedy:
    def test_nearby_endpoint(self, client, iso_solver):
        t0 = time.perf_counter()
        r = _post(client, "BS nội tiết gần nhà nhất, muốn khám chiều nay")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["solver_used"] == "greedy"
        assert "constraints=[nearby]" in (data["routing_reason"] or "")
        assert data["recommended_doctors"], "nearby must propose"
        for d in data["recommended_doctors"]:
            assert d["distance_km"] < 5, d
        assert elapsed < 5, elapsed  # budget smoke (solver itself is ms-scale)

    def test_greedy_solver_timing_and_filters(self, iso_solver):
        import time as _t
        from src.features.solvers import GreedySolver
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage("muốn khám có BHYT")
        t0 = _t.perf_counter()
        res = GreedySolver.solve(nlu)
        # Budget <50ms; reviewer measured ~6.0ms live on pure GreedySolver.solve (8x margin).
        assert _t.perf_counter() - t0 < 0.05
        for d in res["recommended_doctors"]:
            assert d["accepts_bhyt"] is True, d  # parsed filter always applies (D2)
        assert "constraints=[bhyt]" in res["routing_reason"]


class TestDFS:
    MSG_EXACT = (f"Đặt lịch {NAMES[3]} hoặc {NAMES[4]} ở Bạch Mai, "
                 "tuần sau trừ T4, trừ 10h-11h, có BHYT")

    def test_exact_match(self, client, iso_solver):
        r = _post(client, self.MSG_EXACT)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["solver_used"] == "dfs"
        recs = data["recommended_doctors"]
        assert recs, "exact case must match 100%"
        for d in recs:
            assert d["accepts_bhyt"] is True, d
            assert d["tier"] == "central", d
            for s in d["slots"]:
                assert s["weekday"] != "T4", s
                assert date.fromisoformat(s["date"]).weekday() != 2, s
                assert not ("10:00" <= s["time"] < "11:00"), s
        assert "100%" in (data["routing_reason"] or "")

    def test_relax_bhyt_warning(self, client, iso_solver):
        r = _post(client, f"Tê rần ngón chân, mắt mờ, đặt lịch {NAMES[2]} ở Bạch Mai, có BHYT")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["solver_used"] == "dfs"
        assert data["recommended_doctors"], "relax must still propose"
        assert "không đảm bảo BHYT — có thể phát sinh chi phí" in (data["routing_reason"] or "")

    def test_relax_order_never_specialty(self, iso_solver):
        from src.features.solvers import DFSSolver
        from src.features.triage_nlu import parse_triage
        nlu = parse_triage(f"Tê rần ngón chân, mắt mờ, đặt lịch {NAMES[2]} ở Bạch Mai, có BHYT")
        res = DFSSolver.solve(nlu)
        assert res["relaxed"] == ["doctor", "branch", "bhyt"]
        assert "specialty" not in res["relaxed"]
        assert res["recommended_doctors"]


class TestGA:
    def test_simulation_anonymous(self, client, iso_solver):
        import src.features.glucose_tracker as gt
        import sqlite3
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        before = conn.execute("SELECT COUNT(*) FROM glucose_logs").fetchone()[0]
        conn.close()
        t0 = time.perf_counter()
        r = _post(client, f"Mô phỏng {NAMES[0]} nghỉ 3 ngày cho BN tê chân")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["emergency"] is False
        assert data["solver_used"] == "ga"
        summary = data["simulation_summary"]
        for key in ("reassigned", "unplaced", "avg_shift_days", "fitness"):
            assert key in summary, summary
        assert summary["reassigned"] > 0
        assert summary["unplaced"] == 0
        assert elapsed < 5, elapsed
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        after = conn.execute("SELECT COUNT(*) FROM glucose_logs").fetchone()[0]
        conn.close()
        assert after == before  # zero DB writes (D1/D4)

    def test_simulation_skips_dual_write(self, client, make_user, iso_solver):
        import src.features.glucose_tracker as gt
        user, tok = make_user(role="user")
        hdr = {"Authorization": f"Bearer {tok}"}
        pr = client.post("/v1/glucose", json={
            "user_id": user["id"], "value_mgdl": 130, "context": "fasting",
            "notes": "baseline notes",
        }, headers=hdr)
        assert pr.status_code == 200, pr.text
        r = client.post("/v1/triage",
                        json={"message": f"Mô phỏng {NAMES[0]} nghỉ 3 ngày cho BN tê chân",
                              "user_id": user["id"]},
                        headers=hdr)
        assert r.status_code == 200, r.text
        assert r.json()["solver_used"] == "ga"
        assert r.json().get("previsit_notes") is None
        notes = gt.get_logs(user["id"], limit=1)[0].get("notes") or ""
        assert notes == "baseline notes", notes


class TestEndpointCompat:
    def test_emergency_bypasses_routing(self, client, iso_solver):
        r = _post(client, "lơ mơ vã mồ hôi, đặt lịch Bạch Mai BHYT trừ T4")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["emergency"] is True
        assert "115" in (data["message"] or "")
        assert data["recommended_doctors"] == []
        assert data.get("solver_used") is None  # routing never ran

    def test_backward_compat_fields(self, client, iso_solver):
        r = _post(client, "Tê rần 10 đầu ngón chân, mắt mờ 2 hôm")
        assert r.status_code == 200, r.text
        data = r.json()
        for key in ("emergency", "urgency", "symptoms", "suggested_specialty",
                    "recommended_doctors", "previsit_notes", "followup_question"):
            assert key in data, key
        assert data["solver_used"] == "greedy"  # 0 constraints -> greedy
        assert data["recommended_doctors"]

    def test_no_key_deterministic(self, client, iso_solver):
        r1 = _post(client, TestDFS.MSG_EXACT)
        r2 = _post(client, TestDFS.MSG_EXACT)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()


class TestWithKeyPhantomDoctor:
    """F1: with-key LLM merge must not hallucinate requested_doctor (mocked, no API call)."""

    def test_llm_phantom_doctor_dropped(self, iso_solver, monkeypatch):
        import src.features.triage_nlu as nlu_mod
        # Mock the LLM call itself — never hits OpenAI.
        monkeypatch.setattr(nlu_mod, "_parse_llm", lambda text: {
            "requested_doctor": "Bac si la",  # phantom: not a roster ID/name
            "symptoms": [], "urgency": "routine",
            "specialty": "Endocrinology_General",
            "time_constraints": {"excluded_weekdays": []},
            "intents": ["booking"],
        })
        nlu = nlu_mod.parse_triage("muon gap bac si la")
        assert nlu["requested_doctor"] is None
        assert nlu["requested_doctors"] == []
        assert "doctor" not in nlu["constraints"]
        # No spurious "doctor" relaxation downstream.
        from src.features.solvers import DFSSolver, count_constraints
        assert "doctor" not in count_constraints(nlu)
        assert "doctor" not in DFSSolver.solve(nlu).get("relaxed", [])

    def test_llm_valid_doctor_kept_consistent(self, iso_solver, monkeypatch):
        import src.features.triage_nlu as nlu_mod
        monkeypatch.setattr(nlu_mod, "_parse_llm", lambda text: {
            "requested_doctor": NAMES[0],  # real roster name -> resolvable
            "symptoms": [], "urgency": "routine",
            "specialty": "Endocrinology_General",
            "time_constraints": {"excluded_weekdays": []},
            "intents": ["booking"],
        })
        nlu = nlu_mod.parse_triage("muon gap bac si")
        assert nlu["requested_doctor"] == "syn_doctor_01"
        assert nlu["requested_doctors"] == ["syn_doctor_01"]
        assert "doctor" in nlu["constraints"]
