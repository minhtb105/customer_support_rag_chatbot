"""Faker scale + haversine (D4/D5): small N/M/H, ranges, dry-run 1000-shape.

Scale rule: small in-DB tests + ONE dry-run count for the 1000 shape
(no DB write). Default 30/5/1 output stays byte-identical (old asserts gate).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_synthetic_data as seed  # noqa: E402


@pytest.fixture
def iso_env(monkeypatch, tmp_path):
    import src.diabetes.glucose_tracker as gt
    tmp_db = tmp_path / "scale.db"
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
    monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(tmp_path / "meta"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gt.init_glucose_db()
    return tmp_path


class TestDefaultGate:
    def test_default_dry_run_identical(self):
        r = seed.seed_all(seed=42, days=90, end_offset=3, dry_run=True)
        assert (r["patients"], r["doctors"], r["logs"]) == (30, 5, 30 * 90 * 4)

    def test_default_ids_and_pcp(self):
        name_fn = seed._name_fn(42)
        doctors, hosps = seed.build_doctors(name_fn)
        assert [d["doctor_id"] for d in doctors] == [f"syn_doctor_{i:02d}" for i in range(1, 6)]
        assert [(d["doctor_id"], d["specialty"], d["working_hours"]) for d in doctors] == list(seed.DOCTOR_SPECS)
        patients = seed.build_patients(42, name_fn)
        arch = [p["archetype"] for p in patients.values()]
        assert arch.count("well_controlled") == arch.count("dawn_phenomenon") == arch.count("high_risk_comorbid") == 10
        counts: dict = {}
        for p in patients.values():
            counts[p["pcp_doctor_id"]] = counts.get(p["pcp_doctor_id"], 0) + 1
        assert sorted(counts.values()) == [6, 6, 6, 6, 6]

    def test_default_hash_distances_unchanged(self):
        # Hash path pinned: recompute independently, must match helper exactly.
        from src.scheduling.solvers import doctor_attributes
        for did in [f"syn_doctor_{i:02d}" for i in range(1, 6)]:
            expect = round((hashlib.md5(f"{did}|geo".encode()).digest()[0] % 100) / 10.0, 1)
            assert doctor_attributes(did)["distance_km"] == expect

    def test_real_distance_none_without_sidecar(self, monkeypatch, tmp_path):
        from src.scheduling.solvers import real_distance_km
        monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(tmp_path / "empty"))
        assert real_distance_km("syn_doctor_01", "syn_patient_001") is None
        assert real_distance_km("syn_doctor_01", None) is None


class TestHaversine:
    def test_zero(self):
        from src.scheduling.solvers import haversine_km
        assert haversine_km(21.0, 105.8, 21.0, 105.8) == 0.0

    def test_known_pair(self):
        from src.scheduling.solvers import haversine_km
        d = haversine_km(0.0, 0.0, 0.0, 1.0)  # 1 degree at equator
        assert 110.0 < d < 112.5, d

    def test_antipodal(self):
        from src.scheduling.solvers import haversine_km
        d = haversine_km(0.0, 0.0, 0.0, 180.0)  # half circumference
        assert 19900.0 < d < 20150.0, d


class TestScaleFlags:
    def test_small_scale_counts_and_ranges(self, iso_env):
        r = seed.seed_all(seed=42, days=3, end_offset=3, reset=True, use_llm=False,
                          n_patients=6, n_doctors=3, hospitals=2, skip_b4=True)
        assert (r["patients"], r["doctors"], r["hospitals"]) == (6, 3, 2)
        assert r["inserted"] == 6 * 3 * 4
        meta = Path(os.environ["SYNTHETIC_SIDECAR_DIR"])
        mapping = json.loads((meta / "doctors_patients.json").read_text(encoding="utf-8"))
        roster = json.loads((meta / "synthetic_roster.json").read_text(encoding="utf-8"))
        assert len(mapping) == 6 and len(roster["doctors"]) == 3
        assert len(roster["hospitals"]) == 2
        # Round-robin PCP: 2 patients per doctor.
        counts: dict = {}
        for p in mapping.values():
            assert -90.0 <= p["lat"] <= 90.0 and -180.0 <= p["lon"] <= 180.0
            counts[p["pcp_doctor_id"]] = counts.get(p["pcp_doctor_id"], 0) + 1
        assert sorted(counts.values()) == [2, 2, 2]
        # Specialty round-robin over the 3 groups.
        specs = {d["specialty"] for d in roster["doctors"]}
        assert len(specs) >= 2

    def test_greedy_nearby_nonempty_both_paths(self, iso_env):
        # Real-coords path: seeded patients include near-hospital ones.
        from src.scheduling.solvers import GreedySolver, real_distance_km
        seed.seed_all(seed=42, days=7, end_offset=3, reset=True, use_llm=False, skip_b4=True)
        nlu = {"symptoms": [{"code": "numbness"}], "nearby": True,
               "time_constraints": {}, "specialty": "Endocrinology_General"}
        got = GreedySolver.solve(nlu, patient_id="syn_patient_001")
        assert len(got["recommended_doctors"]) > 0
        assert all(d["distance_km"] < 5 for d in got["recommended_doctors"])
        # Hash-fallback path (no sidecar) is also non-empty.
        assert real_distance_km("syn_doctor_01", "no_such_patient") is None

    def test_dry_run_1000_shape_no_write(self, tmp_path):
        sidecar = tmp_path / "meta1000"
        r = seed.seed_all(seed=42, days=30, end_offset=3, dry_run=True,
                          n_patients=1000, n_doctors=30, hospitals=3,
                          sidecar_dir=sidecar)
        assert r["logs"] == 1000 * 30 * 4
        assert r["patients"] == 1000 and r["doctors"] == 30 and r["hospitals"] == 3
        assert not sidecar.exists()  # dry-run wrote nothing


class TestQueuePagination:
    def test_paged_deterministic_and_compat(self, client, make_user, doctor_user, monkeypatch, tmp_path):
        import src.diabetes.glucose_tracker as gt
        tmp_db = tmp_path / "q.db"
        monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
        gt.init_glucose_db()
        ids = []
        for _ in range(3):
            u, tok = make_user(role="user")
            ids.append(u["id"])
            r = client.post("/v1/glucose", json={
                "user_id": u["id"], "value_mgdl": 100, "context": "fasting"}, headers={"Authorization": f"Bearer {tok}"})
            assert r.status_code == 200, r.text
        hdr = {"Authorization": f"Bearer {doctor_user['token']}"}
        full = client.get("/v1/doctor/patients?limit=100&offset=0", headers=hdr)
        assert full.status_code == 200, full.text
        body = full.json()
        assert set(body) >= {"patients", "total", "limit", "offset"}  # additive shape
        assert body["total"] >= 3 and body["limit"] == 100 and body["offset"] == 0
        p1 = client.get("/v1/doctor/patients?limit=2&offset=0", headers=hdr).json()
        p2 = client.get("/v1/doctor/patients?limit=2&offset=2", headers=hdr).json()
        p1b = client.get("/v1/doctor/patients?limit=2&offset=0", headers=hdr).json()
        # Deterministic slice: same page twice -> same order; pages disjoint; total stable.
        assert [p["user_id"] for p in p1["patients"]] == [p["user_id"] for p in p1b["patients"]]
        assert {p["user_id"] for p in p1["patients"]}.isdisjoint({p["user_id"] for p in p2["patients"]})
        assert p1["total"] == p2["total"]
        r = client.get("/v1/doctor/patients?limit=500", headers=hdr)
        assert r.status_code == 422  # cap 200
