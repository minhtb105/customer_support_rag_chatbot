"""Synthetic multi-patient/doctor seed: counts, contexts, archetypes, sidecars, roster.

Fast by design: dry-run/count tests are pure (no DB); DB tests use a small
window (days=7) on a tmp DB via monkeypatched GLUCOSE_DB_PATH + tmp sidecar
dir. Never calls OpenAI (env key removed, fake episodic injected).
"""
import json
import os
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_synthetic_data as seed  # noqa: E402


@pytest.fixture
def iso_env(monkeypatch, tmp_path):
    import src.diabetes.glucose_tracker as gt
    tmp_db = tmp_path / "syn.db"
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_db)
    monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(tmp_path / "meta"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gt.init_glucose_db()
    return tmp_path


@pytest.fixture
def fake_episodic():
    class Fake:
        def __init__(self):
            self.saved = []

        def add_message(self, role, content, metadata=None):
            self.saved.append((role, content, metadata or {}))
            return True

    return Fake()


class TestSyntheticSeed:
    def test_dry_run_counts(self):
        r = seed.seed_all(seed=42, days=90, end_offset=3, dry_run=True)
        assert r["patients"] == 30 and r["doctors"] == 5
        assert r["logs"] == 30 * 90 * 4 == 10800
        assert r["dry_run"] is True

    def test_context_literals_valid(self):
        name_fn = seed._name_fn(42)
        patients = seed.build_patients(42, name_fn)
        plan = seed.build_log_plan(patients, 42, days=3, end_offset=3)
        assert len(plan) == 30 * 3 * 4
        for _, _, ctx, _ in plan:
            assert ctx in seed.VALID_CONTEXTS

    def test_dawn_spike_rate_sanity(self):
        # retune 2026-09-22 (old band 0.25-0.55 kept): re-measured, band unchanged —
        # engine modulates build_log_plan only, gen_value dawn rule intact.
        rng = random.Random(42)
        fasting = [seed.gen_value("dawn_phenomenon", "fasting", rng) for _ in range(2000)]
        rate = sum(1 for v in fasting if v > 180) / len(fasting)
        assert 0.25 < rate < 0.55, rate
        well = [seed.gen_value("well_controlled", "fasting", rng) for _ in range(2000)]
        assert sum(1 for v in well if v > 250) / len(well) < 0.12

    def test_seed_reset_idempotent(self, iso_env, fake_episodic):
        kw = dict(seed=42, days=7, end_offset=3, reset=True, db_path=None,
                  sidecar_dir=None, use_llm=False)
        r1 = seed.seed_all(**kw)
        # inject fake episodic on 2nd run via run_b4 path is covered separately;
        # here just assert row-count idempotency
        r2 = seed.seed_all(**kw)
        assert r1["inserted"] == r2["inserted"] == 30 * 7 * 4
        assert seed.existing_synthetic_count() == 30 * 7 * 4

    def test_sidecar_shape_and_pcp(self, iso_env):
        # retune 2026-09-22: archetype 10/10/10 + PCP 6/doctor kept exact; staged
        # engine complications allowed as extras (asserted subset in distributions).
        seed.seed_all(seed=42, days=3, end_offset=3, reset=True, use_llm=False)
        meta = Path(os.environ["SYNTHETIC_SIDECAR_DIR"])
        mapping = json.loads((meta / "doctors_patients.json").read_text(encoding="utf-8"))
        roster = json.loads((meta / "synthetic_roster.json").read_text(encoding="utf-8"))
        assert len(mapping) == 30
        for i in range(1, 31):
            e = mapping[f"syn_patient_{i:03d}"]
            assert set(e) >= {"display_name", "age", "gender", "comorbidities", "pcp_doctor_id", "archetype"}
            assert e["display_name"] and e["pcp_doctor_id"].startswith("syn_doctor_")
        assert [k for k in mapping if mapping[k]["archetype"] == "well_controlled"].__len__() == 10
        assert len(roster["doctors"]) == 5
        specs = [d["specialty"] for d in roster["doctors"]]
        assert specs.count("Nội tiết chung") == 2 and specs.count("Dinh dưỡng & Lối sống") == 2
        assert all(d["working_hours"] for d in roster["doctors"])
        # PCP round-robin: 6 patients per doctor
        counts = {}
        for e in mapping.values():
            counts[e["pcp_doctor_id"]] = counts.get(e["pcp_doctor_id"], 0) + 1
        assert sorted(counts.values()) == [6, 6, 6, 6, 6]

    def test_deterministic_fallback_no_openai(self, iso_env, fake_episodic, monkeypatch):
        # retune 2026-09-22: B4 notes kept as >0 (exact-count pin dropped — engine
        # meds annotations + B4 both write notes now).
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setitem(sys.modules, "openai", None)  # any import attempt fails -> template path
        r = seed.seed_all(seed=7, days=14, end_offset=3, reset=True,
                          use_llm=True, episodic=None)
        assert r["b4"]["notes"] > 0  # dawn+high-risk spike weeks got notes
        # episodic via fake on explicit run_b4 (seed used real lazy import -> fail-open 0)
        import src.diabetes.glucose_tracker as gt  # noqa
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT notes FROM glucose_logs WHERE user_id LIKE 'syn_patient_%'"
                " AND notes IS NOT NULL").fetchall()
        finally:
            conn.close()
        # M1 cleanup 2026-09-22: every non-null note must come from a known
        # writer — B4 template ("bánh ngọt") or engine meds timeline ("Đang dùng").
        assert rows and all(
            "bánh ngọt" in (n or "") or "Đang dùng" in (n or "") for (n,) in rows
        )
        # determinism: same seed -> same first values
        name_fn = seed._name_fn(7)
        p1 = seed.build_log_plan(seed.build_patients(7, name_fn), 7, 14, 3)
        name_fn = seed._name_fn(7)
        p2 = seed.build_log_plan(seed.build_patients(7, name_fn), 7, 14, 3)
        assert p1 == p2

    def test_b4_episodic_fake(self, iso_env, fake_episodic):
        name_fn = seed._name_fn(42)
        patients = seed.build_patients(42, name_fn)
        plan = seed.build_log_plan(patients, 42, days=14, end_offset=3)
        seed.insert_logs_batch(plan)
        stats = seed.run_b4(patients, plan, use_llm=False, episodic=fake_episodic)
        assert stats["notes"] > 0 and stats["episodic"] == stats["notes"]
        assert any("bánh ngọt" in c for _, c, _ in fake_episodic.saved)

    def test_roster_helper(self, iso_env):
        from src.scheduling import synthetic_roster as sr
        seed.seed_all(seed=42, days=2, end_offset=3, reset=True, use_llm=False)
        assert sr.get_patient_display_name("syn_patient_001")
        assert sr.get_patient_pcp("syn_patient_001").startswith("syn_doctor_")
        assert len(sr.query_doctor_availability()) == 5
        assert len(sr.query_doctor_availability(day="Sat")) >= 1
        assert len(sr.query_doctor_availability(doctor_id="syn_doctor_03")) == 1

    def test_roster_fail_open(self, monkeypatch, tmp_path):
        from src.scheduling import synthetic_roster as sr
        monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(tmp_path / "empty"))
        assert sr.get_patient_display_name("syn_patient_001") is None
        assert sr.get_patient_pcp("syn_patient_001") is None
        assert sr.query_doctor_availability() == []
