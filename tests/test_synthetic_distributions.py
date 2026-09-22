"""Synthea-method distribution asserts (D2/D3/D6).

D6 split: N=30 asserts ONLY counts + double-run determinism + VALID_CONTEXTS.
Prevalence bands ONLY on dry-run plans or N>=300 runs, wide +-15pp, seeded pin.
No POST loops (pure helpers only). Seeded RNG pinned (seed=42).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_synthetic_data as seed  # noqa: E402

SEED = 42
STAGED_KNOWN = {"Bệnh thần kinh ngoại biên giai đoạn sớm", "Bệnh võng mạc giai đoạn sớm",
                "Bệnh thận giai đoạn sớm", "Bệnh thần kinh ngoại biên"}


@pytest.fixture
def iso_env(monkeypatch, tmp_path):
    import src.features.glucose_tracker as gt
    monkeypatch.setattr(gt, "GLUCOSE_DB_PATH", tmp_path / "dist.db")
    monkeypatch.setenv("SYNTHETIC_SIDECAR_DIR", str(tmp_path / "meta"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gt.init_glucose_db()
    return tmp_path


class TestN30Gate:
    def test_counts_and_contexts(self):
        r = seed.seed_all(seed=SEED, days=90, end_offset=3, dry_run=True)
        assert (r["patients"], r["logs"]) == (30, 30 * 90 * 4)

    def test_double_run_determinism(self):
        name_fn = seed._name_fn(SEED)
        p1 = seed.build_log_plan(seed.build_patients(SEED, name_fn), SEED, 90, 3)
        name_fn = seed._name_fn(SEED)
        p2 = seed.build_log_plan(seed.build_patients(SEED, name_fn), SEED, 90, 3)
        assert p1 == p2

    def test_all_contexts_valid(self):
        name_fn = seed._name_fn(SEED)
        plan = seed.build_log_plan(seed.build_patients(SEED, name_fn), SEED, 90, 3)
        assert all(ctx in seed.VALID_CONTEXTS for _, _, ctx, _ in plan)


class TestTrajectories:
    def test_weekly_drift_bounded(self):
        name_fn = seed._name_fn(SEED)
        patients = seed.build_patients(SEED, name_fn)
        traj = seed.build_trajectories(patients, SEED, 90)
        assert len(traj) == 30
        for pid, t in traj.items():
            cum = t["weekly_cum"]
            prev = 0.0
            for c in cum:
                assert abs(c - prev) <= 2.0 + 1e-9, pid
                prev = c

    def test_ramp_week_per_flagged_patient(self):
        name_fn = seed._name_fn(SEED)
        patients = seed.build_patients(SEED, name_fn)
        plan = seed.build_log_plan(patients, SEED, 90, 3)
        fasting: dict = {}
        for pid, _ts, ctx, v in plan:
            if ctx == "fasting":
                fasting.setdefault(pid, []).append(v)
        n_flagged = 0
        for pid, meta in patients.items():
            if meta["archetype"] not in ("dawn_phenomenon", "high_risk_comorbid"):
                continue
            n_flagged += 1
            vs = fasting[pid]
            assert any(a == 100.0 and b == 114.0 and c == 126.0
                       for a, b, c in zip(vs, vs[1:], vs[2:])), pid
        assert n_flagged == 20


class TestLargeScaleBands:
    def test_archetype_rates_within_15pp(self):
        # N=300 dry-run plan (pure, no DB): wide bands absorb engine drift.
        name_fn = seed._name_fn(SEED)
        patients = seed.build_patients(SEED, name_fn, n_patients=300)
        plan = seed.build_log_plan(patients, SEED, 30, 3)
        by_arch: dict = {}
        for pid, _ts, ctx, v in plan:
            if ctx == "fasting":
                by_arch.setdefault(patients[pid]["archetype"], []).append(v)
        dawn = by_arch["dawn_phenomenon"]
        assert abs(sum(1 for v in dawn if v > 180) / len(dawn) - 0.40) <= 0.15
        well_vals = by_arch["well_controlled"]
        assert sum(1 for v in well_vals if v > 250) / len(well_vals) <= 0.05 + 0.15
        high = by_arch["high_risk_comorbid"]
        spike = sum(1 for v in high if v > 250 or v < 70) / len(high)
        assert abs(spike - 0.30) <= 0.15


class TestSidecarStagedAndMeds:
    def test_staged_complications_and_meds_notes(self, iso_env):
        r = seed.seed_all(seed=SEED, days=14, end_offset=3, reset=True, use_llm=False,
                          skip_b4=True)
        assert r["inserted"] == 30 * 14 * 4
        meta = Path(os.environ["SYNTHETIC_SIDECAR_DIR"])
        mapping = json.loads((meta / "doctors_patients.json").read_text(encoding="utf-8"))
        arch = [m["archetype"] for m in mapping.values()]
        assert all(arch.count(a) == 10 for a in ("well_controlled", "dawn_phenomenon", "high_risk_comorbid"))
        for m in mapping.values():
            base_extra = set(m["comorbidities"]) - {"Thừa cân", "Rối loạn giấc ngủ",
                                                    "Tăng huyết áp", "Rối loạn lipid máu"}
            assert base_extra <= STAGED_KNOWN, base_extra
        import src.features.glucose_tracker as gt
        import sqlite3
        conn = sqlite3.connect(str(gt.GLUCOSE_DB_PATH))
        try:
            rows = conn.execute("SELECT notes FROM glucose_logs WHERE user_id LIKE 'syn_patient_%'"
                                " AND notes LIKE '%Đang dùng%'").fetchall()
        finally:
            conn.close()
        assert rows, "engine meds timeline must reach log notes"
