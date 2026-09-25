"""Lab seed B1: counts, idempotency, ref-range sanity, isolation.

Fast: tmp DB via module-namespace monkeypatch (src.labs.store.LAB_DB_PATH),
small patient lists. Never touches glucose/episodic/B4. No OpenAI (delenv).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.seed_lab_data as seed  # noqa: E402


@pytest.fixture
def iso_lab(monkeypatch, tmp_path):
    import src.labs.store as ls
    tmp_db = tmp_path / "labs.db"
    monkeypatch.setattr(ls, "LAB_DB_PATH", tmp_db)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


PIDS = ["syn_patient_001", "syn_patient_002", "syn_patient_003"]


class TestLabSeed:
    def test_dry_run_counts_no_write(self, iso_lab):
        r = seed.seed_all(seed=42, dry_run=True, patient_ids=PIDS)
        assert r["patients"] == 3
        assert r["dry_run"] is True
        # 2-4 reports/patient, 10 obs/report
        assert 6 <= r["reports"] <= 12
        assert r["observations"] == r["reports"] * 10
        import src.labs.store as ls
        assert ls.count_reports() == 0

    def test_seed_reset_idempotent(self, iso_lab):
        kw = dict(seed=42, reset=True, patient_ids=PIDS)
        r1 = seed.seed_all(**kw)
        r2 = seed.seed_all(**kw)
        assert r1["inserted_reports"] == r2["inserted_reports"]
        assert r1["inserted_observations"] == r2["inserted_observations"]
        import src.labs.store as ls
        assert ls.count_reports() == r2["inserted_reports"]
        assert ls.count_observations() == r2["inserted_observations"]

    def test_ref_range_sanity(self, iso_lab):
        import src.labs.store as ls
        import src.labs.lab_thresholds as th
        seed.seed_all(seed=42, reset=True, patient_ids=PIDS)
        reps = ls.get_reports("syn_patient_001")
        assert reps, "expected reports for patient 001"
        for rep in reps:
            for ob in ls.get_observations(rep["report_id"]):
                spec = th.PANEL[th.LOINC_TO_KEY[ob["loinc"]]]
                assert ob["ref_low"] == spec["ref_low"]
                assert ob["ref_high"] == spec["ref_high"]
                assert ob["flag"] == th.flag_value(ob["loinc"], ob["value"])
        # panic codes must all be in-panel LOINCs (no K+/Na+ outside panel)
        panics = [k for k, v in th.PANEL.items()
                  if v["panic_low"] is not None or v["panic_high"] is not None]
        assert panics and set(panics) <= set(th.PANEL)
        assert th.is_panic("4548-4", 13.0)  # HbA1c in-panel panic
        assert th.is_panic("33914-3", 10.0)  # eGFR in-panel panic
        assert not th.is_panic("4548-4", 6.0)

    def test_no_key_deterministic(self, iso_lab, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setitem(sys.modules, "openai", None)
        p1 = seed.build_lab_plan(PIDS, 42)
        p2 = seed.build_lab_plan(PIDS, 42)
        assert p1 == p2

    def test_history_ordered(self, iso_lab):
        import src.labs.store as ls
        seed.seed_all(seed=42, reset=True, patient_ids=PIDS)
        hist = ls.get_history("syn_patient_001", "4548-4")
        assert len(hist) >= 2
        times = [h["sampled_at"] for h in hist]
        assert times == sorted(times)
