"""Disease engine subset (D1) + diabetes_vn.json shape (D5) + drift budget (D3).

Pure tests (no DB, no OpenAI). Seeded RNG pinned.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import disease_engine as eng  # noqa: E402

MODULE = Path(__file__).resolve().parents[1] / "src" / "features" / "modules" / "diabetes_vn.json"


def _mini():
    return {
        "initial": "a",
        "states": {"a": {"kind": "Initial"}, "b": {"kind": "SetAttribute", "attribute": "x", "value": 1},
                   "z": {"kind": "Terminal"}},
        "transitions": [{"from": "a", "kind": "direct", "to": "b"},
                        {"from": "b", "kind": "direct", "to": "z"}],
    }


class TestEngineSubset:
    def test_unknown_state_kind_fails(self):
        mod = _mini()
        mod["states"]["evil"] = {"kind": "Delay"}
        mod["transitions"].append({"from": "b", "kind": "direct", "to": "evil"})
        with pytest.raises(ValueError, match="unknown state kind"):
            eng.validate_module(mod)

    def test_unknown_transition_kind_fails(self):
        mod = _mini()
        mod["transitions"].append({"from": "a", "kind": "table-lookup", "to": "b"})
        with pytest.raises(ValueError, match="unknown transition kind"):
            eng.validate_module(mod)

    def test_bare_probability_fails(self):
        mod = {"initial": "a",
               "states": {"a": {"kind": "Initial"}, "b": {"kind": "Symptom", "symptom": "x"}},
               "transitions": [{"from": "a", "kind": "distributed",
                                 "targets": [{"to": "b", "p": 1.0}]}]}
        with pytest.raises(ValueError, match="bare probability"):
            eng.validate_module(mod)

    def test_weights_must_sum_to_one(self):
        mod = {"initial": "a",
               "states": {"a": {"kind": "Initial"}, "b": {"kind": "Symptom", "symptom": "x"}},
               "transitions": [{"from": "a", "kind": "distributed",
                                 "targets": [{"to": "b", "p": 0.5,
                                              "remarks": {"assumption": "t"}}]}]}
        with pytest.raises(ValueError, match="sum to"):
            eng.validate_module(mod)

    def test_terminal_halts_and_determinism(self):
        e1 = eng.DiseaseEngine(_mini(), seed=1)
        e2 = eng.DiseaseEngine(_mini(), seed=1)
        assert e1.run({}, weeks=10) == e2.run({}, weeks=10)
        assert e1.run({}, weeks=10)["trajectory"][-1]["kind"] == "Terminal"

    def test_conditional_guard_routing(self):
        mod = {"initial": "a",
               "states": {"a": {"kind": "Initial"}, "old": {"kind": "Symptom", "symptom": "o"},
                          "young": {"kind": "Symptom", "symptom": "y"}},
               "transitions": [{"from": "a", "kind": "conditional",
                                 "branches": [{"when": {"attribute": "age", "op": ">=", "value": 60},
                                               "to": "old"}],
                                 "default": "young"}]}
        e = eng.DiseaseEngine(mod, seed=0)
        assert e.run({"age": 70}, weeks=1)["trajectory"][0]["state"] == "old"
        assert e.run({"age": 40}, weeks=1)["trajectory"][0]["state"] == "young"


class TestModuleShape:
    def test_module_loads_and_validates(self):
        eng.load_module(MODULE)  # raises on unknown kinds

    def test_every_probability_has_remarks(self):
        mod = json.loads(MODULE.read_text(encoding="utf-8"))
        n = 0
        for tr in mod["transitions"]:
            for t in tr.get("targets") or []:
                n += 1
                assert "source" in (t.get("remarks") or {}) or "assumption" in (t.get("remarks") or {})
        assert n > 0

    def test_drift_budget_within_two(self):
        mod = json.loads(MODULE.read_text(encoding="utf-8"))
        for name, spec in mod["states"].items():
            if spec.get("kind") == "SetAttribute" and "sample" in spec:
                assert abs(spec["sample"]["min"]) <= 2.0 and abs(spec["sample"]["max"]) <= 2.0, name

    def test_weekly_run_per_archetype(self):
        mod = eng.load_module(MODULE)
        for arch in ("well_controlled", "dawn_phenomenon", "high_risk_comorbid"):
            e = eng.DiseaseEngine(mod, seed=42)
            out = e.run({"archetype": arch, "age": 55}, weeks=12)
            assert len(out["trajectory"]) == 12
            kinds = {r["kind"] for r in out["trajectory"]}
            assert "Observation" in kinds

    def test_pure_drift_never_trends(self):
        # D3: 4-week pure-drift fasting series (max +-2/week on base 100) -> none.
        from src.features.anomaly_detector import detect_trend
        logs = []
        v = 100.0
        for week in range(4):
            for day in range(7):
                logs.append({"value_mgdl": v, "context": "fasting",
                             "measured_at": f"2026-08-{3 + week * 7 + day:02d}T07:00:00"})
            v += 2.0
        assert detect_trend(logs)["type"] == "none"

    def test_ramp_triple_triggers(self):
        # Recorder math 100 -> 114 -> 126 must trip the cascade detector.
        from src.features.anomaly_detector import detect_trend
        logs = [{"value_mgdl": v, "context": "fasting",
                 "measured_at": f"2026-08-{10 + i:02d}T07:00:00"} for i, v in enumerate([100, 114, 126])]
        assert detect_trend(logs)["type"] == "trend_cascade"
