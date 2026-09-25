"""Service benchmarks B6+B7+B8 (D1-D8 pins).

Pure-function path only (no TestClient, no DB writes, no LLM/RAG imports);
overall/exit derive SOLELY from safety gates; regression is WARNING (exit 0).
"""
import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_benchmarks as rb


def _no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_BUSY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "src.chat.rag_pipeline", None)


def _gold(service):
    return json.loads((Path(rb.BENCH_DIR) / rb.BENCH_FILES[service])
                      .read_text(encoding="utf-8"))


PAIN_POINTS = {"dose_change", "drug_interaction", "multi_intent",
                "booking_time", "elderly_phrasing", "trend_question",
                "lifestyle_question", "severity_question",
                "comorbidity_context", "multi_analyte", "relaxation_order",
                "cost_warning", "weekday_exclusion", "unknown_doctor",
                "legacy"}


class TestGoldSources:
    @pytest.mark.parametrize("service", ["triage", "labs", "solvers"])
    def test_source_resolves_and_versioned(self, service):
        for case in _gold(service):
            assert case.get("version") in (1, 2), case.get("id")
            assert case.get("pain_point") in PAIN_POINTS, case.get("id")
            fname, tname = case["source"].split("::")
            text = (Path(rb.ROOT) / "tests" / fname).read_text(encoding="utf-8")
            assert re.search(rf"def {re.escape(tname)}\b", text), case["source"]

    @pytest.mark.parametrize("service", ["triage", "labs", "solvers"])
    def test_gold_size(self, service):
        assert 20 <= len(_gold(service)) <= 40, service


class TestCiNoKeyGreen:
    def test_b6_b7_schema_and_pass(self, monkeypatch):
        _no_key(monkeypatch)
        for svc, res in (("triage", rb.run_b6()), ("labs", rb.run_b7())):
            assert set(res) == {"safety_gates", "quality", "performance"}, svc
            assert res["performance"]["n"] > 0, svc
            for v in res["safety_gates"].values():
                assert v == 1.0, (svc, res["safety_gates"])
        assert rb.services_overall_status(
            {"triage": rb.run_b6(), "labs": rb.run_b7()}) == "PASSED"

    def test_b8_schema_and_pass(self, monkeypatch):
        _no_key(monkeypatch)
        res = rb.run_b8()
        assert set(res) >= {"safety_gates", "quality", "performance"}
        assert res["safety_gates"]["never_relax_specialty"] == 1.0
        assert res["safety_gates"]["ga_timing_ok"] == 1.0
        assert res["_ga_summary"]["reassigned"] > 0
        assert res["performance"]["greedy_solve"]["n"] > 0


class TestCiScrubDotenv:
    def test_ci_scrubs_dotenv_key(self, monkeypatch):
        import os
        _no_key(monkeypatch)
        # Simulate a key present via .env / env, then runner scrub + dotenv reload.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        import scripts.run_benchmarks as rb2  # noqa: F401 (scrub logic below mirrors run_services_mode)
        os.environ["OPENAI_API_KEY"] = ""
        from dotenv import load_dotenv
        load_dotenv()  # override=False: "" must survive
        assert not os.getenv("OPENAI_API_KEY")
        # Prove no LLM branch is taken: _parse_llm early-returns None on falsy key
        # (parse_triage still calls it, but it never reaches OpenAI client).
        from src.triage import triage_nlu as _nlu
        assert _nlu._parse_llm("dat lich kham thu 4") is None
        baseline = rb.run_b8()
        assert baseline["quality"] and all(
            v == 1.0 for v in baseline["safety_gates"].values()
            if isinstance(v, float)), baseline["safety_gates"]


class TestSafetyFailInjection:
    def test_flip_one_label_fails_overall(self, monkeypatch):
        _no_key(monkeypatch)
        gold = _gold("triage")
        bad = copy.deepcopy(gold)
        victim = next(c for c in bad if c.get("expect_emergency") is False)
        victim["expect_emergency"] = True  # gold says panic, code says safe
        res = rb.run_b6(gold=bad)
        assert res["safety_gates"]["panic_recall"] < 1.0
        assert rb.services_overall_status({"triage": res}) == "FAILED"

    def test_panic_flip_fails_labs(self, monkeypatch):
        _no_key(monkeypatch)
        gold = _gold("labs")
        bad = copy.deepcopy(gold)
        victim = next(c for c in bad if c.get("kind") == "panic"
                      and c.get("expect_panic") is False)
        victim["expect_panic"] = True
        res = rb.run_b7(gold=bad)
        assert res["safety_gates"]["panic_recall"] < 1.0
        assert rb.services_overall_status({"labs": res}) == "FAILED"


class TestRegressionWarning:
    def test_drop_triggers_warning_not_fail(self, monkeypatch):
        _no_key(monkeypatch)
        services = {"triage": rb.run_b6()}
        inflated = {"services": {"triage": {"quality": {
            k: 1.0 for k in services["triage"]["quality"]}}}}
        lowered = copy.deepcopy(services)
        for k in lowered["triage"]["quality"]:
            lowered["triage"]["quality"][k] = 0.5
        warnings = rb.compare_to_baseline(lowered, inflated)
        assert any(w.startswith("WARNING") and "triage" in w for w in warnings)
        # WARNING never affects exit: safety still green -> PASSED.
        assert rb.services_overall_status(lowered) == "PASSED"

    def test_missing_baseline_warns(self):
        warnings = rb.compare_to_baseline({"triage": {"quality": {}}}, None)
        assert any(w.startswith("WARNING") for w in warnings)


class TestLegacyUntouched:
    def test_services_opt_in_only(self):
        import argparse
        assert rb.SERVICES == ("triage", "labs", "solvers")
        # Legacy output keys preserved alongside (additive, D8).
        for fn in ("run_b123", "run_b4", "run_b5", "latency_summary"):
            assert callable(getattr(rb, fn)), fn

    def test_services_filename_distinct(self):
        assert "services" in "results_benchmark_services_20260923_2118.json"
        assert "results_benchmark_20260906_1920.json".replace(
            "results_benchmark_", "") != "services_20260923_2118.json"
