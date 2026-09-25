"""Unit tests for Respiratory tracker — peak flow, GOLD, inhaler technique."""
import pytest
from src.vitals.respiratory_tracker import (
    classify_peak_flow, classify_gold, _classify_inhaler,
    add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory
)


class TestClassifyPeakFlow:
    def test_green(self):
        cls, msg = classify_peak_flow(85)
        assert cls == "green"
        assert "Green" in msg

    def test_yellow(self):
        cls, _ = classify_peak_flow(65)
        assert cls == "yellow"

    def test_red(self):
        cls, _ = classify_peak_flow(40)
        assert cls == "red"

    def test_none(self):
        cls, _ = classify_peak_flow(None)
        assert cls == "unknown"

    def test_boundary_80(self):
        cls, _ = classify_peak_flow(80)
        assert cls == "green"

    def test_boundary_50(self):
        cls, _ = classify_peak_flow(50)
        assert cls == "yellow"


class TestClassifyGold:
    def test_gold_stages(self):
        assert classify_gold(5) == "GOLD_1_mild"
        assert classify_gold(15) == "GOLD_2_moderate"
        assert classify_gold(25) == "GOLD_3_severe"
        assert classify_gold(35) == "GOLD_4_very_severe"

    def test_none(self):
        assert classify_gold(None) == "unknown"


class TestClassifyInhaler:
    def test_correct_bool(self):
        cls, _ = _classify_inhaler(True, None, None)
        assert cls == "correct"

    def test_incorrect_bool(self):
        cls, _ = _classify_inhaler(False, None, None)
        assert cls == "incorrect"

    def test_steps_90_percent(self):
        cls, _ = _classify_inhaler(None, 9, 10)
        assert cls == "correct"

    def test_steps_partial(self):
        cls, _ = _classify_inhaler(None, 7, 10)
        assert cls == "partial"

    def test_steps_incorrect(self):
        cls, _ = _classify_inhaler(None, 5, 10)
        assert cls == "incorrect"

    def test_no_data(self):
        cls, _ = _classify_inhaler(None, None, None)
        assert cls == "unknown"


class TestRespiratoryLog:
    def test_add_and_stats(self, temp_user):
        add_respiratory_log(temp_user, peak_flow_percent=85, cat_score=8, inhaler_correct=True)
        add_respiratory_log(temp_user, peak_flow_percent=60, cat_score=15, inhaler_correct=False)
        add_respiratory_log(temp_user, peak_flow_percent=40, cat_score=25, inhaler_correct=False)
        stats = get_respiratory_stats(temp_user)
        assert stats["total_logs"] == 3
        assert stats["avg_peak_flow"] == pytest.approx(61.7, rel=1e-2)
        # 60 + incorrect inhaler => red, 40 => red => 2 reds
        assert stats["classification_counts"]["red"] == 2

    def test_red_rate(self, temp_user):
        add_respiratory_log(temp_user, peak_flow_percent=85)
        add_respiratory_log(temp_user, peak_flow_percent=40)
        stats = get_respiratory_stats(temp_user)
        assert stats["red_rate"] == pytest.approx(0.5, rel=1e-2)

    def test_escalation_red(self, temp_user):
        add_respiratory_log(temp_user, peak_flow_percent=40)
        assert should_escalate_respiratory(temp_user) is True

    def test_no_escalation_green(self, temp_user):
        add_respiratory_log(temp_user, peak_flow_percent=85)
        assert should_escalate_respiratory(temp_user) is False

    def test_get_logs_limit(self, temp_user):
        for i in range(5):
            add_respiratory_log(temp_user, peak_flow_percent=80)
        logs = get_respiratory_logs(temp_user, limit=2)
        assert len(logs) == 2

    def test_overall_classification_worst(self, temp_user):
        # peak green but inhaler incorrect -> red
        rec = add_respiratory_log(temp_user, peak_flow_percent=85, inhaler_correct=False)
        assert rec["classification"] == "red"
        # peak yellow + inhaler partial -> yellow
        rec2 = add_respiratory_log(temp_user, peak_flow_percent=65, inhaler_steps_correct=7, inhaler_steps_total=10)
        assert rec2["classification"] == "yellow"
