"""Unit tests for Track A — glucose_tracker (diabetes spec)."""

import pytest
from src.features.glucose_tracker import (
    classify_glucose,
    should_escalate_to_doctor,
    add_log,
    get_stats,
    get_logs,
)
from src.config import GLUCOSE_THRESHOLDS_MGDL


class TestClassifyGlucose:
    def test_fasting_normal(self):
        cls, _ = classify_glucose(85, "fasting")
        assert cls == "normal"

    def test_fasting_prediabetes(self):
        cls, _ = classify_glucose(110, "fasting")
        assert cls == "elevated"

    def test_fasting_diabetes(self):
        cls, _ = classify_glucose(130, "fasting")
        assert cls == "high"

    def test_fasting_boundary_100(self):
        # 100 is upper bound of normal per thresholds
        cls, _ = classify_glucose(99, "fasting")
        assert cls == "normal"
        cls2, _ = classify_glucose(100, "fasting")
        assert cls2 == "elevated"

    def test_postprandial_normal(self):
        cls, _ = classify_glucose(120, "post_meal_2h")
        assert cls == "normal"

    def test_postprandial_diabetes(self):
        cls, _ = classify_glucose(210, "post_meal_2h")
        assert cls == "high"

    def test_hypoglycemia(self):
        cls, msg = classify_glucose(65, "random")
        assert cls == "low"
        assert "Hạ" in msg or "hypoglycemia" in msg.lower()

    def test_critical_high(self):
        cls, _ = classify_glucose(350, "random")
        assert cls == "critical"

    def test_random_thresholds(self):
        assert classify_glucose(139, "random")[0] == "normal"
        assert classify_glucose(150, "random")[0] == "elevated"
        assert classify_glucose(200, "random")[0] == "high"


class TestGlucoseStatsAndEscalation:
    def test_add_and_stats(self, temp_user):
        add_log(temp_user, 95, context="fasting")
        add_log(temp_user, 110, context="fasting")
        add_log(temp_user, 210, context="post_meal_2h")
        stats = get_stats(temp_user)
        assert stats["total_logs"] == 3
        assert stats["avg_mgdl"] == pytest.approx(138.3, rel=1e-2)
        assert stats["classification_counts"]["high"] == 1

    def test_logs_per_week_kpi(self, temp_user):
        # 1 log → <3/week should be flagged in stats
        add_log(temp_user, 100, context="fasting")
        stats = get_stats(temp_user)
        assert stats["logs_per_week"] < 3

    def test_escalation_critical(self, temp_user):
        add_log(temp_user, 350, context="random")
        assert should_escalate_to_doctor(temp_user) is True

    def test_escalation_three_high(self, temp_user):
        for _ in range(3):
            add_log(temp_user, 180, context="fasting")
        assert should_escalate_to_doctor(temp_user) is True

    def test_no_escalation_when_normal(self, temp_user):
        add_log(temp_user, 90, context="fasting")
        add_log(temp_user, 95, context="fasting")
        assert should_escalate_to_doctor(temp_user) is False

    def test_get_logs_limit(self, temp_user):
        for i in range(5):
            add_log(temp_user, 90 + i, context="random")
        logs = get_logs(temp_user, limit=2)
        assert len(logs) == 2
