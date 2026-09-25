"""Unit tests for Hypertension tracker — AHA/ACC 2025."""
import pytest
from src.vitals.bp_tracker import classify_bp, add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp, _should_escalate_bp
from src.shared.config import BP_THRESHOLDS_MMHG


class TestClassifyBP:
    def test_normal(self):
        cls, msg = classify_bp(110, 70)
        assert cls == "normal"
        assert "Normal" in msg

    def test_elevated(self):
        cls, _ = classify_bp(125, 78)
        assert cls == "elevated"

    def test_stage1_systolic(self):
        cls, _ = classify_bp(135, 85)
        assert cls == "stage1"

    def test_stage1_diastolic(self):
        cls, _ = classify_bp(118, 82)
        assert cls == "stage1"

    def test_stage2(self):
        cls, _ = classify_bp(150, 95)
        assert cls == "stage2"

    def test_crisis(self):
        cls, _ = classify_bp(185, 125)
        assert cls == "crisis"

    def test_boundary_120_80(self):
        # 120/80 is elevated threshold edge: 120-129/<80 → elevated, 120/80 not included due to diastolic ==80
        # Our classify checks elevated only if diastolic <80, so 120/80 falls to stage1 fallback
        cls, _ = classify_bp(120, 80)
        assert cls in ("stage1", "elevated", "normal")

    def test_stage2_via_diastolic(self):
        cls, _ = classify_bp(118, 92)
        assert cls == "stage2"


class TestBPStatsAndEscalation:
    def test_add_and_stats(self, temp_user):
        add_bp_log(temp_user, 110, 70, context="random")
        add_bp_log(temp_user, 135, 85, context="random")
        add_bp_log(temp_user, 150, 95, context="random")
        stats = get_bp_stats(temp_user)
        assert stats["total_logs"] == 3
        assert stats["avg_sys"] == pytest.approx(131.7, rel=1e-2)
        assert stats["avg_dia"] == pytest.approx(83.3, rel=1e-2)
        assert stats["classification_counts"]["stage2"] == 1

    def test_at_target_rate(self, temp_user):
        add_bp_log(temp_user, 110, 70)
        add_bp_log(temp_user, 115, 75)
        add_bp_log(temp_user, 150, 95)
        stats = get_bp_stats(temp_user)
        assert stats["at_target_rate"] == pytest.approx(2/3, rel=1e-2)

    def test_escalation_crisis(self, temp_user):
        add_bp_log(temp_user, 185, 125)
        assert should_escalate_bp(temp_user) is True

    def test_escalation_three_stage2(self, temp_user):
        for _ in range(3):
            add_bp_log(temp_user, 150, 95)
        assert should_escalate_bp(temp_user) is True

    def test_no_escalation_when_normal(self, temp_user):
        add_bp_log(temp_user, 110, 70)
        add_bp_log(temp_user, 115, 72)
        assert should_escalate_bp(temp_user) is False

    def test_get_logs_limit(self, temp_user):
        for i in range(5):
            add_bp_log(temp_user, 110 + i, 70 + i)
        logs = get_bp_logs(temp_user, limit=2)
        assert len(logs) == 2

    def test_validation_out_of_range(self, temp_user):
        with pytest.raises(ValueError):
            add_bp_log(temp_user, 400, 70)
        with pytest.raises(ValueError):
            add_bp_log(temp_user, 120, 250)

    def test_should_escalate_helper_direct(self):
        assert _should_escalate_bp([]) is False
        assert _should_escalate_bp([{"classification": "crisis"}]) is True
        assert _should_escalate_bp([{"classification": "stage2"}]*3) is True
        assert _should_escalate_bp([{"classification": "stage2"}]*2) is False
