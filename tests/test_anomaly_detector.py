"""Unit tests for Chronic Time-Series anomaly middleware (pure, no LLM)."""

from datetime import datetime, timedelta

from src.diabetes.anomaly_detector import (
    analyze_glucose_log,
    detect_spike,
    detect_trend,
)


def _fasting_day(base: datetime, offset_days: int, value: float, idx: int = 0):
    dt = (base + timedelta(days=offset_days)).replace(hour=7, minute=0, second=idx)
    return {
        "id": offset_days * 10 + idx,
        "value_mgdl": value,
        "measured_at": dt.isoformat(),
        "context": "fasting",
    }


class TestDetectSpike:
    def test_spike_high_over_250(self):
        r = detect_spike(260)
        assert r["type"] == "spike_high"

    def test_spike_low_under_70(self):
        r = detect_spike(65)
        assert r["type"] == "spike_low"

    def test_no_spike_boundary(self):
        assert detect_spike(250)["type"] == "none"
        assert detect_spike(70)["type"] == "none"
        assert detect_spike(120)["type"] == "none"


class TestDetectTrend:
    def test_trend_cascade_10_15_pct_3_days(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 100),
            _fasting_day(base, 1, 112),  # +12%
            _fasting_day(base, 2, 125),  # +11.6%
        ]
        r = detect_trend(logs)
        assert r["type"] == "trend_cascade"

    def test_no_false_positive_increase_under_10pct(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 100),
            _fasting_day(base, 1, 105),  # +5%
            _fasting_day(base, 2, 110),  # +4.7%
        ]
        assert detect_trend(logs)["type"] == "none"

    def test_no_false_positive_missing_day(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 100),
            # day 1 missing -> chain broken
            _fasting_day(base, 2, 112),
            _fasting_day(base, 3, 125),
        ]
        assert detect_trend(logs)["type"] == "none"

    def test_no_false_positive_non_fasting_ignored(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 100),
            {"id": 99, "value_mgdl": 250, "measured_at": (base + timedelta(days=1)).isoformat(), "context": "random"},
            _fasting_day(base, 2, 125),
        ]
        assert detect_trend(logs)["type"] == "none"

    def test_group_by_day_uses_daily_avg(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 90, idx=1),
            _fasting_day(base, 0, 110, idx=2),  # avg day0 = 100
            _fasting_day(base, 1, 100, idx=1),
            _fasting_day(base, 1, 124, idx=2),  # avg day1 = 112 (+12%)
            _fasting_day(base, 2, 120, idx=1),
            _fasting_day(base, 2, 130, idx=2),  # avg day2 = 125 (+11.6%)
        ]
        assert detect_trend(logs)["type"] == "trend_cascade"

    def test_zero_guard_no_crash(self):
        base = datetime(2026, 9, 1)
        logs = [
            _fasting_day(base, 0, 0),
            _fasting_day(base, 1, 112),
            _fasting_day(base, 2, 125),
        ]
        assert detect_trend(logs)["type"] == "none"


class TestAnalyze:
    def test_spike_wins_over_trend(self):
        base = datetime(2026, 9, 1)
        recent = [
            _fasting_day(base, 0, 100),
            _fasting_day(base, 1, 112),
            _fasting_day(base, 2, 125),
        ]
        r = analyze_glucose_log(300, recent)
        assert r["anomaly"]["type"] == "spike"
        assert len(r["follow_up_questions"]) > 0

    def test_trend_returns_fqg(self):
        base = datetime(2026, 9, 1)
        recent = [
            _fasting_day(base, 0, 100),
            _fasting_day(base, 1, 112),
            _fasting_day(base, 2, 125),
        ]
        r = analyze_glucose_log(130, recent)
        assert r["anomaly"]["type"] == "trend"
        assert len(r["follow_up_questions"]) > 0

    def test_none_no_fqg(self):
        r = analyze_glucose_log(110, [])
        assert r["anomaly"]["type"] == "none"
        assert r["follow_up_questions"] == []
