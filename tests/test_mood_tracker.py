"""Unit tests for Mental health tracker — PHQ-9/GAD-7 + PII + crisis."""
import pytest
from src.vitals.mood_tracker import (
    classify_phq9, classify_gad7, contains_crisis_keywords, classify_mood,
    add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood, redact_pii
)


class TestClassifyPHQ9:
    def test_minimal(self):
        assert classify_phq9(2)[0] == "minimal"

    def test_mild(self):
        assert classify_phq9(7)[0] == "mild"

    def test_moderate(self):
        assert classify_phq9(12)[0] == "moderate"

    def test_moderately_severe(self):
        assert classify_phq9(17)[0] == "moderately_severe"

    def test_severe(self):
        assert classify_phq9(22)[0] == "severe"

    def test_none(self):
        assert classify_phq9(None)[0] == "unknown"

    def test_boundaries(self):
        assert classify_phq9(4)[0] == "minimal"
        assert classify_phq9(5)[0] == "mild"
        assert classify_phq9(9)[0] == "mild"
        assert classify_phq9(10)[0] == "moderate"


class TestClassifyGAD7:
    def test_minimal(self):
        assert classify_gad7(3)[0] == "minimal"

    def test_mild(self):
        assert classify_gad7(7)[0] == "mild"

    def test_moderate(self):
        assert classify_gad7(12)[0] == "moderate"

    def test_severe(self):
        assert classify_gad7(18)[0] == "severe"

    def test_none(self):
        assert classify_gad7(None)[0] == "unknown"


class TestCrisis:
    def test_detect_english(self):
        flag, kws = contains_crisis_keywords("I want to kill myself")
        assert flag is True
        assert len(kws) >= 1

    def test_detect_vietnamese(self):
        flag, kws = contains_crisis_keywords("tôi muốn tự tử")
        assert flag is True

    def test_no_crisis(self):
        flag, kws = contains_crisis_keywords("I feel a bit down today")
        assert flag is False
        assert kws == []

    def test_classify_mood_crisis_overrides(self):
        cls, msg, is_crisis, kws = classify_mood(5, 5, "I want to die")
        assert cls == "crisis"
        assert is_crisis is True
        assert "1800-1567" in msg or "hotline" in msg.lower()


class TestRedactPII:
    def test_phone(self):
        assert "[REDACTED_PHONE]" in redact_pii("call me 0912345678 please")

    def test_email(self):
        assert "[REDACTED_EMAIL]" in redact_pii("my email test@example.com here")

    def test_id(self):
        assert "[REDACTED_ID]" in redact_pii("CCCD 123456789012")

    def test_name_field(self):
        # redact via PII_REDACT_FIELDS
        assert "[REDACTED]" in redact_pii("name: Nguyen Van A")

    def test_none(self):
        assert redact_pii(None) is None
        assert redact_pii("") == ""


class TestMoodLog:
    def test_add_and_stats(self, temp_user):
        add_mood_log(temp_user, phq9_score=5, gad7_score=6, mood_notes="ok")
        add_mood_log(temp_user, phq9_score=20, gad7_score=18, mood_notes="very bad")
        stats = get_mood_stats(temp_user)
        assert stats["total_logs"] == 2
        assert stats["avg_phq9"] == pytest.approx(12.5, rel=1e-2)
        assert stats["avg_gad7"] == pytest.approx(12.0, rel=1e-2)
        assert stats["classification_counts"]["severe"] == 1

    def test_crisis_flag(self, temp_user):
        rec = add_mood_log(temp_user, phq9_score=10, gad7_score=10, mood_notes="I want to kill myself")
        assert rec["crisis_flag"] is True
        assert len(rec["crisis_keywords"]) >= 1
        assert rec["classification"] == "crisis"

    def test_pii_redacted(self, temp_user):
        rec = add_mood_log(temp_user, phq9_score=5, gad7_score=5, mood_notes="phone 0912345678 email a@b.com")
        assert rec["original_notes_had_pii"] is True
        assert "[REDACTED_PHONE]" in rec["mood_notes"]
        assert "[REDACTED_EMAIL]" in rec["mood_notes"]

    def test_validation(self, temp_user):
        with pytest.raises(ValueError):
            add_mood_log(temp_user, phq9_score=30)
        with pytest.raises(ValueError):
            add_mood_log(temp_user, gad7_score=30)

    def test_escalation_crisis(self, temp_user):
        add_mood_log(temp_user, phq9_score=5, gad7_score=5, mood_notes="tự tử")
        assert should_escalate_mood(temp_user) is True

    def test_no_escalation_minimal(self, temp_user):
        add_mood_log(temp_user, phq9_score=2, gad7_score=2, mood_notes="fine")
        assert should_escalate_mood(temp_user) is False

    def test_get_logs_limit(self, temp_user):
        for _ in range(5):
            add_mood_log(temp_user, phq9_score=5)
        logs = get_mood_logs(temp_user, limit=2)
        assert len(logs) == 2
