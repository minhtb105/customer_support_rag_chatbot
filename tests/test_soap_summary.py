"""Unit tests for Track B — soap_summary (diabetes pre-visit)."""

import pytest
from src.features.soap_summary import generate_soap, soap_to_markdown
from src.features.glucose_tracker import add_log


class TestSoapSummary:
    def test_rule_based_fallback_no_openai(self, temp_user, mock_openai):
        add_log(temp_user, 130, context="fasting")
        add_log(temp_user, 180, context="post_meal_2h")
        data = generate_soap(temp_user, days=14, language="vi")
        assert "soap" in data
        for key in ("subjective", "objective", "assessment"):
            assert key in data["soap"]
            assert isinstance(data["soap"][key], str)
            assert len(data["soap"][key]) > 0
        # P is ALWAYS empty — the doctor decides (placeholder rendered by UI)
        assert data["soap"]["plan"] == ""

    def test_soap_audit_links_and_hba1c(self, temp_user, mock_openai):
        add_log(temp_user, 130, context="fasting", notes="khát nước nhiều")
        add_log(temp_user, 180, context="post_meal_2h")
        data = generate_soap(temp_user, days=14, language="vi")
        assert "HbA1c" in data["soap"]["assessment"]
        assert "[Xem log #" in data["soap"]["subjective"]
        assert "[Xem log #" in data["soap"]["objective"]
        assert "[Xem log #" in data["soap"]["assessment"]
        assert data["soap"]["plan"] == ""

    def test_soap_contains_kpi(self, temp_user, mock_openai):
        add_log(temp_user, 95, context="fasting")
        data = generate_soap(temp_user, days=7, language="vi")
        # rule-based mentions KPI when <3/week
        assert "KPI" in data["soap"]["subjective"] or "KPI" in data["soap"]["assessment"]

    def test_soap_no_logs(self, temp_user, mock_openai):
        data = generate_soap(temp_user, days=14, language="vi")
        assert data["stats"]["total_logs"] == 0
        assert "Chưa có" in data["soap"]["subjective"] or "Chưa có" in data["soap"]["objective"] or "No" in data["soap"]["objective"]

    def test_markdown_format(self, temp_user, mock_openai):
        add_log(temp_user, 110, context="fasting")
        data = generate_soap(temp_user, days=7, language="vi")
        md = soap_to_markdown(data)
        assert "# Tóm tắt trước tái khám" in md
        assert "## S — Subjective" in md
        assert "## O — Objective" in md
        assert "## A — Assessment" in md
        assert "## P — Plan" in md
        assert "không thay thế" in md

    def test_period_field(self, temp_user, mock_openai):
        add_log(temp_user, 100, context="random")
        data = generate_soap(temp_user, days=30, language="vi")
        assert data["period"] == "30 ngày"
