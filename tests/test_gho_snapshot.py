"""Tests cho GHO Dual-Storage: ETL, textualize, SQLite tool, intent, SSRF guard."""
import sqlite3

import pytest

from src.monitors import gho_snapshot as g
from src.monitors.utils import assert_url_allowed


def _rows():
    return [
        {"IndicatorCode": "X", "SpatialDim": "VNM", "TimeDimType": "YEAR", "TimeDim": 2010,
         "Dim1": "SEX_BTSX", "NumericValue": 5.4, "Low": 4.0, "High": 7.0},
        {"IndicatorCode": "X", "SpatialDim": "VNM", "TimeDimType": "YEAR", "TimeDim": 2020,
         "Dim1": "SEX_BTSX", "NumericValue": 7.3, "Low": 6.0, "High": 9.0},
        # non-YEAR bi loai
        {"IndicatorCode": "X", "SpatialDim": "VNM", "TimeDimType": "DAY", "TimeDim": 5,
         "Dim1": "SEX_BTSX", "NumericValue": 99.0, "Low": 0, "High": 0},
        # trung lap bi dedup
        {"IndicatorCode": "X", "SpatialDim": "VNM", "TimeDimType": "YEAR", "TimeDim": 2020,
         "Dim1": "SEX_BTSX", "NumericValue": 7.3, "Low": 6.0, "High": 9.0},
    ]


def test_etl_keeps_year_dedups():
    recs = g.etl_records("X", _rows())
    assert len(recs) == 2
    assert recs[0]["year"] == 2010 and recs[1]["year"] == 2020
    assert recs[1]["value"] == 7.3


def test_textualize_mentions_latest_and_delta(monkeypatch):
    monkeypatch.setitem(g.GHO_INDICATORS, "X", {"vi": "tỷ lệ mẫu", "unit": "%"})
    recs = g.etl_records("X", _rows())
    lines = g.textualize(recs)
    assert any("2020" in ln and "7.3%" in ln for ln in lines)
    assert any("1.9" in ln for ln in lines)  # 7.3 - 5.4


def test_sqlite_upsert_and_tool_math(monkeypatch, tmp_path):
    db = tmp_path / "gho.db"
    monkeypatch.setattr(g, "GHO_DB_PATH", str(db))
    recs = g.etl_records("NCD_DIABETES_PREVALENCE_CRUDE", [
        {"IndicatorCode": "NCD_DIABETES_PREVALENCE_CRUDE", "SpatialDim": "VNM",
         "TimeDimType": "YEAR", "TimeDim": 2012, "Dim1": "SEX_BTSX",
         "NumericValue": 5.4, "Low": 1, "High": 2},
        {"IndicatorCode": "NCD_DIABETES_PREVALENCE_CRUDE", "SpatialDim": "VNM",
         "TimeDimType": "YEAR", "TimeDim": 2022, "Dim1": "SEX_BTSX",
         "NumericValue": 7.3, "Low": 1, "High": 2},
    ])
    assert g.upsert_sqlite(recs) == 2
    assert g.upsert_sqlite(recs) == 2  # idempotent
    n = sqlite3.connect(str(db)).execute("select count(*) from gho_stats").fetchone()[0]
    assert n == 2
    out = g.query_gho_stats("tỷ lệ tăng bao nhiêu?")
    assert out is not None and "1.9" in out


def test_query_returns_none_without_db(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "GHO_DB_PATH", str(tmp_path / "missing.db"))
    assert g.query_gho_stats("tỷ lệ?") is None


def test_intent_keywords():
    assert g.looks_like_stats_question("Tỷ lệ tiểu đường ở Việt Nam hiện nay?")
    assert g.looks_like_stats_question("Tăng bao nhiêu phần trăm so với 10 năm trước?")
    assert not g.looks_like_stats_question("Triệu chứng của hạ đường huyết là gì?")


def test_ssrf_allows_gho_rejects_private():
    assert_url_allowed("https://ghoapi.azureedge.net/api/NCD_DIABETES_PREVALENCE_CRUDE")
    with pytest.raises(Exception):
        assert_url_allowed("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(Exception):
        assert_url_allowed("http://127.0.0.1:8000/v1/health")
