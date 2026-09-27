"""vn_parse gold — Vietnamese spoken numbers, ±dau, zero-error policy.

Any gold fail = fix parser, never lower the bar. Ambiguous cases must
return need-re-speak (ambiguous=True), never a guess.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voice.vn_parse import parse_vi


def _ok(text, value, context):
    r = parse_vi(text)
    assert r["ambiguous"] is False, f"{text!r} -> ambiguous ({r['reason']})"
    assert r["value"] == value, f"{text!r} -> {r['value']} != {value}"
    assert r["context"] == context, f"{text!r} -> {r['context']} != {context}"
    assert str(value) in r["confirm_text"]


def _amb(text):
    r = parse_vi(text)
    assert r["ambiguous"] is True, f"{text!r} should be ambiguous, got {r}"


def test_digits_passthrough_accented():
    _ok("126 lúc đói", 126, "fasting")
    _ok("280 lúc đói", 280, "fasting")


def test_digits_passthrough_unaccented():
    _ok("126 luc doi", 126, "fasting")
    _ok("140 sau an", 140, "post_meal_2h")


def test_words_compound():
    _ok("một trăm hai mươi sáu lúc đói", 126, "fasting")
    _ok("mot tram hai muoi sau luc doi", 126, "fasting")


def test_words_tam_ba():
    _ok("một trăm tám lúc đói", 108, "fasting")
    _ok("một trăm ba lúc đói", 103, "fasting")


def test_ruoi_le():
    _ok("một trăm rưỡi lúc đói", 150, "fasting")
    _ok("mot tram le sau luc doi", 106, "fasting")


def test_nghin_out_of_range_ambiguous():
    _amb("một nghìn lúc đói")


def test_decimal_ambiguous():
    _amb("một trăm phẩy năm lúc đói")
    _amb("126 chấm 5 lúc đói")


def test_range_ambiguous():
    _amb("từ 100 đến 120 lúc đói")
    _amb("100-120 lúc đói")


def test_mixed_digit_words_ambiguous():
    _amb("1 trăm lúc đói")


def test_no_context_ambiguous():
    _amb("một trăm hai mươi sáu")
    _amb("126")


def test_no_number_ambiguous():
    _amb("lúc đói")
    _amb("xin chào")


def test_out_of_range_ambiguous():
    _amb("10 lúc đói")
    _amb("700 lúc đói")


def test_context_variants():
    _ok("140 sau ăn", 140, "post_meal_2h")
    _ok("110 trước ngủ", 110, "bedtime")
    _ok("95 trước ăn", 95, "pre_meal")


def test_multi_number_ambiguous():
    _amb("126 và 140 lúc đói")
