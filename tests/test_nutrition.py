"""Unit test công thức khẩu phần + validator (chạy offline, không cần API)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nutrition.formula import (  # noqa: E402
    avail_carbs, carb_choices, glycemic_load, gi_category, recommend,
)
from nutrition.validate import validate  # noqa: E402


def test_oi_example_matches_spec():
    # Ví dụ trong spec: ổi 100g — carbs 14.3, fiber 5.4, GI 31 -> GL 2.8
    assert avail_carbs(14.3, 5.4, 100) == 8.9
    assert glycemic_load(31, 8.9) == 2.8 or abs(glycemic_load(31, 8.9) - 2.8) < 0.05
    assert gi_category(31) == "Low"
    r = recommend(14.3, 5.4, 100, 31)
    assert r["mild_g"] == 100 and r["severe_g"] == 70
    assert r["choices_mild"] == 0.5


def test_com_trang_co_ve_gl_cao():
    # Cơm trắng 150g (chén): carbs ~28/100g, fiber 0.4, GI 73
    r = recommend(28.0, 0.4, 150, 73)
    assert r["gi_category"] == "High"
    assert r["gl_std"] > 20
    assert r["mild_g"] < 150  # phải co khẩu phần
    assert r["severe_g"] <= r["mild_g"]
    assert r["gl_mild"] <= 10.5


def test_mon_khong_gi_giu_khau_phan():
    r = recommend(0.0, 0.0, 100, None)
    assert r["mild_g"] == 100 and r["severe_g"] == 70
    assert r["gl_std"] is None and r["gi_category"] == "Unknown"


def test_validator_catches_bad_sugars():
    bad = {
        "food_id": "FR001", "names": {"vi": "X", "en": "Y"}, "category": "fruit",
        "serving": {"size_g": 100, "description": "100g"},
        "nutrition_per_100g": {"energy_kcal": 50, "total_carbohydrate_g": 10,
                               "sugars_g": 9.5, "dietary_fiber_g": 5,
                               "protein_g": 1, "total_fat_g": 0.5},
        "glycemic": {"gi": 40, "gi_tier": "table1"},
        "diabetes_recommendation": {"mild_g": 100, "severe_g": 70, "gl_std": 2.0,
                                    "gl_mild": 2.0, "gl_severe": 1.4,
                                    "choices_std": 0.5, "choices_mild": 0.5,
                                    "choices_severe": 0.5, "notes": "ok"},
        "metadata": {"source": ["t"], "is_local_vietnam": True,
                     "last_updated": "2026-10-06"},
    }
    errs = validate(bad)
    assert any("sugars_g" in e for e in errs), errs
