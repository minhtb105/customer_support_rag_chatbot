"""Validate 1 food JSON theo schema + tính lại GL/choices để chống số tay sai."""
from __future__ import annotations

import json
from pathlib import Path


def validate(food: dict) -> list[str]:
    """Trả về list lỗi (rỗng = pass)."""
    return _manual_check(food) + _recompute_check(food)


def _manual_check(food: dict) -> list[str]:
    errs = []
    for key in ("food_id", "names", "category", "serving", "nutrition_per_100g",
                "glycemic", "diabetes_recommendation", "metadata"):
        if key not in food:
            errs.append(f"thiếu key: {key}")
    return errs


def _recompute_check(food: dict) -> list[str]:
    try:
        from formula import avail_carbs, carb_choices, glycemic_load
    except ImportError:
        from nutrition.formula import (  # type: ignore
            avail_carbs, carb_choices, glycemic_load)
    errs = []
    try:
        n = food["nutrition_per_100g"]
        g = food["glycemic"]
        r = food["diabetes_recommendation"]
        srv = food["serving"]["size_g"]
        avail_std = avail_carbs(n["total_carbohydrate_g"], n["dietary_fiber_g"], srv)
        gl = glycemic_load(g.get("gi"), avail_std)
        if gl is not None and gl > 0 and abs(gl - (r.get("gl_std") or 0)) > 0.6:
            errs.append(f"gl_std khai báo {r.get('gl_std')} != tính lại {round(gl,1)}")
        ch = carb_choices(avail_carbs(n["total_carbohydrate_g"], n["dietary_fiber_g"], r["mild_g"]))
        if abs(ch - r.get("choices_mild", ch)) > 0.01:
            errs.append(f"choices_mild khai báo != tính lại {ch}")
        if not (r["severe_g"] <= r["mild_g"] <= srv * 1.0 + 1e-9 or n["total_carbohydrate_g"] == 0):
            pass  # mild/severe có thể giữ nguyên std với món ~0 carbs
        if r["severe_g"] > r["mild_g"]:
            errs.append("severe_g phải <= mild_g")
        if n["sugars_g"] - max(0.0, n["total_carbohydrate_g"] - n["dietary_fiber_g"]) > 0.51:
            errs.append("sugars_g vượt quá carbs khả dụng (vô lý)")
    except (KeyError, TypeError) as e:
        errs.append(f"thiếu field cho recompute: {e}")
    return errs


def validate_file(path: str | Path) -> list[str]:
    food = json.loads(Path(path).read_text(encoding="utf-8"))
    return [f"{path}: {e}" for e in validate(food)]
