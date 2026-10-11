"""
Công thức khẩu phần mild/severe — pure, deterministic, test được.

- carbs khả dụng = (total_carb - fiber), KHÔNG dùng đường tổng (đường <= khả dụng).
- GL = GI * carbs_khả_dụng(khẩu phần) / 100. Ngưỡng: thấp ≤10, TB 11–19, cao ≥20.
- mild: co khẩu phần chuẩn về GL≈10 (sàn 50g), đồng thời cap ≤2 carb-choice.
- severe: 70% mild (hướng GL≈7).
- Không GI: giữ khẩu phần chuẩn, notes generic + disclaimer.
"""
from __future__ import annotations

import math

GL_LOW_MAX = 10.0
GL_SEVERE_TARGET = 7.0
SEVERE_RATIO = 0.7
CARB_CHOICE_G = 15.0
MAX_CHOICES_MILD = 2.0
MIN_SERVING_G = 50.0


def round_nice(g: float) -> float:
    """Làm tròn về bội 10 gần nhất, sàn 50g (trừ khi đầu vào đã nhỏ hơn)."""
    if g <= 0:
        raise ValueError("serving phải > 0")
    return float(max(MIN_SERVING_G, round(g / 10.0) * 10))


def round_half(x: float) -> float:
    return round(x * 2.0) / 2.0


def avail_carbs(total_carb_100g: float, fiber_100g: float, serving_g: float) -> float:
    avail_100 = max(0.0, total_carb_100g - fiber_100g)
    return avail_100 * serving_g / 100.0


def carb_choices(avail_g: float) -> float:
    return round_half(max(0.0, avail_g) / CARB_CHOICE_G)


def glycemic_load(gi: float | None, avail_g: float) -> float | None:
    if gi is None:
        return None
    return gi * max(0.0, avail_g) / 100.0


def gi_category(gi: float | None) -> str:
    if gi is None:
        return "Unknown"
    if gi <= 55:
        return "Low"
    if gi <= 69:
        return "Medium"
    return "High"


def recommend(total_carb_100g: float, fiber_100g: float, serving_g: float,
              gi: float | None) -> dict:
    """Trả về dict đủ field để build JSON: mild_g/severe_g/gl_mild/gl_severe/
    choices_mild/choices_severe/gi_category + notes gợi ý (caller thêm notes tay)."""
    avail_std = avail_carbs(total_carb_100g, fiber_100g, serving_g)
    gl_std = glycemic_load(gi, avail_std)
    if gl_std is None or gl_std <= 0:
        mild_g = float(serving_g)
    else:
        mild_g = serving_g * min(1.0, GL_LOW_MAX / gl_std)
        # cap ≤2 carb-choice
        if avail_100g := max(0.0, total_carb_100g - fiber_100g):
            cap_g = MAX_CHOICES_MILD * CARB_CHOICE_G * 100.0 / avail_100g
            mild_g = min(mild_g, cap_g)
        mild_g = round_nice(mild_g)
    severe_g = min(mild_g, round_nice(mild_g * SEVERE_RATIO))
    avail_mild = avail_carbs(total_carb_100g, fiber_100g, mild_g)
    avail_sev = avail_carbs(total_carb_100g, fiber_100g, severe_g)
    return {
        "mild_g": mild_g,
        "severe_g": severe_g,
        "gl_std": round(gl_std, 1) if gl_std is not None else None,
        "gl_mild": glycemic_load(gi, avail_mild) and round(glycemic_load(gi, avail_mild), 1),
        "gl_severe": glycemic_load(gi, avail_sev) and round(glycemic_load(gi, avail_sev), 1),
        "choices_std": carb_choices(avail_std),
        "choices_mild": carb_choices(avail_mild),
        "choices_severe": carb_choices(avail_sev),
        "gi_category": gi_category(gi),
    }
