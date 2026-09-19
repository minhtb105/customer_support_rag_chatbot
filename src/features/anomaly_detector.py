"""
Chronic Time-Series — Anomaly middleware (Diabetes demo).

Pure functions, no LLM:
- detect_spike: hard thresholds >250 / <70 (safety-first).
  Low/critical classification message from glucose_tracker takes
  priority over FQG phrasing (see wire in src/api/main.py).
- detect_trend: heuristic demo only (NOT a medical standard):
  group fasting logs by day -> daily avg -> 3 consecutive days where
  each day rises 10-15% vs previous day. Skip missing days, guard div-by-zero,
  ignore non-fasting contexts.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List


SPIKE_HIGH = 250
SPIKE_LOW = 70

# Follow-up questions (FQG templates, phrasing may be refined by gpt-4o-mini;
# full multi-turn FQG is future-work — see README).
FQG_SPIKE_HIGH = [
    "Gần đây bạn có ăn đồ ngọt / tinh bột nhiều hơn bình thường không?",
    "Bạn có quên uống thuốc hoặc bỏ liều nào không?",
    "Bạn có đang stress, ốm, hoặc mất ngủ không?",
]
FQG_SPIKE_LOW = [
    "Bạn có bỏ bữa hoặc ăn ít hơn bình thường không?",
    "Bạn có uống thuốc / tiêm insulin rồi nhưng chưa ăn không?",
    "Bạn có vận động mạnh hơn bình thường không?",
]
FQG_TREND = [
    "3 ngày gần đây bữa tối của bạn có nhiều tinh bột / đồ ngọt không?",
    "Bạn có thay đổi thuốc hoặc giờ uống thuốc không?",
    "Giấc ngủ và vận động tuần này có gì khác không?",
]


def detect_spike(value_mgdl: float) -> Dict[str, Any]:
    """Hard spike check. Returns {type, reason} with type in spike_high/spike_low/none."""
    try:
        v = float(value_mgdl)
    except (TypeError, ValueError):
        return {"type": "none", "reason": ""}
    if v > SPIKE_HIGH:
        return {
            "type": "spike_high",
            "reason": f"Spike cao: {v:g} mg/dL vượt ngưỡng cứng {SPIKE_HIGH} mg/dL — cần xử trí an toàn ngay.",
        }
    if v < SPIKE_LOW:
        return {
            "type": "spike_low",
            "reason": f"Spike thấp: {v:g} mg/dL dưới ngưỡng cứng {SPIKE_LOW} mg/dL — nguy cơ hạ đường huyết, xử trí ngay.",
        }
    return {"type": "none", "reason": ""}


def _parse_day(measured_at: Any) -> str | None:
    if not measured_at:
        return None
    s = str(measured_at)
    try:
        # ISO "YYYY-MM-DDTHH:MM:SS..." -> date part
        if "T" in s:
            return s.split("T")[0]
        dt = datetime.fromisoformat(s)
        return dt.date().isoformat()
    except Exception:
        return s[:10] if len(s) >= 10 else None


def detect_trend(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic demo trend: 3 consecutive days of fasting avg rising 10-15%/day.

    - Only context == "fasting" counts.
    - Group by day -> mean/day, sorted ascending.
    - Need 3 days where day[i] and day[i+1] and day[i+2] exist as
      consecutive calendar days (missing days break the chain).
    - Each step must satisfy 0.10 <= (curr-prev)/prev <= 0.15 (guard prev == 0).
    - Returns {type: trend_cascade|none, reason}.
    """
    if not logs:
        return {"type": "none", "reason": ""}
    by_day: Dict[str, List[float]] = defaultdict(list)
    for l in logs or []:
        try:
            if (l.get("context") or "") != "fasting":
                continue
            v = float(l.get("value_mgdl"))
        except (TypeError, ValueError, AttributeError):
            continue
        day = _parse_day(l.get("measured_at"))
        if not day:
            continue
        by_day[day].append(v)
    if len(by_day) < 3:
        return {"type": "none", "reason": ""}
    days_sorted = sorted(by_day.keys())
    avgs = {d: sum(vs) / len(vs) for d, vs in by_day.items()}
    # map day string -> ordinal for consecutiveness check
    try:
        ordinals = {d: datetime.fromisoformat(d).date().toordinal() for d in days_sorted}
    except Exception:
        return {"type": "none", "reason": ""}
    for i in range(len(days_sorted) - 2):
        d0, d1, d2 = days_sorted[i], days_sorted[i + 1], days_sorted[i + 2]
        if ordinals[d1] != ordinals[d0] + 1 or ordinals[d2] != ordinals[d1] + 1:
            continue
        a0, a1, a2 = avgs[d0], avgs[d1], avgs[d2]
        if a0 == 0 or a1 == 0:
            continue
        r1 = (a1 - a0) / a0
        r2 = (a2 - a1) / a1
        if 0.10 <= r1 <= 0.15 and 0.10 <= r2 <= 0.15:
            return {
                "type": "trend_cascade",
                "reason": (
                    f"Cascade tăng lúc đói: {d0} {a0:.0f} → {d1} {a1:.0f} "
                    f"(+{r1*100:.0f}%) → {d2} {a2:.0f} (+{r2*100:.0f}%). "
                    "Heuristic demo, không phải chuẩn y khoa."
                ),
            }
    return {"type": "none", "reason": ""}


def analyze_glucose_log(value_mgdl: float, recent_logs: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Combine spike + trend. Spike wins (safety-first). Returns {anomaly, follow_up_questions}."""
    spike = detect_spike(value_mgdl)
    if spike["type"] == "spike_high":
        return {"anomaly": {"type": "spike", "direction": "high", "reason": spike["reason"], "detail": spike["type"]}, "follow_up_questions": list(FQG_SPIKE_HIGH)}
    if spike["type"] == "spike_low":
        return {"anomaly": {"type": "spike", "direction": "low", "reason": spike["reason"], "detail": spike["type"]}, "follow_up_questions": list(FQG_SPIKE_LOW)}
    trend = detect_trend(recent_logs or [])
    if trend["type"] == "trend_cascade":
        return {"anomaly": {"type": "trend", "direction": "up", "reason": trend["reason"], "detail": trend["type"]}, "follow_up_questions": list(FQG_TREND)}
    return {"anomaly": {"type": "none", "reason": ""}, "follow_up_questions": []}
