"""
P3 — Tra cứu dinh dưỡng cấu trúc (không LLM): query món ăn -> số liệu + khẩu phần.

Tái dùng foods JSON (data/raw/nutrition/foods/) + alias matching (vi/en/aliases),
không gọi embeddings/LLM nên nhanh (<50ms) và deterministic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.shared.config import BASE_DIR
except ImportError:
    from shared.config import BASE_DIR  # type: ignore

FOODS_DIR = Path(str(BASE_DIR)) / "data" / "raw" / "nutrition" / "foods"


@lru_cache(maxsize=1)
def _load_foods() -> List[Dict[str, Any]]:
    out = []
    if not FOODS_DIR.is_dir():
        return out
    for fp in sorted(FOODS_DIR.glob("*.json")):
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _alias_in_query(alias: str, q_padded: str) -> bool:
    """Như retriever._alias_in_query (copy để khỏi kéo langchain vào endpoint
    nhẹ): alias <=3 ký tự phải khớp nguyên từ."""
    import re as _re
    if not alias:
        return False
    if len(alias) <= 3:
        return _re.search(r"(?<!\w)" + _re.escape(alias) + r"(?!\w)", q_padded) is not None
    return alias in q_padded


def lookup_foods(query: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Match tên VI/EN/alias với query, 2 chiều (tên trong query hoặc query trong tên).

    - alias dài trong query ("trà sữa 500ml" không match, nhưng "trà sữa" match).
    - query ngắn nằm gọn trong tên ("trà sữa" trong "Trà sữa 500ml").
    """
    q = f" {_norm(query)} "
    if len(q.strip()) < 2:
        return []
    qtokens = [t for t in q.split() if len(t) >= 2]
    scored = []
    for f in _load_foods():
        names = [f.get("names", {}).get("vi", ""), f.get("names", {}).get("en", "")]
        names += (f.get("names", {}).get("aliases_vi", []) or [])
        best, hit = 0, ""
        for n in names:
            nl = _norm(n)
            if len(nl) < 2:
                continue
            if _alias_in_query(nl, q) and len(nl) > best:
                best, hit = len(nl) * 2, nl
            elif qtokens and all(_alias_in_query(t, f" {nl} ") for t in qtokens):
                s = len(_norm(query))
                if s > best:
                    best, hit = s, nl
        if best:
            scored.append((best, f, hit))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, f, hit in scored:
        if f["food_id"] not in seen:
            seen.add(f["food_id"])
            out.append({"food": f, "matched": hit})
        if len(out) >= top_n:
            break
    return out


def to_result(item: Dict[str, Any], profile: str = "both") -> Dict[str, Any]:
    f = item["food"]
    r = f["diabetes_recommendation"]
    rec: Dict[str, Any] = {}
    if profile in ("mild", "both"):
        rec["mild"] = {"serving_g": r["mild_g"], "gl": r["gl_mild"],
                       "carb_choices": r["choices_mild"]}
    if profile in ("severe", "both"):
        rec["severe"] = {"serving_g": r["severe_g"], "gl": r["gl_severe"],
                         "carb_choices": r["choices_severe"]}
    return {
        "food_id": f["food_id"],
        "vi": f["names"]["vi"],
        "en": f["names"]["en"],
        "matched": item["matched"],
        "category": f["category"],
        "serving": f["serving"],
        "per_100g": f["nutrition_per_100g"],
        "gi": f["glycemic"]["gi"],
        "gi_tier": f["glycemic"]["gi_tier"],
        "recommendation": rec,
        "notes": r["notes"],
        "source": f["metadata"]["source"],
        "review_flag": f["metadata"].get("review_flag", False),
    }
