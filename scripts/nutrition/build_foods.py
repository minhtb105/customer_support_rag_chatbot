"""
Build foods JSON từ specs + 3 nguồn macros (VN FCT / USDA / manual) + công thức.

Chạy:  python scripts/nutrition/build_foods.py
Output:
  data/raw/nutrition/foods/<FOODID>.json  (68 món)
  data/raw/nutrition/_interim/REVIEW.csv   (món cần duyệt tay)
  fail nếu bất kỳ JSON nào không pass validator.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
INTERIM = ROOT / "data" / "raw" / "nutrition" / "_interim"
OUT_DIR = ROOT / "data" / "raw" / "nutrition" / "foods"

from formula import recommend  # noqa: E402
from validate import validate  # noqa: E402

DISCLAIMER = ("Số liệu tham khảo, có thể chênh lệch theo cách chế biến. "
              "Hỏi bác sĩ/dinh dưỡng viên cho khẩu phần cá nhân và theo dõi đường huyết.")

# Tên VN FCT giữ nguyên chính tả nguồn; alias giúp query đúng dù nguồn sai chính tả
FOOD_ALIASES = {
    "FRU037": ["mắc cọp"],
    "CON005": ["mắc cọp nước đường"],
}


def load_vn() -> dict:
    rows = list(csv.DictReader(
        open(INTERIM / "vn_fct_macros.csv", encoding="utf-8-sig")))
    return {r["ma_so"]: r for r in rows}


def usda_macros(query: str, cache: dict) -> tuple[dict, str]:
    """Pull USDA, ưu tiên Foundation > SR Legacy > Survey > Branded. Trả (macros, label)."""
    if query in cache:
        c = cache[query]
        return c["macros"], c["label"]
    sys.path.insert(0, str(HERE))
    from fetch_usda import search, detail, macros
    from dotenv import load_dotenv
    import os
    load_dotenv(ROOT / ".env")
    key = (os.getenv("USDA_API_KEY") or "").strip()
    cands = search(query, key, top=25)
    order = {"Foundation": 0, "SR Legacy": 1, "Survey (FNDDS)": 2, "Branded": 3}
    cands.sort(key=lambda c: order.get(c.get("dataType"), 9))
    picked = None
    errs = []
    for cand in cands:
        try:
            m = macros(detail(cand["fdcId"], key))
            missing = [k for k in ("energy_kcal", "total_carbohydrate_g", "protein_g",
                                   "total_fat_g") if m.get(k) is None]
            if missing:
                errs.append(f"{cand['fdcId']} thiếu {missing}")
                continue
            picked = (cand, m)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{cand.get('fdcId')} {type(e).__name__}")
    if not picked:
        raise ValueError(f"USDA không có bản đủ số liệu: {errs[:4]}")
    cand, m = picked
    if m.get("sugars_g") is None:
        m["sugars_g"] = 0.0
    if m.get("dietary_fiber_g") is None:
        m["dietary_fiber_g"] = 0.0
    label = (f"USDA {m['data_type']} fdcId={m['fdc_id']} ({m['description']})")
    cache[query] = {"macros": m, "label": label}
    return m, label


def resolve_macros(spec: dict, vn: dict, ucache: dict) -> tuple[dict, str, bool, str]:
    """Trả (macros, source_label, is_estimated, quality_note)."""
    src = spec["macro_src"]
    if src.startswith("vn:"):
        ma = src[3:]
        r = vn.get(ma)
        if not r:
            raise ValueError(f"{spec['food_id']}: không thấy mã VN {ma}")
        carbs = float(r["glucid_g"] or 0)
        fiber = float(r["fiber_g"] or 0)
        sugars = float(r["sugars_g"]) if r["sugars_g"] else None
        quality = ""
        if sugars is None:
            sugars, quality = 0.0, "sugars: chưa có số liệu gốc"
        elif sugars - max(0.0, carbs - fiber) > 0.51:
            sugars = round(max(0.0, carbs - fiber), 2)
            quality = "sugars gốc vượt carbs, đã clamp"
        return {"energy_kcal": float(r["energy_kcal"] or 0),
                "total_carbohydrate_g": carbs,
                "sugars_g": sugars,
                "dietary_fiber_g": fiber,
                "protein_g": float(r["protein_g"] or 0),
                "total_fat_g": float(r["lipid_g"] or 0)}, f"VN FCT 2007 mã {ma} ({r['ten_vi']})", False, quality
    if src.startswith("usda:"):
        m, label = usda_macros(src[5:], ucache)
        avail = max(0.0, m["total_carbohydrate_g"] - m["dietary_fiber_g"])
        quality = ""
        if m["sugars_g"] - avail > 0.51:
            m["sugars_g"] = round(avail, 2)
            quality = "sugars nhãn vượt carbs, đã clamp"
        return {k: float(m[k]) for k in ("energy_kcal", "total_carbohydrate_g", "sugars_g",
                                        "dietary_fiber_g", "protein_g", "total_fat_g")}, label, False, quality
    if src.startswith("manual:"):
        vals = [float(x) for x in src[7:].split(";")]
        keys = ("energy_kcal", "total_carbohydrate_g", "dietary_fiber_g", "sugars_g",
                "protein_g", "total_fat_g")
        # thứ tự spec: kcal;carb;fiber;sugar;protein;fat
        order = ("energy_kcal", "total_carbohydrate_g", "dietary_fiber_g", "sugars_g",
                 "protein_g", "total_fat_g")
        assert len(vals) == 6, src
        return dict(zip(order, vals)), "ước tính từ thành phần món (cần duyệt)", True, ""
    raise ValueError(f"macro_src lạ: {src}")


def build_notes(spec: dict, rec: dict) -> str:
    parts = [spec["notes"]]
    if rec["gi_category"] == "High":
        parts.append("GI cao: ăn đúng khẩu phần gợi ý, kết hợp rau/đạm, không ăn lúc đói.")
    elif rec["gi_category"] == "Low":
        parts.append("GI thấp: lựa chọn tốt trong nhóm, vẫn giữ đúng khẩu phần.")
    parts.append(DISCLAIMER)
    return " ".join(parts)


def main() -> None:
    vn = load_vn()
    ucache: dict = {}
    try:
        ucache = json.loads((INTERIM / "usda_cache.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    specs = list(csv.DictReader(
        open(HERE / "foods_spec.csv", encoding="utf-8-sig")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review, errors = [], []
    for spec in specs:
        fid = spec["food_id"]
        try:
            macros, source_label, is_est, quality = resolve_macros(spec, vn, ucache)
            if quality:
                source_label += f" ({quality})"
            serving = float(spec["serving_g"])
            gi = float(spec["gi"]) if spec["gi"] else None
            rec = recommend(macros["total_carbohydrate_g"], macros["dietary_fiber_g"],
                            serving, gi)
            food = {
                "food_id": fid,
                "names": {"vi": spec["vi"], "en": spec["en"],
                          "aliases_vi": [a for a in spec["aliases"].split("|") if a]
                          + FOOD_ALIASES.get(fid, [])},
                "category": spec["category"],
                "serving": {"size_g": serving, "description": spec["serving_desc"]},
                "nutrition_per_100g": {k: round(v, 2) for k, v in macros.items()},
                "glycemic": {"gi": gi, "gi_tier": spec["gi_tier"]},
                "diabetes_recommendation": {
                    "mild_g": rec["mild_g"], "severe_g": rec["severe_g"],
                    "gl_std": rec["gl_std"], "gl_mild": rec["gl_mild"],
                    "gl_severe": rec["gl_severe"], "choices_std": rec["choices_std"],
                    "choices_mild": rec["choices_mild"],
                    "choices_severe": rec["choices_severe"],
                    "notes": build_notes(spec, rec),
                },
                "metadata": {
                    "source": [source_label, f"GI: {spec['gi_ref']} [{spec['gi_tier']}]"],
                    "is_local_vietnam": spec["is_local"].lower() == "true",
                    "last_updated": "2026-10-06",
                    "review_flag": spec["review"].lower() == "true" or is_est
                    or spec["gi_tier"] in ("proxy", "estimated") or bool(quality),
                },
            }
            errs = validate(food)
            if errs:
                errors.append(f"{fid}: {errs}")
                continue
            (OUT_DIR / f"{fid}.json").write_text(
                json.dumps(food, ensure_ascii=False, indent=1), encoding="utf-8")
            if food["metadata"]["review_flag"]:
                reasons = []
                if is_est or spec["gi_tier"] in ("proxy", "estimated"):
                    reasons.append("proxy/estimated/manual")
                if quality:
                    reasons.append(quality)
                if spec["review"].lower() == "true" and not reasons:
                    reasons.append("spec review=true")
                review.append({"food_id": fid, "vi": spec["vi"],
                               "reason": "+".join(reasons),
                               "gi": gi, "tier": spec["gi_tier"], "macro_src": spec["macro_src"]})
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fid}: {type(e).__name__} {e}")
    (INTERIM / "usda_cache.json").write_text(
        json.dumps(ucache, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with open(INTERIM / "REVIEW.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["food_id", "vi", "reason", "gi", "tier", "macro_src"])
        w.writeheader()
        w.writerows(review)
    log.info("OK %d foods, REVIEW %d, ERRORS %d",
             len(specs) - len(errors), len(review), len(errors))
    for e in errors:
        log.error(e)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
