"""
Pull macros/100g từ USDA FoodData Central cho món nhập khẩu (danh sách chốt).

Khớp thủ công: in top candidates để chọn fdcId đúng (Foundation/SR Legacy ưu tiên),
rồi lưu vào foods_spec. Bước này bán tự động, có review.
Chạy:  python scripts/nutrition/fetch_usda.py "guava raw"
       python scripts/nutrition/fetch_usda.py --batch
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = "https://api.nal.usda.gov/fdc/v1"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"

BATCH_QUERIES = [
    "apple raw", "grapes raw", "oats", "milk whole", "yogurt plain",
    "chocolate milk", "bread white", "potato raw", "sweet potato raw",
    "peanuts", "chicken breast", "egg whole", "soda cola", "orange juice",
    "ice cream vanilla",
]

WANT_IDS = (1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
            1079, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1003, 1004, 1005,
            2000)


def _key() -> str:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    k = (os.getenv("USDA_API_KEY") or "").strip().strip("\"'")
    if not k:
        raise RuntimeError("thiếu USDA_API_KEY trong .env")
    return k


def _headers(api_key: str) -> dict:
    return {"X-Api-Key": api_key, "Content-Type": "application/json"}


def search(query: str, api_key: str, top: int = 5) -> list[dict]:
    import requests
    r = requests.post(f"{BASE}/foods/search",
                      headers=_headers(api_key),
                      json={"query": query, "pageSize": top},
                      timeout=60)
    r.raise_for_status()
    return r.json().get("foods", [])


def detail(fdc_id: int, api_key: str) -> dict:
    import requests
    # KHÔNG lọc nutrients (param lọc trả rỗng) — lấy full rồi lọc ở macros()
    r = requests.get(f"{BASE}/food/{fdc_id}",
                     headers=_headers(api_key),
                     timeout=60)
    r.raise_for_status()
    return r.json()


def macros(food: dict) -> dict:
    """food detail -> dict macros/100g (None nếu thiếu)."""
    vals = {}
    for n in food.get("foodNutrients", []):
        nid = n.get("nutrient", {}).get("id", n.get("nutrientId"))
        if nid in WANT_IDS and nid not in vals:
            v = n.get("amount", n.get("value"))
            if v is not None:
                vals[nid] = v
    g = lambda *ids: next((vals[i] for i in ids if vals.get(i) is not None), None)
    return {
        "fdc_id": food.get("fdcId"),
        "description": food.get("description"),
        "data_type": food.get("dataType"),
        "energy_kcal": g(1008),
        "protein_g": g(1003),
        "total_fat_g": g(1004),
        "total_carbohydrate_g": g(1005),
        "dietary_fiber_g": g(1079),
        "sugars_g": g(2000, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018),
    }


def main() -> None:
    key = _key()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        for f in search(sys.argv[1], key):
            print(f.get("fdcId"), "|", f.get("dataType"), "|", f.get("description"))
        return
    out = {}
    for q in BATCH_QUERIES:
        try:
            cands = search(q, key, top=3)
            out[q] = [{"fdcId": c.get("fdcId"), "dataType": c.get("dataType"),
                       "description": c.get("description")} for c in cands]
            log.info("%s -> %s", q, out[q][0]["description"] if out[q] else None)
        except Exception as e:  # noqa: BLE001
            log.warning("%s FAIL: %s", q, str(e)[:150])
    (OUT / "usda_candidates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Ghi candidates -> usda_candidates.json (chọn fdcId tay ở build step)")


if __name__ == "__main__":
    main()
