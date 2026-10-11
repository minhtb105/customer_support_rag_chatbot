"""
Sinh spec rows cho món mới từ REVIEW.csv (append vào foods_spec.csv, KHÔNG ghi đè).

Quy tắc điền field thiếu:
- category: từ prefix food_id (CON->canned, SPC->spice).
- en: ten_en VN FCT theo mã vn: (fallback: tên VI + review).
- serving: (1) VDD quydoi theo mã vn (ưu tiên đơn vị nguyên chiếc, 20-600g);
  (2) default theo category + review.
- notes: reason của REVIEW (ghi chú chuyên môn của user).
- aliases/is_local: rỗng/true.
Chạy:  python scripts/nutrition/gen_new_specs.py
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERIM = ROOT / "data" / "raw" / "nutrition" / "_interim"
SPEC = HERE / "foods_spec.csv"

CAT = {"GRC": "grain", "TUB": "tuber", "FRU": "fruit", "VEG": "vegetable",
       "PRO": "protein", "DAI": "dairy", "BEV": "beverage", "SNK": "snack_sweet",
       "MIX": "mixed_dish", "CON": "canned", "SPC": "spice"}
# default serving (g, desc) khi không có VDD/USDA/label
DEFAULT_SRV = {
    "grain": (100, "1 phần"),
    "tuber": (150, "1 củ/khúc vừa"),
    "fruit": (100, "1 phần"),
    "vegetable": (200, "1 đĩa/bát"),
    "protein": (100, "1 phần"),
    "dairy": (200, "1 ly/hũ"),
    "beverage": (250, "1 ly"),
    "snack_sweet": (30, "1 phần nhỏ"),
    "mixed_dish": (300, "1 phần"),
    "canned": (100, "1 phần"),
    "spice": (5, "1 thìa cà phê"),
}
WHOLE_UNIT = re.compile(r"miệng bát|1 quả|1 củ|1 miếng|1 gói|1 bắp|1 cây|1 con|1 quả|nguyên (quả|củ|con)|1 bát|1 chén", re.I)


def load_vn() -> dict:
    return {r["ma_so"]: r for r in
            csv.DictReader(open(INTERIM / "vn_fct_macros.csv", encoding="utf-8-sig"))}


def load_vdd() -> dict:
    m: dict[str, list] = {}
    try:
        for r in csv.DictReader(open(INTERIM / "quydoi_servings.csv", encoding="utf-8-sig")):
            m.setdefault(r["ma_tp"].lstrip("0"), []).append(r)
    except FileNotFoundError:
        pass
    return m


def pick_serving(ma: str, cat: str, vdd: dict) -> tuple[float, str, bool]:
    """Trả (gram, desc, is_default)."""
    for r in vdd.get(ma.lstrip("0"), []):
        try:
            g = float(r["gram_chin"])
        except (ValueError, TypeError):
            continue
        if 20 <= g <= 600 and WHOLE_UNIT.search(r["don_vi"] or ""):
            return g, r["don_vi"], False
    for r in vdd.get(ma.lstrip("0"), []):
        try:
            g = float(r["gram_chin"])
        except (ValueError, TypeError):
            continue
        if 20 <= g <= 600:
            return g, r["don_vi"], False
    g, d = DEFAULT_SRV.get(cat, (100, "1 phần"))
    return float(g), d, True


def main() -> None:
    vn = load_vn()
    vdd = load_vdd()
    have = set()
    with open(SPEC, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            have.add(r["food_id"])
    review = list(csv.DictReader(open(INTERIM / "REVIEW.csv", encoding="utf-8-sig")))
    new_rows, skipped = [], []
    for r in review:
        fid = (r["food_id"] or "").strip()
        if not fid or fid in have:
            continue
        prefix = fid[:3]
        cat = CAT.get(prefix)
        if not cat:
            skipped.append((fid, "prefix lạ"))
            continue
        macro_src = (r["macro_src"] or "").strip()
        m = re.match(r"vn:(\d+)", macro_src)
        en, vn_row = "", None
        vi_name = r["vi"]
        if m:
            vn_row = vn.get(m.group(1))
            if not vn_row:
                skipped.append((fid, f"không thấy mã VN {m.group(1)}"))
                continue
            en = vn_row["ten_en"] or vn_row["ten_vi"]
            vi_name = vn_row["ten_vi"]  # tên VN FCT là source of truth
        elif macro_src.startswith("manual:"):
            en = r["vi"]
        elif macro_src.startswith("usda:"):
            en = macro_src[5:]
        else:
            skipped.append((fid, f"macro_src lạ {macro_src[:30]}"))
            continue
        if m:
            serving_g, serving_desc, is_default = pick_serving(m.group(1), cat, vdd)
        else:
            serving_g, serving_desc = DEFAULT_SRV.get(cat, (100, "1 phần"))
            is_default = True
        gi = (r["gi"] or "").strip()
        tier = (r["tier"] or "").strip() or ("na" if not gi else "proxy")
        gi_ref = {"table1": "Sydney Int.Tables 2021", "table2": "Sydney Int.Tables 2021",
                  "proxy": "ước tính theo nhóm (cần duyệt)",
                  "estimated": "ước tính (cần duyệt)",
                  "na": "không áp dụng"}.get(tier, "REVIEW user")
        notes = (r["reason"] or "").strip() or "Theo dõi đường huyết cá nhân."
        new_rows.append({
            "food_id": fid, "vi": vi_name, "en": en, "aliases": "",
            "category": cat, "serving_g": serving_g, "serving_desc": serving_desc,
            "macro_src": macro_src, "gi": gi, "gi_tier": tier, "gi_ref": gi_ref,
            "notes": notes, "is_local": "true",
            "review": "true" if (is_default or tier in ("proxy", "estimated")) else "false",
        })
    with open(SPEC, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["food_id", "vi", "en", "aliases", "category",
                                          "serving_g", "serving_desc", "macro_src", "gi",
                                          "gi_tier", "gi_ref", "notes", "is_local", "review"])
        w.writerows(new_rows)
    log.info("appended=%d skipped=%d", len(new_rows), len(skipped))
    for s in skipped[:20]:
        log.warning("SKIP %s: %s", s[0], s[1])


if __name__ == "__main__":
    main()
