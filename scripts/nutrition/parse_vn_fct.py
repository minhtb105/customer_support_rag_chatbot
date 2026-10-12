"""
Parse trang món ăn VN FCT (đã decode theo font) -> CSV một dòng/món.

Cột: ma_so, ten_vi, ten_en, thai_bo_pct, energy_kcal, protein_g, lipid_g,
     glucid_g, fiber_g, sugars_g (Đường tổng số; '-' -> rỗng)
Trang món ăn có "Mã số:" + "Tên thực phẩm (Vietnamese):".
Chạy:  python scripts/nutrition/parse_vn_fct.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _parse_io import INTERIM, load_json, log, save_csv

OUT_CSV = "vn_fct_macros.csv"

WANT = {
    "energy_kcal": r"Năng lượng\s*\(\s*Energy\s*\)\s*KCal\s*([0-9.\-]+)",
    "protein_g": r"\nProtein\s*g\s*([0-9.\-]+)",
    "lipid_g": r"Lipid\s*\(Fat\)\s*g\s*([0-9.\-]+)",
    "glucid_g": r"Glucid\s*\(Carbohydrate\)\s*g\s*([0-9.\-]+)",
    "fiber_g": r"Celluloza\s*\(Fiber\)\s*g\s*([0-9.\-]+)",
    "sugars_g": r"Đường tổng số\s*\(Sugar\)\s*g\s*([0-9.\-]+)",
}


def num(s: str | None) -> float | None:
    if not s or s.strip() == "-":
        return None
    try:
        return float(s.strip())
    except ValueError:
        return None


def parse_page(text: str) -> dict | None:
    if "Mã số:" not in text or "Tên thực phẩm" not in text:
        return None
    m = re.search(r"Mã số:\s*(\d+)", text)
    v = re.search(r"Tên thực phẩm\s*\(Vietnamese\):\s*\n(.+)", text)
    e = re.search(r"Tên tiếng Anh\s*\(English\):\s*\n(.+)", text)
    w = re.search(r"Thải bỏ\s*\(%\):\s*([0-9.\-]+)", text)
    if not (m and v):
        return None
    ten_vi = v.group(1).strip()
    ten_en = (e.group(1).strip() if e else "")
    if len(m.group(1)) > 5 or "Tên" in ten_vi or "English" in ten_vi:
        # layout nhóm thịt/thủy sản: tên nằm đầu trang (VI), EN sau dòng STT
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        ten_vi = lines[0] if lines else ten_vi
        ten_en = ""
        for i, ln in enumerate(lines[:6]):
            if re.fullmatch(r"\d{1,4}", ln) and i + 1 < len(lines):
                ten_en = lines[i + 1]
                break
        if "Tên" in ten_vi or "English" in ten_vi or len(ten_vi) > 60:
            return None  # dính trang mục lục/header
    row = {
        "ma_so": m.group(1).strip(),
        "ten_vi": ten_vi,
        "ten_en": ten_en or ((e.group(1).strip() if e else "")),
        "thai_bo_pct": num(w.group(1)) if w else None,
    }
    for key, pat in WANT.items():
        mm = re.search(pat, text)
        row[key] = num(mm.group(1)) if mm else None
    return row


def main() -> None:
    pages = load_json("vn_fct_pages.json")
    rows = []
    for idx in sorted(pages, key=int):
        r = parse_page(pages[idx])
        if r:
            r["page"] = idx
            rows.append(r)
    save_csv(OUT_CSV, rows, ["page", "ma_so", "ten_vi", "ten_en",
                             "thai_bo_pct", "energy_kcal", "protein_g",
                             "lipid_g", "glucid_g", "fiber_g", "sugars_g"])
    n_sugar = sum(1 for r in rows if r["sugars_g"] is not None)
    log.info("Parsed %d món -> %s (có sugars: %d)", len(rows), INTERIM / OUT_CSV, n_sugar)


if __name__ == "__main__":
    main()
