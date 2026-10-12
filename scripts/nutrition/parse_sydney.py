"""
Parse Sydney GI supplementary tables -> CSV (ref_id, food_en, country, year, gi, gl, tier).

Mỗi dòng thực phẩm: ID / mô tả (nhiều dòng) / Country / Year / GI±SEM / GL / ...
Chỉ lấy 6 field cần cho join món VN. Nhiễu parse chấp nhận được — bước build
match thủ công + review.
Chạy:  python scripts/nutrition/parse_sydney.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _parse_io import load_json, log, save_csv

PAT = re.compile(
    r"(?m)^(\d{1,4}) \n(.+?) \n([A-Za-z][A-Za-z .'\-()]*?) \n(\d{4}\*?) \n(\d+)±(\d+)? \n(\d+) \n",
    re.S,
)


JUNK_TOKENS = ("Glucose,", "HemoCue", "Enzymatic", "Capillary,",
                 "Venous", "plasma", "whole blood", "Standard", "Normal,",
                 "Bread, 2h", "White bread", "NS ", "UO", "YSI", "HemoCue")


TRIPLE = re.compile(r"(?m)^([A-Za-z][A-Za-z .'\-()]*?) \n(\d{4}\*?) \n(\d{1,3})±\S* \n(\d{1,3}) \n")


def parse(text: str, tier: str) -> list[dict]:
    """Neo từ neo GI: country/year/GI/GL rồi đi ngược lấy mô tả món."""
    lines = text.split(" \n")
    rows = []
    for m in TRIPLE.finditer(text):
        country, year, gi, gl = m.group(1), m.group(2), m.group(3), m.group(4)
        start = text.count(" \n", 0, m.start(1))
        if not (0 < int(gi) <= 150 and 0 <= int(gl) <= 80):
            continue
        if len(country.strip()) > 40 or len(country.strip()) < 2:
            continue
        # đi ngược tối đa 5 dòng gom mô tả, dừng ở dòng số thuần (ref_id/refcode)
        desc_parts = []
        ref_id = ""
        for j in range(start - 1, max(-1, start - 6), -1):
            ln = lines[j].strip() if 0 <= j < len(lines) else ""
            if re.fullmatch(r"\d{1,4}", ln or ""):
                ref_id = ln
                break
            if ln:
                desc_parts.append(ln)
        desc = " ".join(reversed(desc_parts))
        desc = " ".join(desc.split())
        if len(desc) < 3 or len(desc) > 300:
            continue
        if any(t in desc for t in ("Glucose,", "HemoCue", "Enzymatic", "Capillary,",
                                   "Venous", "whole blood", "Standard", "Normal,")):
            continue
        rows.append({"ref_id": ref_id, "food_en": desc, "country": country.strip(),
                     "year": year, "gi": int(gi), "gl": int(gl), "tier": tier})
    return rows


def main() -> None:
    raw = load_json("sydney_raw.json")
    all_rows = parse(raw["sydney_table1.pdf"], "table1") + parse(
        raw["sydney_table2.pdf"], "table2")
    save_csv("sydney_gi.csv", all_rows, ["ref_id", "food_en", "country", "year",
                                         "gi", "gl", "tier"])
    t1 = sum(1 for r in all_rows if r["tier"] == "table1")
    log.info("Parsed %d rows (table1=%d, table2=%d)", len(all_rows), t1, len(all_rows) - t1)


if __name__ == "__main__":
    main()
