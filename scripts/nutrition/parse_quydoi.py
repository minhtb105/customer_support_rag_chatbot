r"""
Parse bảng chuyển đổi VDD -> CSV (ma_tp, ten, don_vi, gram_chin).

Heuristic: dòng đơn vị (cột đầu có chữ bát/quả/miếng/muỗng/cốc/gói/củ/
quả/mớ/lon/chai/thìa/viên/múi/khoanh...) + số đầu tiên sau đó = trọng
lượng chín của đơn vị đó. Tên món + mã lấy từ dòng 'Mã \d+' gần nhất trước đó.
Chạy:  python scripts/nutrition/parse_quydoi.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _parse_io import load_json, log, save_csv

UNIT_RE = re.compile(r"bát|quả|miếng|muỗng|thìa|cốc|gói|củ|mớ|lon|chai|viên|múi|khoanh|chén|đĩa|bẹ|nhánh|tai|con|khúc|lát|thanh|cây|hộp|túi|bịch|gắp|nắm|bó|quầy|trái", re.I)
MA_RE = re.compile(r"Mã\s*(\d{4,5})", re.I)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def clean(s) -> str:
    return " ".join((s or "").split())


def main() -> None:
    tabs = load_json("quydoi_tables.json")
    rows = []
    for key in sorted(tabs):
        grid = tabs[key]
        cur_ma, cur_ten = "", ""
        for r in grid:
            cells = [clean(c) for c in r]
            line = " | ".join(c for c in cells if c)
            m = MA_RE.search(line)
            if m and len(cells[0]) < 40:
                # dòng tiêu đề món: tên thường ở cell đầu
                cur_ma = m.group(1)
                cur_ten = cells[0].replace("Mã " + cur_ma, "").strip()
                continue
            if cur_ma and cells[0] and UNIT_RE.search(cells[0]):
                nums = NUM_RE.findall(" ".join(cells[1:]))
                if nums:
                    try:
                        g = float(nums[0].replace(",", "."))
                    except ValueError:
                        continue
                    if 5 <= g <= 2000:
                        rows.append({"ma_tp": cur_ma, "ten": cur_ten,
                                     "don_vi": cells[0][:60], "gram_chin": g,
                                     "table": key})
    save_csv("quydoi_servings.csv", rows, ["ma_tp", "ten", "don_vi", "gram_chin", "table"])
    log.info("parsed %d serving rows", len(rows))


if __name__ == "__main__":
    main()
