"""
Tải Bảng thành phần thực phẩm VN (FAO mirror) + trích text thô từng trang.

Output:
  data/raw/nutrition/_interim/vn_fct.pdf
  data/raw/nutrition/_interim/vn_fct_pages.json  (page -> text, để bước map dùng)

Không parse số ở đây (layout bảng phức tạp) — parse ở map step với food list chốt.
Chạy:  python scripts/nutrition/fetch_vn_fct.py [--skip-download]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fetch import OUT, download as _download, log, save_json
from decode_vni import decode_span

URL = "https://www.fao.org/fileadmin/templates/food_composition/documents/pdf/VTN_FCT_2007.pdf"
PDF = OUT / "vn_fct.pdf"
PAGES_JSON = OUT / "vn_fct_pages.json"


def download() -> Path:
    return _download(URL, PDF, min_size=1_000_000)


def extract() -> dict:
    import fitz
    doc = fitz.open(PDF)
    pages = {}
    for i, page in enumerate(doc):
        parts = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    parts.append(decode_span(s["text"], s["font"]))
        pages[str(i)] = "\n".join(parts)
    save_json(PAGES_JSON, pages)
    log.info("Trích %d trang (decode theo font) -> %s", len(pages), PAGES_JSON)
    return pages


def main() -> None:
    if "--skip-download" not in sys.argv:
        download()
    pages = extract()
    # in mẫu 1 trang thực phẩm để kiểm tra layout
    for idx, text in pages.items():
        if "Glucid" in text or "glucid" in text:
            print(f"--- page {idx} head ---")
            print(text[:1200])
            break


if __name__ == "__main__":
    main()
