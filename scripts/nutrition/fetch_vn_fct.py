"""
Tải Bảng thành phần thực phẩm VN (FAO mirror) + trích text thô từng trang.

Output:
  data/raw/nutrition/_interim/vn_fct.pdf
  data/raw/nutrition/_interim/vn_fct_pages.json  (page -> text, để bước map dùng)

Không parse số ở đây (layout bảng phức tạp) — parse ở map step với food list chốt.
Chạy:  python scripts/nutrition/fetch_vn_fct.py [--skip-download]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

URL = "https://www.fao.org/fileadmin/templates/food_composition/documents/pdf/VTN_FCT_2007.pdf"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"
PDF = OUT / "vn_fct.pdf"
PAGES_JSON = OUT / "vn_fct_pages.json"


def download() -> Path:
    import requests
    OUT.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size > 1_000_000:
        log.info("PDF đã có: %.1f MB", PDF.stat().st_size / 1e6)
        return PDF
    log.info("Downloading %s", URL)
    with requests.get(URL, timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(PDF, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    log.info("Xong: %.1f MB", PDF.stat().st_size / 1e6)
    return PDF


def extract() -> dict:
    import fitz
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from decode_vni import decode_span
    doc = fitz.open(PDF)
    pages = {}
    for i, page in enumerate(doc):
        parts = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    parts.append(decode_span(s["text"], s["font"]))
        pages[str(i)] = "\n".join(parts)
    PAGES_JSON.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
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
