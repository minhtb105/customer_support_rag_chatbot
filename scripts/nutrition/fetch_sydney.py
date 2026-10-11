"""
Tải Sydney GI 2021 supplementary tables (mirror công khai) + trích text thô.

Output:
  data/raw/nutrition/_interim/sydney_table1.pdf / sydney_table2.pdf
  data/raw/nutrition/_interim/sydney_raw.json  ({table1: text, table2: text})
Parse số ở bước sau (layout bảng).
Chạy:  python scripts/nutrition/fetch_sydney.py [--skip-download]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

URLS = {
    # Table 1: ISO-consistent (dùng chính)
    "sydney_table1.pdf": "https://nutritotal.com.br/pro/wp-content/uploads/sites/3/2021/09/Tabela1_IG.pdf",
    # Table 2: method deviations / wide variability (dùng phụ, gi_tier=table2)
    "sydney_table2.pdf": "https://nutritotal.com.br/pro/wp-content/uploads/2021/09/Tabela2_IG.pdf",
}
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"


def download() -> None:
    import requests
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        p = OUT / name
        if p.exists() and p.stat().st_size > 500_000:
            log.info("%s đã có: %.1f MB", name, p.stat().st_size / 1e6)
            continue
        log.info("Downloading %s", name)
        with requests.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        log.info("Xong %s: %.1f MB", name, p.stat().st_size / 1e6)


def extract() -> dict:
    import fitz
    out = {}
    for name in URLS:
        doc = fitz.open(OUT / name)
        out[name] = "\n".join(page.get_text() for page in doc)
        log.info("%s: %d trang, %d ký tự", name, len(doc), len(out[name]))
    (OUT / "sydney_raw.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    if "--skip-download" not in sys.argv:
        download()
    extract()


if __name__ == "__main__":
    main()
