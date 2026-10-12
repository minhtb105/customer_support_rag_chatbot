"""
Tải Sydney GI 2021 supplementary tables (mirror công khai) + trích text thô.

Output:
  data/raw/nutrition/_interim/sydney_table1.pdf / sydney_table2.pdf
  data/raw/nutrition/_interim/sydney_raw.json  ({table1: text, table2: text})
Parse số ở bước sau (layout bảng).
Chạy:  python scripts/nutrition/fetch_sydney.py [--skip-download]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fetch import OUT, download as _download, log, pdf_text, save_json

URLS = {
    # Table 1: ISO-consistent (dùng chính)
    "sydney_table1.pdf": "https://nutritotal.com.br/pro/wp-content/uploads/sites/3/2021/09/Tabela1_IG.pdf",
    # Table 2: method deviations / wide variability (dùng phụ, gi_tier=table2)
    "sydney_table2.pdf": "https://nutritotal.com.br/pro/wp-content/uploads/2021/09/Tabela2_IG.pdf",
}


def download() -> None:
    for name, url in URLS.items():
        _download(url, OUT / name)


def extract() -> dict:
    out = {}
    for name in URLS:
        out[name] = pdf_text(OUT / name)
        log.info("%s: %d ký tự", name, len(out[name]))
    save_json(OUT / "sydney_raw.json", out)
    return out


def main() -> None:
    if "--skip-download" not in sys.argv:
        download()
    extract()


if __name__ == "__main__":
    main()
