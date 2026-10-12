"""
Trích bảng chuyển đổi trọng lượng VDD (trang 28-56) -> CSV trung gian.

Thử fitz.find_tables() trước, fallback text thô để parse tay.
Chạy:  python scripts/nutrition/fetch_quydoi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fetch import OUT, log, save_json

PDF = OUT / "vdd_quy_doi.pdf"


def main() -> None:
    import fitz
    doc = fitz.open(PDF)
    tabs = {}
    for i in range(28, 57):
        try:
            found = list(doc[i].find_tables())
        except Exception:
            found = []
        log.info("p%d: %d tables", i, len(found))
        for ti, t in enumerate(found):
            try:
                rows = t.extract()
                tabs[f"p{i}_t{ti}"] = rows
                log.info("  t%d: %d rows x %d cols, head=%s",
                         ti, len(rows), len(rows[0]) if rows else 0,
                         str(rows[0])[:120] if rows else None)
            except Exception as e:  # noqa: BLE001
                log.warning("  t%d extract fail: %s", ti, str(e)[:100])
    save_json(OUT / "quydoi_tables.json", tabs)
    log.info("saved %d tables", len(tabs))


if __name__ == "__main__":
    main()
