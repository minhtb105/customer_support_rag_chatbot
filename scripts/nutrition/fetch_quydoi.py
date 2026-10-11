"""
Trích bảng chuyển đổi trọng lượng VDD (trang 28-56) -> CSV trung gian.

Thử fitz.find_tables() trước, fallback text thô để parse tay.
Chạy:  python scripts/nutrition/fetch_quydoi.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"
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
    (OUT / "quydoi_tables.json").write_text(
        json.dumps(tabs, ensure_ascii=False), encoding="utf-8")
    log.info("saved %d tables", len(tabs))


if __name__ == "__main__":
    main()
