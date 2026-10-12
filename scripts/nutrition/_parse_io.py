"""Shared logging + INTERIM + json/csv-utf8-sig for nutrition parsers."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nutrition_parse")

INTERIM = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"


def load_json(name: str) -> dict:
    return json.loads((INTERIM / name).read_text(encoding="utf-8"))


def save_csv(name: str, rows: list[dict], fieldnames: list[str]) -> Path:
    path = INTERIM / name
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path
