"""Shared download + PDF text for nutrition fetchers (httpx, already a dep)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nutrition_fetch")

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"


def download(url: str, dest: str | Path, min_size: int = 500_000) -> Path:
    """GET url -> dest (skip if exists and bigger than min_size)."""
    import httpx

    dest = Path(dest)
    OUT.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_size:
        log.info("%s da co: %.1f MB", dest.name, dest.stat().st_size / 1e6)
        return dest
    log.info("Downloading %s", url)
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    log.info("Xong %s: %.1f MB", dest.name, dest.stat().st_size / 1e6)
    return dest


def pdf_text(pdf: str | Path) -> str:
    """Plain text of a PDF (fitz)."""
    import fitz

    doc = fitz.open(pdf)
    return "\n".join(page.get_text() for page in doc)


def get_json(url: str, headers: dict | None = None, timeout: int = 60) -> dict:
    import httpx

    r = httpx.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 60) -> dict:
    import httpx

    r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_json(path: str | Path, obj: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path
