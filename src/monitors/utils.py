"""Shared utils for monitors — SSRF allowlist, hashing, HTTP"""
from __future__ import annotations
import hashlib
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

try:
    from src.shared.config import ALLOWED_MONITOR_DOMAINS
except ImportError:
    from shared.config import ALLOWED_MONITOR_DOMAINS  # type: ignore

# ---------- SSRF ----------
def is_url_allowed(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()
        # exact or subdomain allowed
        for allowed in ALLOWED_MONITOR_DOMAINS:
            if host == allowed or host.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False

def assert_url_allowed(url: str):
    if not is_url_allowed(url):
        raise ValueError(f"URL not in allowlist: {url}")

PRIVATE_IP_RE = re.compile(r"^(127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.)")

def is_private_host(host: str) -> bool:
    return bool(PRIVATE_IP_RE.match(host))

# ---------- hashing ----------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------- HTTP with retry ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GuidelineMonitor/1.0 WHO-RAG",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}

def head_with_etag(url: str, timeout: int = 15) -> dict:
    """HEAD request to get etag/last-modified without downloading body."""
    assert_url_allowed(url)
    try:
        r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return {
            "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
            "content_length": r.headers.get("Content-Length"),
            "content_type": r.headers.get("Content-Type"),
            "final_url": r.url,
            "status": r.status_code,
        }
    except Exception as e:
        # fallback: try GET with stream then abort
        return {"error": str(e), "etag": None, "last_modified": None}

def download_file_safe(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download with allowlist check and PDF magic validation."""
    assert_url_allowed(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            if downloaded < 1024:
                tmp.unlink(missing_ok=True)
                return False
            # magic check for PDF
            if dest.suffix.lower() == ".pdf":
                with open(tmp, "rb") as f:
                    magic = f.read(4)
                if magic != b"%PDF":
                    with open(tmp, "rb") as fh:
                        head = fh.read(300).decode(errors="ignore").lower()
                    if "<html" in head or "<!doctype" in head:
                        tmp.unlink(missing_ok=True)
                        return False
            tmp.replace(dest)
            return True
    except Exception as e:
        print(f"[download] failed {url}: {e}")
        return False

# ---------- FDA cache ----------
_FDA_CACHE: dict = {}
_FDA_CACHE_TS: dict = {}

def cached_get(url: str, ttl_seconds: int = 3600, timeout: int = 20, retries: int = 3) -> dict:
    """In-memory cache for FDA openFDA anonymous calls (rate-limit 240/min) với 429 backoff."""
    assert_url_allowed(url)
    now = time.time()
    if url in _FDA_CACHE and (now - _FDA_CACHE_TS.get(url, 0)) < ttl_seconds:
        return _FDA_CACHE[url]
    # polite delay if needed
    time.sleep(0.25)  # ~240/min max
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after else (2 * (attempt + 1))
                except Exception:
                    wait = 2 * (attempt + 1)
                # exponential backoff capped 30s
                wait = min(wait * (2 ** attempt), 30)
                print(f"[FDA] 429 Too Many Requests, backoff {wait}s (attempt {attempt+1}/{retries+1})")
                time.sleep(wait)
                last_err = RuntimeError(f"429 Too Many Requests, Retry-After {retry_after}")
                continue
            r.raise_for_status()
            j = r.json()
            _FDA_CACHE[url] = j
            _FDA_CACHE_TS[url] = time.time()
            return j
        except Exception as e:
            last_err = e
            if attempt < retries and "429" not in str(e):
                # For non-429 transient errors, brief backoff
                time.sleep(1 * (attempt + 1))
                continue
            if "429" in str(e) and attempt < retries:
                continue
            raise last_err  # type: ignore
    raise last_err  # type: ignore

def extract_text_from_pdf(pdf_path: Path, max_chars: int = 20000) -> str:
    """Extract text via docling or fallback pypdf."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        doc = converter.convert(str(pdf_path)).document
        # try chunk_strategies helper
        try:
            from src.shared.chunk_strategies import extract_full_text_from_doc
            txt = extract_full_text_from_doc(doc)
        except ImportError:
            from shared.chunk_strategies import extract_full_text_from_doc  # type: ignore
            txt = extract_full_text_from_doc(doc)
        return txt[:max_chars]
    except Exception:
        # fallback pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            txt = ""
            for page in reader.pages[:20]:
                try:
                    txt += (page.extract_text() or "") + "\n"
                except Exception:
                    continue
                if len(txt) >= max_chars:
                    break
            return txt[:max_chars]
        except Exception as e:
            return f"[extract_failed: {e}]"
