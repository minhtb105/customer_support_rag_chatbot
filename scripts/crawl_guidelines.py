"""
Guideline Data Pipeline — WHO IRIS, ADA, AHA/ACC, GOLD, GINA, mhGAP, BYT

Crawler tải PDF guideline từ các nguồn chính thống về data/raw/pdfs/<source>/.
Mỗi nguồn có manifest riêng; pipeline hỗ trợ:
  - crawl --source who_iris --limit 5
  - crawl --source byt --query "đái tháo đường"
  - crawl --all --limit 3
  - download --manifest data/guideline_manifest.json
  - status

Yêu cầu: requests, beautifulsoup4, lxml đã có trong pyproject.toml.
BYT Quyết định / Thông tư: crawl từ moh.gov.vn + thuvienphapluat.vn (public).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_ROOT = BASE_DIR / "data" / "raw" / "pdfs"
MANIFEST_PATH = BASE_DIR / "data" / "guideline_manifest.json"
METADATA_PATH = BASE_DIR / "data" / "guideline_metadata.json"

# Nguồn → thư mục
SOURCE_DIRS = {
    "who_iris": PDF_ROOT / "who_iris",
    "ada": PDF_ROOT / "ada",
    "aha_acc": PDF_ROOT / "aha_acc",
    "gold": PDF_ROOT / "gold",
    "gina": PDF_ROOT / "gina",
    "mhgap": PDF_ROOT / "mhgap",
    "byt": PDF_ROOT / "byt",
    "diabetes": PDF_ROOT / "diabetes",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 GuidelineCrawler/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}

# ------------------------------------------------------------
# Curated seed URLs — public, stable, không cần auth
# ------------------------------------------------------------
# Mỗi entry: (source, title, url, year, tags)
CURATED_SEEDS: List[dict] = [
    # --- WHO IRIS — Diabetes & NCD ---
    {
        "source": "who_iris",
        "title": "WHO Guideline on Diagnosis and Management of Type 2 Diabetes (HEARTS-D)",
        "url": "https://iris.who.int/handle/10665/331710",
        "year": 2020,
        "tags": ["diabetes", "ncd", "hearts"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/331710/9789240001367-eng.pdf",
    },
    {
        "source": "who_iris",
        "title": "WHO Package of Essential NCD Interventions (PEN) 2020",
        "url": "https://iris.who.int/handle/10665/334186",
        "year": 2020,
        "tags": ["ncd", "pen", "phc"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/334186/9789240009226-eng.pdf",
    },
    {
        "source": "who_iris",
        "title": "WHO Guidelines on Physical Activity and Sedentary Behaviour 2020",
        "url": "https://iris.who.int/handle/10665/336656",
        "year": 2020,
        "tags": ["physical_activity"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/336656/9789240015128-eng.pdf",
    },
    {
        "source": "who_iris",
        "title": "WHO mhGAP Intervention Guide v2.0",
        "url": "https://iris.who.int/handle/10665/250239",
        "year": 2016,
        "tags": ["mhgap", "mental_health"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/250239/9789241549790-eng.pdf",
    },
    # --- ADA ---
    {
        "source": "ada",
        "title": "ADA Standards of Care in Diabetes 2024 (Abridged PDF)",
        "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        "year": 2024,
        "tags": ["diabetes", "ada_soc"],
        "direct_pdf": None,  # cần crawl hoặc dùng link public thay thế
        "alt_pdf": "https://diabetesjournals.org/care/article/47/Supplement_1/S1/153954/Standards-of-Care-in-Diabetes-2024-Abridged-for",
    },
    # --- GOLD / GINA ---
    {
        "source": "gold",
        "title": "GOLD Global Strategy for COPD 2024",
        "url": "https://goldcopd.org/2024-gold-report/",
        "year": 2024,
        "tags": ["copd", "gold"],
        "direct_pdf": "https://goldcopd.org/wp-content/uploads/2024/02/GOLD-2024_v1.2-11Jan24_WMV.pdf",
    },
    {
        "source": "gina",
        "title": "GINA Global Strategy for Asthma 2024",
        "url": "https://ginasthma.org/2024-gina-main-report/",
        "year": 2024,
        "tags": ["asthma", "gina"],
        "direct_pdf": "https://ginasthma.org/wp-content/uploads/2024/05/GINA-2024-Strategy-Report-24_05_22_WMS.pdf",
    },
    # --- AHA/ACC ---
    {
        "source": "aha_acc",
        "title": "AHA/ACC Guideline on Chronic Coronary Disease 2023",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001168",
        "year": 2023,
        "tags": ["cardiology", "aha_acc"],
        "direct_pdf": None,
    },
    # --- BYT (Quyết định / Hướng dẫn) ---
    {
        "source": "byt",
        "title": "BYT — Hướng dẫn chẩn đoán và điều trị Đái tháo đường type 2 (QĐ 3319/QĐ-BYT 2017, cập nhật 2020)",
        "url": "https://kcb.vn/van-ban-quy-pham-phap-luat/",
        "year": 2020,
        "tags": ["diabetes", "byt", "vietnam"],
        "direct_pdf": None,
        "note": "Cần crawl từ moh.gov.vn hoặc thuvienphapluat.vn; nếu không truy cập được, giữ chỗ để bổ sung thủ công.",
    },
    # --- Diabetes curated (3 file public yêu cầu) ---
    {
        "source": "diabetes",
        "title": "WHO Classification of Diabetes Mellitus 2019",
        "url": "https://iris.who.int/handle/10665/325182",
        "year": 2019,
        "tags": ["diabetes", "who"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/325182/9789241515702-eng.pdf",
    },
    {
        "source": "diabetes",
        "title": "WHO HEARTS-D Diagnosis and Management of Type 2 Diabetes",
        "url": "https://iris.who.int/handle/10665/331710",
        "year": 2020,
        "tags": ["diabetes", "who", "hearts-d"],
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/331710/9789240001367-eng.pdf",
    },
    {
        "source": "diabetes",
        "title": "ADA Standards of Care in Diabetes — 2024 Abridged (ADA)",
        "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        "year": 2024,
        "tags": ["diabetes", "ada"],
        "direct_pdf": None,
        "alt_pdf": "https://ada-journals.cld.bz/Standards-of-Care-in-Diabetes-2024-Abridged-for-Primary-Care-Providers",
    },
]

# Fallback PDFs khi direct_pdf = None — các bản public mirror ổn định
FALLBACK_PDFS = {
    "ada_soc_2024": "https://www.ncbi.nlm.nih.gov/books/NBK584707/pdf/Bookshelf_NBK584707.pdf",  # placeholder, thực tế sẽ thử alt
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dirs():
    for d in SOURCE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text.strip())
    return text[:max_len].strip("_").lower()


def download_file(url: str, dest: Path, timeout: int = 60, retries: int = 2) -> bool:
    """Tải file với retry, kiểm tra Content-Type và kích thước tối thiểu."""
    for attempt in range(retries + 1):
        try:
            log.info(f"  -> GET {url} (attempt {attempt+1}/{retries+1})")
            with requests.get(url, headers=HEADERS, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                # chấp nhận pdf hoặc octet-stream; nếu là html thì có thể là landing page
                if "html" in ctype.lower() and "pdf" not in url.lower():
                    log.warning(f"     Content-Type html ({ctype}), có thể không phải PDF trực tiếp: {url}")
                total = int(r.headers.get("Content-Length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                downloaded = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                if downloaded < 1024:
                    log.warning(f"     File quá nhỏ ({downloaded} bytes), bỏ qua: {dest.name}")
                    tmp.unlink(missing_ok=True)
                    return False
                # kiểm tra magic bytes PDF
                with open(tmp, "rb") as f:
                    magic = f.read(4)
                if magic != b"%PDF" and dest.suffix.lower() == ".pdf":
                    # có thể là html trả về
                    with open(tmp, "rb") as f:
                        head = f.read(200).decode(errors="ignore").lower()
                    if "<html" in head or "<!doctype" in head:
                        log.warning(f"     Nhận HTML thay vì PDF: {url}")
                        tmp.unlink(missing_ok=True)
                        return False
                tmp.replace(dest)
                log.info(f"  [OK] {dest.name} ({downloaded/1024:.1f} KB)")
                return True
        except Exception as e:
            log.warning(f"  [FAIL] {url}: {e}")
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return False


def resolve_iris_handle_to_content_url(handle_url: str) -> Optional[str]:
    """
    Resolve iris.who.int/handle/10665/XXXX -> DSpace 7 bitstream content API.
    Handles là dạng https://iris.who.int/handle/10665/325182
    Trả về https://iris.who.int/server/api/core/bitstreams/<uuid>/content
    """
    try:
        # extract handle id
        m = re.search(r"10665/\d+", handle_url)
        if not m:
            return None
        handle = m.group(0)
        # 1) discover item via search
        search_url = "https://iris.who.int/server/api/discover/search/objects"
        r = requests.get(search_url, params={"query": handle}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        j = r.json()
        objs = j.get("_embedded", {}).get("searchResult", {}).get("_embedded", {}).get("objects", [])
        item_id = None
        for o in objs:
            idx = o.get("_embedded", {}).get("indexableObject", {})
            if idx.get("handle") == handle:
                item_id = idx.get("id") or idx.get("uuid")
                break
        if not item_id and objs:
            # fallback lấy object đầu tiên
            item_id = objs[0].get("_embedded", {}).get("indexableObject", {}).get("id")
        if not item_id:
            log.warning(f"    IRIS resolve: không tìm thấy item cho {handle}")
            return None
        # 2) bundles -> ORIGINAL
        r2 = requests.get(f"https://iris.who.int/server/api/core/items/{item_id}/bundles", headers=HEADERS, timeout=20)
        r2.raise_for_status()
        j2 = r2.json()
        bundle_id = None
        for b in j2.get("_embedded", {}).get("bundles", []):
            if b.get("name") == "ORIGINAL":
                bundle_id = b.get("uuid") or b.get("id")
                break
        if not bundle_id:
            log.warning(f"    IRIS resolve: không tìm thấy ORIGINAL bundle cho {handle}")
            return None
        # 3) bitstreams -> chọn PDF lớn nhất (thường là eng)
        r3 = requests.get(f"https://iris.who.int/server/api/core/bundles/{bundle_id}/bitstreams", headers=HEADERS, timeout=20)
        r3.raise_for_status()
        j3 = r3.json()
        best = None
        best_size = -1
        for bs in j3.get("_embedded", {}).get("bitstreams", []):
            name = bs.get("name", "").lower()
            size = bs.get("sizeBytes", 0) or 0
            # ưu tiên eng pdf, bỏ qua thumbnail
            if name.endswith(".pdf") and size > best_size:
                # ưu tiên eng
                if "eng" in name or best is None:
                    best = bs
                    best_size = size
                elif best and "eng" not in best.get("name","").lower():
                    best = bs
                    best_size = size
        if best:
            content_href = best.get("_links", {}).get("content", {}).get("href")
            if content_href:
                log.info(f"    IRIS resolved {handle} -> {content_href} ({best.get('name')}, {best_size/1024:.0f} KB)")
                return content_href
        return None
    except Exception as e:
        log.warning(f"    IRIS resolve failed {handle_url}: {e}")
        return None


def discover_pdfs_from_iris_handle(handle_url: str) -> List[str]:
    """Wrapper cũ — giờ dùng API resolver, giữ interface list."""
    url = resolve_iris_handle_to_content_url(handle_url)
    return [url] if url else []


def crawl_who_iris_search(query: str, limit: int = 5) -> List[dict]:
    """Crawl WHO IRIS search API (public)."""
    # WHO IRIS dùng DSpace REST; search đơn giản qua handle discovery
    # Fallback: dùng curated seeds nếu search không khả dụng
    log.info(f"[WHO IRIS] search query='{query}' limit={limit}")
    # Thử search endpoint
    search_url = "https://iris.who.int/rest/search"
    # WHO IRIS REST có thể yêu cầu khác; nếu fail thì trả về curated
    try:
        params = {"query": query, "limit": limit}
        r = requests.get(search_url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            results = []
            # parse tuỳ format
            items = data.get("items") or data.get("results") or []
            for it in items[:limit]:
                handle = it.get("handle") or it.get("link")
                title = it.get("title") or it.get("name") or "WHO IRIS result"
                if handle:
                    url = handle if handle.startswith("http") else f"https://iris.who.int/handle/{handle}"
                    results.append({
                        "source": "who_iris",
                        "title": title,
                        "url": url,
                        "year": it.get("year") or 2024,
                        "tags": [query],
                        "direct_pdf": None,
                    })
            if results:
                return results
    except Exception as e:
        log.info(f"  WHO IRIS search fallback: {e}")
    # fallback curated
    q = query.lower()
    return [s for s in CURATED_SEEDS if s["source"] == "who_iris" and any(q in t for t in s.get("tags", []))][:limit]


def crawl_byt_search(query: str, limit: int = 5) -> List[dict]:
    """Crawl BYT — placeholders; thực tế cần crawl moh.gov.vn."""
    log.info(f"[BYT] search query='{query}' limit={limit} — using curated seeds (moh.gov.vn crawl cần bổ sung)")
    q = query.lower()
    # Lọc curated BYT
    candidates = [s for s in CURATED_SEEDS if s["source"] == "byt"]
    # Nếu query diabetes, trả về diabetes-related
    if "ái tháo" in q or "diabetes" in q or "đái" in q:
        return candidates[:limit]
    return candidates[:limit]


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------
@dataclass
class ManifestEntry:
    source: str
    title: str
    url: str
    year: Optional[int]
    tags: List[str]
    filename: str
    dest_path: str
    direct_pdf: Optional[str]
    status: str  # pending | downloaded | failed | skipped
    size_kb: Optional[float] = None
    sha256: Optional[str] = None


def build_manifest(sources: Optional[List[str]] = None, limit_per_source: int = 5, query: Optional[str] = None) -> List[dict]:
    """Tạo manifest từ curated seeds + crawl bổ sung."""
    ensure_dirs()
    if sources is None:
        sources = list(SOURCE_DIRS.keys())

    manifest: List[ManifestEntry] = []
    for src in sources:
        seeds = [s for s in CURATED_SEEDS if s["source"] == src]
        if query and src in ("who_iris", "byt"):
            if src == "who_iris":
                extra = crawl_who_iris_search(query, limit=limit_per_source)
            else:
                extra = crawl_byt_search(query, limit=limit_per_source)
            # merge, ưu tiên seeds
            seen_urls = {s["url"] for s in seeds}
            for e in extra:
                if e["url"] not in seen_urls:
                    seeds.append(e)
        # giới hạn
        seeds = seeds[:limit_per_source] if limit_per_source else seeds
        for s in seeds:
            fname = slugify(s["title"]) + ".pdf"
            dest = SOURCE_DIRS[src] / fname
            manifest.append(ManifestEntry(
                source=src,
                title=s["title"],
                url=s["url"],
                year=s.get("year"),
                tags=s.get("tags", []),
                filename=fname,
                dest_path=str(dest.relative_to(BASE_DIR)),
                direct_pdf=s.get("direct_pdf"),
                status="pending",
            ))
    return [asdict(m) for m in manifest]


def save_manifest(entries: List[dict], path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    log.info(f"[Manifest] saved {len(entries)} entries -> {path}")


def load_manifest(path: Path = MANIFEST_PATH) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_download(manifest_path: Path = MANIFEST_PATH, dry_run: bool = False) -> dict:
    """Thực thi tải theo manifest."""
    entries = load_manifest(manifest_path)
    if not entries:
        log.warning("Manifest rỗng — chạy crawl trước.")
        return {"downloaded": 0, "failed": 0, "skipped": 0}

    stats = {"downloaded": 0, "failed": 0, "skipped": 0}
    for e in entries:
        dest = BASE_DIR / e["dest_path"]
        if dest.exists() and dest.stat().st_size > 1024:
            e["status"] = "skipped"
            stats["skipped"] += 1
            log.info(f"[SKIP] exists {dest.name}")
            continue
        if dry_run:
            log.info(f"[DRY] would download {e['title']} -> {e['dest_path']}")
            continue

        pdf_url = e.get("direct_pdf")
        # nếu chưa có direct_pdf, thử discover từ handle
        if not pdf_url and "iris.who.int/handle" in e["url"]:
            candidates = discover_pdfs_from_iris_handle(e["url"])
            if candidates:
                pdf_url = candidates[0]
                e["direct_pdf"] = pdf_url
                log.info(f"  discovered PDF: {pdf_url}")

        # fallback cho ADA
        if not pdf_url and e["source"] in ("ada", "diabetes") and "ada" in e["source"]:
            # thử alt_pdf
            alt = next((s.get("alt_pdf") for s in CURATED_SEEDS if s["title"] == e["title"] and s.get("alt_pdf")), None)
            if alt:
                pdf_url = alt

        if not pdf_url:
            log.warning(f"[NO PDF URL] {e['title']} — cần bổ sung thủ công.")
            e["status"] = "failed"
            stats["failed"] += 1
            continue

        ok = download_file(pdf_url, dest)
        e["status"] = "downloaded" if ok else "failed"
        if ok:
            stats["downloaded"] += 1
            e["size_kb"] = round(dest.stat().st_size / 1024, 1)
            # sha256
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            e["sha256"] = h.hexdigest()[:16]
        else:
            stats["failed"] += 1
        time.sleep(1.0)  # polite

    save_manifest(entries, manifest_path)
    # ghi metadata riêng cho pipeline indexer
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats": stats,
        "sources": sorted({e["source"] for e in entries}),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"[Done] {stats}")
    return stats


# ------------------------------------------------------------
# Đặc biệt: tải 3 PDF diabetes public (yêu cầu user)
# ------------------------------------------------------------
# Đã resolve qua DSpace 7 API (server/api/core/bitstreams/.../content) — ổn định
DIABETES_3_PUBLIC = [
    {
        "filename": "WHO_Classification_Diabetes_2019.pdf",
        "url": "https://iris.who.int/server/api/core/bitstreams/2cb3ab68-a52a-402e-ad47-8bc5a4edc834/content",
        "source": "diabetes",
        "title": "WHO Classification of Diabetes Mellitus 2019",
        "handle": "10665/325182",
    },
    {
        "filename": "WHO_HEARTS_D_Diabetes_2020.pdf",
        "url": "https://iris.who.int/server/api/core/bitstreams/2a0b4f68-7155-4ad1-b543-945791e31830/content",
        "source": "diabetes",
        "title": "WHO HEARTS-D Diagnosis & Management of Type 2 Diabetes 2020",
        "handle": "10665/331710",
    },
    {
        "filename": "WHO_PEN_2020_NCD.pdf",
        "url": "https://iris.who.int/server/api/core/bitstreams/b9f09202-a320-4c07-ba2c-afe0d1186339/content",
        "source": "diabetes",
        "title": "WHO Package of Essential NCD Interventions (PEN) 2020 — có chương Diabetes",
        "handle": "10665/334186",
        "note": "Dùng PEN thay ADA khi ADA paywall; ADA abridged sẽ thử tải riêng.",
    },
    {
        "filename": "ADA_Standards_of_Care_2024_Abridged.pdf",
        "url": "https://diabetesjournals.org/care/article/47/Supplement_1/S1/153954/Standards-of-Care-in-Diabetes-2024-Abridged-for",
        "source": "diabetes",
        "alt_url": "https://ada-journals.cld.bz/Standards-of-Care-in-Diabetes-2024-Abridged-for-Primary-Care-Providers",
        "title": "ADA Standards of Care in Diabetes 2024 — Abridged",
        "note": "ADA site paywall 403 — sẽ tạo placeholder hướng dẫn tải thủ công nếu fail.",
        "fallback_url": None,
        "optional": True,
    },
]


def download_diabetes_3(force: bool = False) -> dict:
    """Tải 3 (+1 optional) PDF diabetes public vào data/raw/pdfs/diabetes/."""
    dest_dir = SOURCE_DIRS["diabetes"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "files": []}
    for item in DIABETES_3_PUBLIC:
        dest = dest_dir / item["filename"]
        is_optional = item.get("optional", False)
        if dest.exists() and dest.stat().st_size > 1024 and not force:
            log.info(f"[SKIP] {dest.name} exists")
            stats["skipped"] += 1
            stats["files"].append(str(dest))
            continue
        url = item["url"]
        ok = download_file(url, dest)
        # thử alt/fallback nếu fail và là ADA
        if not ok and item.get("alt_url"):
            log.info(f"  trying alt_url for {item['filename']}")
            ok = download_file(item["alt_url"], dest)
        if not ok and item.get("fallback_url"):
            log.info(f"  trying fallback_url for {item['filename']}")
            fallback_dest = dest_dir / ("FALLBACK_" + item["filename"])
            ok2 = download_file(item["fallback_url"], fallback_dest)
            if ok2:
                stats["files"].append(str(fallback_dest))
                stats["downloaded"] += 1
                continue
        if ok:
            stats["downloaded"] += 1
            stats["files"].append(str(dest))
        else:
            if is_optional:
                log.warning(f"  [OPTIONAL FAIL] {item['title']} — sẽ bỏ qua, không tính vào failed chính.")
                # tạo placeholder README cho file optional
                placeholder = dest.with_suffix(".DOWNLOAD_FAILED.txt")
                placeholder.write_text(
                    f"Optional download failed (paywall): {item['title']}\n"
                    f"URL: {url}\n"
                    f"Alt: {item.get('alt_url','')}\n"
                    f"Hướng dẫn tải thủ công:\n"
                    f"  1. Truy cập https://diabetesjournals.org/care/issue/47/Supplement_1\n"
                    f"  2. Tải 'Standards of Care in Diabetes — 2024' PDF\n"
                    f"  3. Đặt file vào: {dest}\n"
                    f"  4. Chạy lại: python -m src.indexer\n"
                    f"WHO IRIS 3 file chính đã đủ để demo RAG.\n",
                    encoding="utf-8",
                )
                # không tính vào failed
                continue
            stats["failed"] += 1
            placeholder = dest.with_suffix(".DOWNLOAD_FAILED.txt")
            placeholder.write_text(
                f"Failed to download: {item['title']}\n"
                f"URL: {url}\n"
                f"Handle: {item.get('handle','')}\n"
                f"Hãy tải thủ công và đặt file vào: {dest}\n"
                f"Thử resolve lại: python -c \"from scripts.crawl_guidelines import resolve_iris_handle_to_content_url; print(resolve_iris_handle_to_content_url('https://iris.who.int/handle/{item.get('handle','')}'))\"\n",
                encoding="utf-8",
            )
            log.warning(f"  placeholder created: {placeholder.name}")
        time.sleep(1)
    log.info(f"[Diabetes 3] {stats}")
    return stats


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def cmd_crawl(args):
    entries = build_manifest(
        sources=args.source if args.source else None,
        limit_per_source=args.limit,
        query=args.query,
    )
    save_manifest(entries)
    print(f"Crawled {len(entries)} entries. Manifest: {MANIFEST_PATH}")
    for e in entries:
        print(f"  - [{e['source']}] {e['title']} -> {e['dest_path']} | pdf={e['direct_pdf']}")


def cmd_download(args):
    stats = execute_download(MANIFEST_PATH, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


def cmd_diabetes(args):
    stats = download_diabetes_3(force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_status(args):
    entries = load_manifest()
    if not entries:
        print("No manifest. Run: python scripts/crawl_guidelines.py crawl --all")
        return
    by_status = {}
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    print(f"Manifest: {MANIFEST_PATH} ({len(entries)} entries)")
    print(json.dumps(by_status, indent=2))
    # liệt kê file thực tồn tại
    for src, d in SOURCE_DIRS.items():
        if d.exists():
            files = list(d.glob("*.pdf"))
            print(f"  {src}: {len(files)} PDFs in {d}")
            for f in files[:5]:
                print(f"    - {f.name} ({f.stat().st_size/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Guideline Data Pipeline Crawler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crawl = sub.add_parser("crawl", help="Crawl metadata và tạo manifest")
    p_crawl.add_argument("--source", nargs="+", choices=list(SOURCE_DIRS.keys()), help="Nguồn cần crawl")
    p_crawl.add_argument("--all", action="store_true", help="Crawl tất cả nguồn (curated seeds)")
    p_crawl.add_argument("--limit", type=int, default=5, help="Số lượng tối đa mỗi nguồn")
    p_crawl.add_argument("--query", type=str, default=None, help="Từ khóa tìm kiếm (WHO IRIS / BYT)")
    p_crawl.set_defaults(func=cmd_crawl)

    p_dl = sub.add_parser("download", help="Tải PDF theo manifest")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_dia = sub.add_parser("diabetes3", help="Tải 3 PDF diabetes public (WHO + ADA)")
    p_dia.add_argument("--force", action="store_true", help="Tải lại dù đã tồn tại")
    p_dia.set_defaults(func=cmd_diabetes)

    p_st = sub.add_parser("status", help="Xem trạng thái manifest & file")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    # --all nghĩa là không filter source
    if getattr(args, "cmd", None) == "crawl" and args.all:
        args.source = None
    args.func(args)


if __name__ == "__main__":
    main()
