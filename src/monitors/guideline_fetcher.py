"""Guideline Fetcher — HEAD/etag check, staging download, diff, summarize"""
from __future__ import annotations
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from bs4 import BeautifulSoup

try:
    from src.config import BASE_DIR, STAGING_DIR, PDF_DIR, GUIDELINE_SOURCE_DIRS
    from src.monitors.db import create_guideline_version, get_guideline_version, list_guideline_versions, update_source_check, get_source
    from src.monitors.utils import head_with_etag, download_file_safe, sha256_file, extract_text_from_pdf, assert_url_allowed
    from src.monitors.summarizer import summarize_guideline_diff
except ImportError:
    from config import BASE_DIR, STAGING_DIR, PDF_DIR, GUIDELINE_SOURCE_DIRS  # type: ignore
    from monitors.db import create_guideline_version, get_guideline_version, list_guideline_versions, update_source_check, get_source  # type: ignore
    from monitors.utils import head_with_etag, download_file_safe, sha256_file, extract_text_from_pdf, assert_url_allowed  # type: ignore
    from monitors.summarizer import summarize_guideline_diff  # type: ignore

# ---------- source definitions ----------
# Reuse crawl_guidelines CURATED_SEEDS but simplified to direct PDF where possible
GUIDELINE_SOURCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ada_soc": {
        "source": "ada",
        "title": "ADA Standards of Care in Diabetes",
        "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        "direct_pdf": None,  # paywall, will try HEAD and fallback
        "check_url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        "version_regex": r"Standards of Care.*?(\d{4})",
    },
    "gold": {
        "source": "gold",
        "title": "GOLD Global Strategy for COPD",
        "url": "https://goldcopd.org/2024-gold-report/",
        "direct_pdf": "https://goldcopd.org/wp-content/uploads/2024/02/GOLD-2024_v1.2-11Jan24_WMV.pdf",
        "check_url": "https://goldcopd.org/wp-content/uploads/2024/02/GOLD-2024_v1.2-11Jan24_WMV.pdf",
    },
    "gina": {
        "source": "gina",
        "title": "GINA Global Strategy for Asthma",
        "url": "https://ginasthma.org/2024-gina-main-report/",
        "direct_pdf": "https://ginasthma.org/wp-content/uploads/2024/05/GINA-2024-Strategy-Report-24_05_22_WMS.pdf",
        "check_url": "https://ginasthma.org/wp-content/uploads/2024/05/GINA-2024-Strategy-Report-24_05_22_WMS.pdf",
    },
    "who_mhgap": {
        "source": "mhgap",
        "title": "WHO mhGAP Intervention Guide v2.0",
        "url": "https://iris.who.int/handle/10665/250239",
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/250239/9789241549790-eng.pdf",
        "check_url": "https://iris.who.int/bitstream/handle/10665/250239/9789241549790-eng.pdf",
        "resolve_handle": "https://iris.who.int/handle/10665/250239",
    },
    "who_diabetes": {
        "source": "diabetes",
        "title": "WHO Classification of Diabetes / HEARTS-D",
        "url": "https://iris.who.int/handle/10665/325182",
        "direct_pdf": "https://iris.who.int/bitstream/handle/10665/325182/9789241515702-eng.pdf",
        "check_url": "https://iris.who.int/bitstream/handle/10665/325182/9789241515702-eng.pdf",
        "resolve_handle": "https://iris.who.int/handle/10665/325182",
    },
    "aha_acc_htn": {
        "source": "aha_acc",
        "title": "AHA/ACC Hypertension Guideline",
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
        "direct_pdf": "https://www.ahajournals.org/doi/pdf/10.1161/HYP.0000000000000065",
        "check_url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
    },
    "byt_diabetes": {
        "source": "byt",
        "title": "BYT — Hướng dẫn chẩn đoán & điều trị Đái tháo đường (QĐ 3319)",
        "url": "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-3319-QD-BYT-2017-huong-dan-chan-doan-dieu-tri-dai-thao-duong-356066.aspx",
        "check_url": "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-3319-QD-BYT-2017-huong-dan-chan-doan-dieu-tri-dai-thao-duong-356066.aspx",
        "is_byt_scrape": True,
    },
    "byt_3798": {
        "source": "byt",
        "title": "BYT — Quy trình lâm sàng ĐTĐ típ 2 (QĐ 3798/2017)",
        "url": "https://daithaoduong.kcb.vn/quy-trinh-lam-sang-dieu-tri-dai-thao-duong",
        "check_url": "https://daithaoduong.kcb.vn/quy-trinh-lam-sang-dieu-tri-dai-thao-duong",
        "is_byt_scrape": True,
        "note": "kcb.vn phục vụ chung 1 file PDF cho cả QĐ 3319 và 3798 — corpus lưu 1 bản, không duplicate.",
    },
    "byt_5481": {
        "source": "byt",
        "title": "BYT — HD chẩn đoán & điều trị ĐTĐ típ 2 (QĐ 5481/2020, thay thế 3319)",
        "url": "https://bvcdn.org.vn/quyet-dinh-so-5481-qd-byt-ngay-30-thang-12-nam-2020-cua-bo-y-te-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2",
        "check_url": "https://bvcdn.org.vn/quyet-dinh-so-5481-qd-byt-ngay-30-thang-12-nam-2020-cua-bo-y-te-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2",
        "is_byt_scrape": True,
    },
}

def _slugify(text: str) -> str:
    t = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    t = re.sub(r"[-\s]+", "_", t.strip())
    return t[:80].strip("_").lower()

def _find_latest_approved(source: str) -> Optional[Dict[str, Any]]:
    rows, _ = list_guideline_versions(source=source, limit=20)
    for r in rows:
        if r["status"] == "approved":
            return r
    return None

def check_guideline_update(source_key: str, force_download: bool = False) -> Dict[str, Any]:
    """Check single guideline source for update via HEAD/etag, return status dict."""
    cfg = GUIDELINE_SOURCE_CONFIGS.get(source_key)
    if not cfg:
        return {"source_key": source_key, "error": "unknown source_key", "found_new": False}
    source = cfg["source"]
    check_url = cfg.get("check_url") or cfg["url"]
    # HEAD check
    head = head_with_etag(check_url)
    etag = head.get("etag")
    last_mod = head.get("last_modified")
    # Compare with stored
    stored = get_source(source_key)
    needs_download = force_download
    if not needs_download and stored:
        if etag and stored.get("last_etag") and etag != stored["last_etag"]:
            needs_download = True
        elif last_mod and stored.get("last_modified") and last_mod != stored["last_modified"]:
            needs_download = True
        elif not etag and not last_mod:
            # no etag support -> check monthly by forcing? For now weekly deep check will force
            needs_download = False
        # if never checked, need download
        if not stored.get("last_checked_at"):
            needs_download = True
    elif not stored:
        needs_download = True

    # For BYT scrape, always do scrape check (weekly)
    if cfg.get("is_byt_scrape"):
        return _check_byt_scrape(source_key, cfg, force_download=force_download)

    if not needs_download and not force_download:
        # update last_checked
        update_source_check(source_key, etag=etag, last_modified=last_mod)
        return {"source_key": source_key, "checked_url": check_url, "etag": etag, "last_modified": last_mod, "found_new": False, "reason": "etag unchanged"}

    # Need to download to staging
    return _download_and_stage(source_key, cfg, etag=etag, last_modified=last_mod)

def _check_byt_scrape(source_key: str, cfg: Dict[str, Any], force_download: bool = False) -> Dict[str, Any]:
    """Scrape thuvienphapluat.vn + moh.gov.vn for BYT diabetes updates."""
    url = cfg["url"]
    try:
        assert_url_allowed(url)
        # scrape thuvienphapluat
        from src.monitors.utils import HEADERS
        import requests
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # extract title and date
        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True)[:200] if title_el else cfg["title"]
        # try find date
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", r.text)
        pub_date = None
        if date_match:
            try:
                from datetime import datetime as dt
                d = dt.strptime(date_match.group(1), "%d/%m/%Y")
                pub_date = d.strftime("%Y-%m-%d")
            except Exception:
                pass
        # also check for PDF link
        pdf_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf") and "thuvienphapluat" in href or href.lower().endswith(".pdf"):
                # resolve
                from urllib.parse import urljoin
                pdf_link = urljoin(url, href)
                break
        # Compare hash of HTML content as etag surrogate
        content_hash = hashlib.sha256(r.text.encode()).hexdigest()[:16]
        stored = get_source(source_key)
        if not force_download and stored and stored.get("last_hash") == content_hash:
            update_source_check(source_key, last_hash=content_hash)
            return {"source_key": source_key, "found_new": False, "reason": "BYT html hash unchanged", "title": title}
        # Found change -> stage as HTML snapshot + try PDF
        staging_base = STAGING_DIR / cfg["source"] / f"byt_scrape_{content_hash}"
        staging_base.mkdir(parents=True, exist_ok=True)
        html_path = staging_base / "snapshot.html"
        html_path.write_text(r.text, encoding="utf-8")
        sha = hashlib.sha256(r.text.encode()).hexdigest()
        # Create guideline_version entry (staging_path = html snapshot)
        # Find latest approved to diff
        latest = _find_latest_approved(cfg["source"])
        old_text = None
        if latest and latest.get("corpus_path"):
            try:
                old_path = Path(latest["corpus_path"])
                if old_path.exists() and old_path.suffix == ".pdf":
                    old_text = extract_text_from_pdf(old_path)
                elif old_path.exists():
                    old_text = old_path.read_text(encoding="utf-8", errors="ignore")[:8000]
            except Exception:
                pass
        new_text = r.text[:8000]
        summary = summarize_guideline_diff(
            source=cfg["source"],
            title=title,
            old_text=old_text,
            new_text=new_text,
            version_hint=pub_date or content_hash,
            publication_date_hint=pub_date,
        )
        # If PDF link found, also stage PDF
        pdf_staging = None
        if pdf_link:
            try:
                pdf_path = staging_base / "document.pdf"
                if download_file_safe(pdf_link, pdf_path):
                    pdf_staging = str(pdf_path)
                    # also extract better text
                    new_text = extract_text_from_pdf(pdf_path)
                    summary = summarize_guideline_diff(
                        source=cfg["source"],
                        title=title,
                        old_text=old_text,
                        new_text=new_text,
                        version_hint=pub_date or content_hash,
                        publication_date_hint=pub_date,
                    )
            except Exception:
                pass
        staging_path = pdf_staging or str(html_path)
        # create DB entry
        gv = create_guideline_version(
            source=cfg["source"],
            title=title,
            url=pdf_link or url,
            version_label=pub_date or f"BYT-{content_hash}",
            publication_date=pub_date,
            sha256=sha,
            etag=content_hash,
            staging_path=staging_path,
            change_summary_json=json.dumps(summary, ensure_ascii=False),
            diff_text=new_text[:2000],
            supersedes_id=latest["id"] if latest else None,
        )
        update_source_check(source_key, last_hash=content_hash, last_modified=pub_date)
        # create notification via shared helper
        try:
            from src.monitors.service import notify_guideline_pending
            notify_guideline_pending(gv, summary)
        except Exception:
            pass
        return {"source_key": source_key, "found_new": True, "guideline_version_id": gv["id"], "title": title, "sha256": sha}
    except Exception as e:
        return {"source_key": source_key, "found_new": False, "error": str(e)[:500]}

def _download_and_stage(source_key: str, cfg: Dict[str, Any], etag: Optional[str] = None, last_modified: Optional[str] = None) -> Dict[str, Any]:
    direct_pdf = cfg.get("direct_pdf")
    # if no direct PDF (e.g., ADA paywall), we can't download PDF — create a placeholder pending manual
    if not direct_pdf:
        # For ADA, we treat as pending_manual — still create entry to flag review
        # Try to resolve handle if present
        handle = cfg.get("resolve_handle")
        if handle:
            # Try to resolve via DSpace API like crawl_guidelines does
            try:
                from scripts.crawl_guidelines import resolve_iris_handle_to_content_url  # type: ignore
                resolved = resolve_iris_handle_to_content_url(handle)
                if resolved:
                    direct_pdf = resolved
            except Exception:
                pass
        if not direct_pdf:
            # create placeholder entry noting paywall
            summary = {
                "tom_tat_tieng_viet": f"[{cfg['title']}] chưa có PDF trực tiếp (paywall). Cần tải thủ công từ {cfg['url']}. Hệ thống gắn cờ để chuyên gia bổ sung.",
                "changed_sections": [],
                "dosage_changes": [],
                "new_recommendations": [],
                "removed_recommendations": [],
                "version_phat_hien": "",
                "ngay_xuat_ban": "",
                "muc_do_quan_trong": "medium",
                "_paywall": True,
            }
            # Avoid duplicate paywall entry if already pending
            rows, _ = list_guideline_versions(source=cfg["source"], status="pending_review", limit=5)
            for r in rows:
                if "paywall" in (r.get("change_summary_json") or "").lower() and r["title"] == cfg["title"]:
                    update_source_check(source_key, etag=etag, last_modified=last_modified)
                    return {"source_key": source_key, "found_new": False, "reason": "paywall placeholder already pending"}
            gv = create_guideline_version(
                source=cfg["source"],
                title=cfg["title"],
                url=cfg["url"],
                version_label="manual-required",
                sha256=None,
                etag=etag,
                staging_path=None,
                change_summary_json=json.dumps(summary, ensure_ascii=False),
                diff_text="Paywall — cần tải thủ công",
            )
            try:
                from src.monitors.service import notify_guideline_pending
                notify_guideline_pending(gv, summary)
            except Exception:
                pass
            update_source_check(source_key, etag=etag, last_modified=last_modified)
            return {"source_key": source_key, "found_new": True, "guideline_version_id": gv["id"], "note": "paywall placeholder"}
    # Download PDF to staging
    # Determine version label from etag or date
    version_label = (etag or last_modified or datetime.utcnow().strftime("%Y%m%d"))[:32]
    version_label = re.sub(r"[^\w-]", "_", version_label)[:32]
    staging_dir = STAGING_DIR / cfg["source"] / version_label
    staging_dir.mkdir(parents=True, exist_ok=True)
    fname = _slugify(cfg["title"]) + ".pdf"
    dest = staging_dir / fname
    ok = download_file_safe(direct_pdf, dest)
    if not ok:
        return {"source_key": source_key, "found_new": False, "error": f"download failed {direct_pdf}"}
    sha = sha256_file(dest)
    # Check duplicate sha vs latest approved
    latest = _find_latest_approved(cfg["source"])
    if latest and latest.get("sha256") == sha:
        # same content, just update check
        update_source_check(source_key, etag=etag, last_modified=last_modified, last_hash=sha)
        # cleanup staging duplicate
        try:
            dest.unlink()
            if not any(staging_dir.iterdir()):
                staging_dir.rmdir()
        except Exception:
            pass
        return {"source_key": source_key, "found_new": False, "reason": "sha256 identical to approved"}
    # Also check pending duplicate
    rows_pending, _ = list_guideline_versions(source=cfg["source"], status="pending_review", limit=20)
    for r in rows_pending:
        if r.get("sha256") == sha:
            update_source_check(source_key, etag=etag, last_modified=last_modified, last_hash=sha)
            return {"source_key": source_key, "found_new": False, "reason": "sha256 already pending_review"}

    # Extract texts for diff
    old_text = None
    if latest and latest.get("corpus_path"):
        try:
            old_path = Path(latest["corpus_path"])
            if old_path.exists():
                old_text = extract_text_from_pdf(old_path)
        except Exception:
            pass
    new_text = extract_text_from_pdf(dest)
    summary = summarize_guideline_diff(
        source=cfg["source"],
        title=cfg["title"],
        old_text=old_text,
        new_text=new_text,
        version_hint=version_label,
        publication_date_hint=last_modified,
    )
    gv = create_guideline_version(
        source=cfg["source"],
        title=cfg["title"],
        url=direct_pdf,
        version_label=version_label,
        publication_date=last_modified,
        sha256=sha,
        etag=etag,
        staging_path=str(dest),
        change_summary_json=json.dumps(summary, ensure_ascii=False),
        diff_text=new_text[:3000],
        supersedes_id=latest["id"] if latest else None,
    )
    update_source_check(source_key, etag=etag, last_modified=last_modified, last_hash=sha)
    try:
        from src.monitors.service import notify_guideline_pending
        notify_guideline_pending(gv, summary)
    except Exception as e:
        print(f"[guideline notify] {e}")
    return {"source_key": source_key, "found_new": True, "guideline_version_id": gv["id"], "sha256": sha, "staging_path": str(dest)}

def check_all_guidelines(force: bool = False) -> List[Dict[str, Any]]:
    results = []
    for key in GUIDELINE_SOURCE_CONFIGS.keys():
        try:
            res = check_guideline_update(key, force_download=force)
            results.append(res)
        except Exception as e:
            results.append({"source_key": key, "found_new": False, "error": str(e)[:300]})
        time.sleep(0.5)  # polite between sources
    return results
