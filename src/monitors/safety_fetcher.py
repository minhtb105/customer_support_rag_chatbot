"""Safety Fetcher — FDA openFDA (anonymous + cache) + BYT/DAV weekly scrape"""
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from src.config import FDA_API_BASE, FDA_CACHE_TTL_SECONDS
    from src.monitors.db import create_safety_alert, get_source, update_source_check
    from src.monitors.utils import cached_get, assert_url_allowed, HEADERS
    from src.monitors.summarizer import summarize_safety_alert
except ImportError:
    from config import FDA_API_BASE, FDA_CACHE_TTL_SECONDS  # type: ignore
    from monitors.db import create_safety_alert, get_source, update_source_check  # type: ignore
    from monitors.utils import cached_get, assert_url_allowed, HEADERS  # type: ignore
    from monitors.summarizer import summarize_safety_alert  # type: ignore

# ---------- FDA openFDA ----------
def _fda_severity_from_classification(classification: str) -> str:
    cl = (classification or "").lower()
    if "class i" in cl:
        return "critical"
    if "class ii" in cl:
        return "high"
    if "class iii" in cl:
        return "medium"
    return "medium"

def _fda_alert_type(reason: str) -> str:
    r = (reason or "").lower()
    if "recall" in r:
        return "recall"
    if "interaction" in r:
        return "interaction"
    if "boxed" in r or "warning" in r:
        return "box_warning"
    if "shortage" in r:
        return "shortage"
    return "label_change"

def fetch_fda_recalls(limit: int = 10, days_back: int = 7) -> List[Dict[str, Any]]:
    """Fetch FDA enforcement recalls (openFDA anonymous)."""
    # Search for recent recalls: recall_initiation_date within days_back
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
    # openFDA search syntax: recall_initiation_date:[20240101+TO+20251231]
    url = f"{FDA_API_BASE}/drug/enforcement.json?search=recall_initiation_date:[{since}+TO+99991231]&limit={limit}&sort=recall_initiation_date:desc"
    try:
        data = cached_get(url, ttl_seconds=FDA_CACHE_TTL_SECONDS)
        results = data.get("results", [])
        out = []
        for item in results[:limit]:
            raw = {
                "alert_title": item.get("product_description", "")[:200] or item.get("reason_for_recall", "")[:200],
                "drug_name": item.get("product_description", "").split(",")[0][:100] if item.get("product_description") else item.get("product_type", ""),
                "reason_for_recall": item.get("reason_for_recall", ""),
                "classification": item.get("classification", ""),
                "recall_number": item.get("recall_number", ""),
                "recall_initiation_date": item.get("recall_initiation_date", ""),
                "city": item.get("city", ""),
                "state": item.get("state", ""),
                "country": item.get("country", ""),
                "voluntary_mandated": item.get("voluntary_mandated", ""),
                "distribution_pattern": item.get("distribution_pattern", ""),
                "product_description": item.get("product_description", ""),
                "openfda": item.get("openfda", {}),
            }
            out.append(raw)
        return out
    except Exception as e:
        print(f"[FDA] fetch recalls failed: {e}")
        return []

def fetch_fda_label_warnings(limit: int = 5, days_back: int = 14) -> List[Dict[str, Any]]:
    """Fetch recent label changes with boxed warnings (best effort, may be sparse)."""
    # Use drug/label with boxed_warning search — openFDA sometimes slow, keep small
    url = f"{FDA_API_BASE}/drug/label.json?search=boxed_warning:*&limit={limit}"
    try:
        data = cached_get(url, ttl_seconds=FDA_CACHE_TTL_SECONDS)
        results = data.get("results", [])
        out = []
        for item in results[:limit]:
            boxed = item.get("boxed_warning", [""])[0] if item.get("boxed_warning") else ""
            out.append({
                "alert_title": (item.get("openfda", {}).get("brand_name", [""])[0] or "Boxed warning update") + " — boxed warning",
                "drug_name": item.get("openfda", {}).get("brand_name", [""])[0] if item.get("openfda", {}).get("brand_name") else "",
                "boxed_warning": boxed[:500],
                "openfda": item.get("openfda", {}),
                "effective_time": item.get("effective_time", ""),
            })
        return out
    except Exception as e:
        print(f"[FDA] label warnings failed: {e}")
        return []

def check_fda_alerts() -> Dict[str, Any]:
    """Check FDA openFDA for new recalls + warnings, stage as safety_alerts."""
    found = 0
    created_ids: List[str] = []
    # Recalls
    recalls = fetch_fda_recalls(limit=10, days_back=7)
    for raw in recalls:
        severity = _fda_severity_from_classification(raw.get("classification", ""))
        alert_type = _fda_alert_type(raw.get("reason_for_recall", ""))
        # Use recall_number as url surrogate
        recall_no = raw.get("recall_number", "")
        url = f"https://api.fda.gov/drug/enforcement.json?search=recall_number:\"{recall_no}\"" if recall_no else "https://api.fda.gov/drug/enforcement.json"
        # Determine published date
        pub = raw.get("recall_initiation_date")
        if pub and len(pub) == 8:
            try:
                pub = datetime.strptime(pub, "%Y%m%d").strftime("%Y-%m-%d")
            except Exception:
                pub = None
        # Summarize via LLM (VI)
        summary = summarize_safety_alert(raw, source="FDA")
        # Map summary severity if more accurate
        if summary.get("muc_do"):
            # trust LLM if it downgrades/upgrades within reason, but keep critical if Class I
            if severity == "critical":
                final_sev = "critical"
            else:
                sev_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
                final_sev = sev_map.get(summary.get("muc_do", severity), severity)
        else:
            final_sev = severity
        drug_name = raw.get("drug_name") or (summary.get("thuoc_lien_quan", [""])[0] if summary.get("thuoc_lien_quan") else "")
        alert = create_safety_alert(
            source="FDA",
            alert_type=alert_type,
            alert_title=raw.get("alert_title") or summary.get("tieu_de_vi", "FDA Recall"),
            severity=final_sev,
            drug_name=drug_name[:100] if drug_name else None,
            alert_url=url,
            published_date=pub,
            raw_json=json.dumps(raw, ensure_ascii=False),
            ai_summary=json.dumps(summary, ensure_ascii=False),
            ai_risk_score=summary.get("diem_rui_ro", 0.5),
            assigned_role="pharmacist",
        )
        # if newly created (check created_at ~ now), count
        # create_safety_alert dedup returns existing if duplicate, so check is_notified?
        # We check if alert was just created by seeing if summary matches? Simpler: always count if not duplicate
        # We'll count if alert's created_at is within last 2 minutes
        try:
            created_at = datetime.fromisoformat(alert["created_at"])
            if (datetime.utcnow() - created_at).total_seconds() < 120:
                found += 1
                created_ids.append(alert["id"])
                # notify
                try:
                    from src.monitors.service import notify_safety_pending
                    notify_safety_pending(alert, summary)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.2)

    # Label boxed warnings (less frequent) — treat as high
    labels = fetch_fda_label_warnings(limit=3, days_back=30)
    for raw in labels:
        summary = summarize_safety_alert(raw, source="FDA")
        alert = create_safety_alert(
            source="FDA",
            alert_type="box_warning",
            alert_title=raw.get("alert_title", "Boxed warning"),
            severity="high",
            drug_name=raw.get("drug_name"),
            alert_url="https://api.fda.gov/drug/label.json",
            published_date=raw.get("effective_time"),
            raw_json=json.dumps(raw, ensure_ascii=False),
            ai_summary=json.dumps(summary, ensure_ascii=False),
            ai_risk_score=summary.get("diem_rui_ro", 0.6),
            assigned_role="pharmacist",
        )
        try:
            created_at = datetime.fromisoformat(alert["created_at"])
            if (datetime.utcnow() - created_at).total_seconds() < 120:
                found += 1
                created_ids.append(alert["id"])
                try:
                    from src.monitors.service import notify_safety_pending
                    notify_safety_pending(alert, summary)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.2)

    # update source check
    try:
        update_source_check("fda_recall", last_hash=str(found))
    except Exception:
        pass

    return {"source": "FDA", "found_new": found, "created_ids": created_ids, "checked_at": datetime.utcnow().isoformat()}

# ---------- BYT / DAV scrape (weekly) ----------
def fetch_byt_dav_alerts() -> Dict[str, Any]:
    """Scrape DAV (dav.gov.vn) and MOH for thu hồi / cảnh báo dược."""
    urls_to_try = [
        ("DAV", "https://dav.gov.vn"),
        ("MOH", "https://moh.gov.vn/thong-bao-thu-hoi-thuoc.html"),  # may redirect
        ("thuvienphapluat", "https://thuvienphapluat.vn/tim-kiem-van-ban.aspx?keyword=thu+h%E1%BB%93i+thu%E1%BB%91c&type=0"),
    ]
    found = 0
    created_ids: List[str] = []
    for source_name, url in urls_to_try:
        try:
            assert_url_allowed(url)
        except Exception:
            continue
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            # Heuristic: find links containing thu hồi / cảnh báo / recall
            keywords = ["thu hồi", "thu hoi", "cảnh báo", "canh bao", "recall", "đình chỉ"]
            links = []
            for a in soup.find_all("a", href=True):
                txt = a.get_text(strip=True).lower()
                href = a["href"]
                if any(k in txt for k in keywords):
                    from urllib.parse import urljoin
                    full = urljoin(url, href)
                    links.append((a.get_text(strip=True)[:200], full))
            # Limit to 5 per source
            for title, full_url in links[:5]:
                # dedup via DB will handle
                raw = {"alert_title": title, "alert_url": full_url, "source": source_name, "scraped_from": url}
                severity = "high" if "thu hồi" in title.lower() or "recall" in title.lower() else "medium"
                summary = summarize_safety_alert(raw, source=source_name)
                alert = create_safety_alert(
                    source=source_name,
                    alert_type="recall" if "thu hồi" in title.lower() else "label_change",
                    alert_title=title,
                    severity=severity,
                    drug_name=None,
                    alert_url=full_url,
                    published_date=None,
                    raw_json=json.dumps(raw, ensure_ascii=False),
                    ai_summary=json.dumps(summary, ensure_ascii=False),
                    ai_risk_score=summary.get("diem_rui_ro", 0.5),
                    assigned_role="pharmacist",
                )
                try:
                    created_at = datetime.fromisoformat(alert["created_at"])
                    if (datetime.utcnow() - created_at).total_seconds() < 120:
                        found += 1
                        created_ids.append(alert["id"])
                        try:
                            from src.monitors.service import notify_safety_pending
                            notify_safety_pending(alert, summary)
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.3)
        except Exception as e:
            print(f"[BYT scrape {source_name}] {e}")
            continue
    # update checks
    for key in ("dav_thuhoi", "moh_canhbao", "byt_diabetes"):
        try:
            update_source_check(key, last_hash=str(found))
        except Exception:
            pass
    return {"source": "BYT/DAV", "found_new": found, "created_ids": created_ids}

def check_all_safety() -> Dict[str, Any]:
    """Run both FDA and BYT checks."""
    res_fda = check_fda_alerts()
    time.sleep(1)
    res_byt = fetch_byt_dav_alerts()
    total = res_fda.get("found_new", 0) + res_byt.get("found_new", 0)
    return {"fda": res_fda, "byt_dav": res_byt, "total_found": total}
