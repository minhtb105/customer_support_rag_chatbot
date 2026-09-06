"""Scheduler — runs guideline (monthly/quarterly) + safety (daily/weekly) checks
Uses `schedule` library (already in requirements). Run as sidecar:
  python -m src.monitors.scheduler
Or imported as background thread from FastAPI lifespan.
"""
from __future__ import annotations
import time
import logging
from datetime import datetime

try:
    import schedule  # type: ignore
    _HAS_SCHEDULE = True
except ImportError:
    _HAS_SCHEDULE = False
    schedule = None  # type: ignore

try:
    from src.monitors.db import create_run, finish_run, delete_superseded_expired
    from src.monitors.guideline_fetcher import check_all_guidelines, check_guideline_update
    from src.monitors.safety_fetcher import check_all_safety, check_fda_alerts, fetch_byt_dav_alerts
    from src.config import MONITOR_SUPERSEDED_RETENTION_DAYS
except ImportError:
    from monitors.db import create_run, finish_run, delete_superseded_expired  # type: ignore
    from monitors.guideline_fetcher import check_all_guidelines, check_guideline_update  # type: ignore
    from monitors.safety_fetcher import check_all_safety, check_fda_alerts, fetch_byt_dav_alerts  # type: ignore
    from config import MONITOR_SUPERSEDED_RETENTION_DAYS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def job_fda_daily():
    log.info("[scheduler] FDA daily check start")
    rid = create_run("fda_recall")
    try:
        res = check_fda_alerts()
        finish_run(rid, found_new=res.get("found_new", 0), status="success")
        log.info(f"[scheduler] FDA done: {res}")
    except Exception as e:
        finish_run(rid, status="failed", error=str(e)[:500])
        log.exception(f"[scheduler] FDA failed: {e}")

def job_byt_weekly():
    log.info("[scheduler] BYT/DAV weekly check start")
    rid = create_run("dav_thuhoi")
    try:
        res = fetch_byt_dav_alerts()
        finish_run(rid, found_new=res.get("found_new", 0), status="success")
        log.info(f"[scheduler] BYT done: {res}")
    except Exception as e:
        finish_run(rid, status="failed", error=str(e)[:500])
        log.exception(f"[scheduler] BYT failed: {e}")

def job_guideline_monthly_head():
    log.info("[scheduler] Guideline monthly HEAD check start")
    # HEAD-only (no force)
    try:
        results = check_all_guidelines(force=False)
        total = sum(1 for r in results if r.get("found_new"))
        log.info(f"[scheduler] Guideline monthly done: found {total}/{len(results)}")
    except Exception as e:
        log.exception(f"[scheduler] Guideline monthly failed: {e}")

def job_guideline_quarterly_deep():
    log.info("[scheduler] Guideline quarterly deep check start")
    try:
        results = check_all_guidelines(force=True)
        total = sum(1 for r in results if r.get("found_new"))
        log.info(f"[scheduler] Guideline quarterly done: found {total}")
    except Exception as e:
        log.exception(f"[scheduler] Guideline quarterly failed: {e}")

def job_cleanup_superseded():
    log.info("[scheduler] Cleanup superseded start")
    try:
        deleted = delete_superseded_expired(days=MONITOR_SUPERSEDED_RETENTION_DAYS)
        log.info(f"[scheduler] Cleanup deleted {deleted}")
    except Exception as e:
        log.exception(f"[scheduler] Cleanup failed: {e}")

def setup_schedule():
    if not _HAS_SCHEDULE:
        log.warning("schedule library not installed, scheduler disabled")
        return
    # Safety: daily 02:00 for FDA critical
    schedule.every().day.at("02:00").do(job_fda_daily)
    # BYT weekly Monday 03:00
    schedule.every().monday.at("03:00").do(job_byt_weekly)
    # Guideline monthly HEAD on 1st day 04:00 — we simulate via daily check if day==1
    schedule.every().day.at("04:00").do(lambda: job_guideline_monthly_head() if datetime.utcnow().day == 1 else None)
    # Quarterly deep: 01 Jan/Apr/Jul/Oct 05:00
    schedule.every().day.at("05:00").do(lambda: job_guideline_quarterly_deep() if datetime.utcnow().day == 1 and datetime.utcnow().month in (1,4,7,10) else None)
    # Cleanup weekly Sunday 06:00
    schedule.every().sunday.at("06:00").do(job_cleanup_superseded)
    log.info("Scheduler jobs registered: FDA daily 02:00, BYT weekly Mon 03:00, guideline monthly 04:00 (day1), quarterly deep, cleanup Sunday 06:00")

def run_loop(poll_seconds: int = 60):
    setup_schedule()
    if not _HAS_SCHEDULE:
        return
    log.info("Monitor scheduler loop started (poll 60s)")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            log.exception(f"scheduler run_pending error: {e}")
        time.sleep(poll_seconds)

def start_background_thread():
    import threading
    setup_schedule()
    if not _HAS_SCHEDULE:
        return None
    def _loop():
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                log.exception(f"scheduler bg error: {e}")
            time.sleep(60)
    t = threading.Thread(target=_loop, daemon=True, name="monitor-scheduler")
    t.start()
    log.info("Monitor scheduler background thread started")
    return t

if __name__ == "__main__":
    run_loop()
