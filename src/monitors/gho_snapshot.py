"""GHO Snapshot — WHO Global Health Observatory OData (Dual-Storage Strategy)

(a) Thu thap: snapshot dinh ky tu GHO OData API + ETL ngay (giữ IndicatorCode cần
    thiết, lọc SpatialDim VNM, bỏ metadata rác).
(b) Luu tru & truy xuat — 2 luong:
    - Luong A (Textualization -> Chroma): datapoint -> cau tieng Viet tu nhien,
      index nhu chunk RAG thuong (cau hoi mo).
    - Luong B (Structured + Tool Calling): JSON sach -> SQLite local
      (data/processed/gho_stats.db) + tool query readonly cho tinh toan chinh xac.

Theo R1 (debate): snapshot KHONG qua human-review; audit qua monitor_runs.
Theo R2: khong staging-flow, khong public API endpoint v1 (tool noi bo).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

log = logging.getLogger(__name__)

try:
    from src.config import BASE_DIR, GHO_API_BASE, GHO_INDICATORS, GHO_SNAPSHOT_DIR, GHO_TEXT_DIR, GHO_DB_PATH
    from src.monitors.db import create_run, finish_run
    from src.monitors.utils import assert_url_allowed
except ImportError:  # type: ignore
    from config import BASE_DIR, GHO_API_BASE, GHO_INDICATORS, GHO_SNAPSHOT_DIR, GHO_TEXT_DIR, GHO_DB_PATH  # type: ignore
    from monitors.db import create_run, finish_run  # type: ignore
    from monitors.utils import assert_url_allowed  # type: ignore

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GuidelineCrawler/1.0", "Accept": "application/json"}

SEX_LABEL = {"SEX_BTSX": "cả hai giới", "SEX_MLE": "nam", "SEX_FMLE": "nữ"}

# Cau hoi ve so lieu/so sanh -> dung tool SQL thay vi RAG so
STATS_KEYWORDS = [
    "tỷ lệ", "ty le", "phần trăm", "phan tram", "%", "tăng bao nhiêu", "tang bao nhieu",
    "giảm bao nhiêu", "giam bao nhieu", "so với", "so voi", "so sánh", "so sanh",
    "prevalence", "coverage", "bao nhiêu", "bao nhieu", "thống kê", "thong ke",
    "tình hình", "tinh hinh", "xu hướng", "xu huong",
]


def fetch_indicator(code: str, countries: tuple = ("VNM",), timeout: int = 60) -> List[Dict[str, Any]]:
    """GET GHO OData entity set cho 1 indicator, loc theo SpatialDim."""
    filt = " or ".join([f"SpatialDim eq '{c}'" for c in countries])
    url = f"{GHO_API_BASE}/{code}?$filter={filt}"
    assert_url_allowed(url)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json().get("value", [])


def etl_records(code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Giu field can thiet, bo metadata rac. Chi giu hang nam (YEAR)."""
    out = []
    for row in rows:
        if row.get("TimeDimType") != "YEAR":
            continue
        try:
            year = int(row.get("TimeDim"))
            value = float(row.get("NumericValue"))
        except (TypeError, ValueError):
            continue
        out.append({
            "indicator_code": code,
            "country": row.get("SpatialDim"),
            "year": year,
            "sex": row.get("Dim1") or "SEX_BTSX",
            "value": round(value, 2),
            "low": row.get("Low"),
            "high": row.get("High"),
        })
    # dedup (code, country, year, sex) — giu ban ghi dau tien
    seen, deduped = set(), []
    for rec in sorted(out, key=lambda x: (x["year"], x["sex"])):
        key = (rec["indicator_code"], rec["country"], rec["year"], rec["sex"])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    return deduped


def save_snapshot(code: str, records: List[Dict[str, Any]]) -> Path:
    d = Path(GHO_SNAPSHOT_DIR) / code
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    p = d / f"{stamp}.json"
    p.write_text(json.dumps({
        "indicator_code": code,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": f"{GHO_API_BASE}/{code}",
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def upsert_sqlite(records: List[Dict[str, Any]]) -> int:
    Path(GHO_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(GHO_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gho_stats (
            indicator_code TEXT NOT NULL,
            country TEXT NOT NULL,
            year INTEGER NOT NULL,
            sex TEXT NOT NULL,
            value REAL NOT NULL,
            low REAL, high REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (indicator_code, country, year, sex)
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for rec in records:
        conn.execute(
            "INSERT INTO gho_stats (indicator_code, country, year, sex, value, low, high, fetched_at)"
            " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(indicator_code, country, year, sex)"
            " DO UPDATE SET value=excluded.value, low=excluded.low, high=excluded.high, fetched_at=excluded.fetched_at",
            (rec["indicator_code"], rec["country"], rec["year"], rec["sex"],
             rec["value"], rec["low"], rec["high"], now),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def textualize(records: List[Dict[str, Any]]) -> List[str]:
    """Datapoint -> cau tieng Viet (chi diem chinh: BTSX + nam gan nhat moi chi so)."""
    by_indicator: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_indicator.setdefault(rec["indicator_code"], []).append(rec)
    lines = []
    for code, recs in by_indicator.items():
        meta = GHO_INDICATORS.get(code, {})
        vi_name = meta.get("vi", code)
        btsx = sorted([r for r in recs if r["sex"] == "SEX_BTSX"], key=lambda x: x["year"])
        if not btsx:
            continue
        latest = btsx[-1]
        lines.append(
            f"Theo số liệu của Tổ chức Y tế Thế giới (WHO Global Health Observatory), "
            f"{vi_name} tại Việt Nam năm {latest['year']} là {latest['value']}%."
        )
        if len(btsx) >= 2:
            first = btsx[0]
            delta = round(latest["value"] - first["value"], 2)
            trend = "tăng" if delta > 0 else "giảm"
            lines.append(
                f"So với năm {first['year']} ({first['value']}%), chỉ số này đã {trend} "
                f"{abs(delta)} điểm phần trăm sau {latest['year'] - first['year']} năm."
            )
    return lines


def save_texts(lines: List[str], country: str = "VNM") -> Path:
    Path(GHO_TEXT_DIR).mkdir(parents=True, exist_ok=True)
    p = Path(GHO_TEXT_DIR) / f"gho_{country}.md"
    p.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return p


def index_gho_texts(lines: List[str], country: str = "VNM") -> Dict[str, int]:
    """Index cau textualized vao ca 4 strategy DB (ids on dinh de rerun replace)."""
    import hashlib, time
    try:
        from chunk_strategies import ChunkingStrategy
    except ImportError:
        from src.chunk_strategies import ChunkingStrategy
    stats: Dict[str, int] = {}
    try:
        from src.indexer import get_or_create_vectorstore, _resolve_index_db_dir
        from src.metadata_store import (upsert_file_and_chunks, get_chunk_hashes_for_file,
                                        find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    except ImportError:
        from indexer import get_or_create_vectorstore, _resolve_index_db_dir  # type: ignore
        from metadata_store import (upsert_file_and_chunks, get_chunk_hashes_for_file,  # type: ignore
                                    find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    for strategy in [ChunkingStrategy.STRUCTURE, ChunkingStrategy.SLIDING,
                     ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID_SECTION_SEMANTIC]:
        vdb = get_or_create_vectorstore(_resolve_index_db_dir(strategy.value))
        file_key = f"gho_text_{country}::{strategy.value}"
        # xoa vectors cu cua file_key nay (replace semantics)
        try:
            old = get_chunk_hashes_for_file(file_key)
            if old:
                vdb.delete(ids=find_vector_ids_for_chunk_hashes(old))
                delete_chunks_by_hashes(old)
        except Exception as e:
            log.warning(f"[gho] cleanup cu that bai: {e}")
        texts, metas, ids, rows = [], [], [], []
        for i, line in enumerate(lines):
            vid = f"gho_{country}_{i}"
            meta = {"source_id": "gho", "chunking_strategy": strategy.value,
                    "section_path": f"gho/{country}", "page_numbers": "",
                    "file_name": f"gho_{country}.md", "country": country}
            texts.append(line)
            metas.append({k: (v if isinstance(v, str) else str(v)) for k, v in meta.items()})
            ids.append(vid)
            rows.append({
                "chunk_hash": hashlib.sha256(f"{line}gho/{country}{i}".encode()).hexdigest(),
                "file_name": file_key, "chunk_index": i, "vector_id": vid,
                "extra_meta": json.dumps(meta, ensure_ascii=False),
            })
        if texts:
            vdb.add_texts(texts=texts, metadatas=metas, ids=ids)
            upsert_file_and_chunks(file_key, f"gho-{country}-{int(time.time())}", rows)
        stats[strategy.value] = len(texts)
    return stats


def run_snapshot(countries: tuple = ("VNM",), force: bool = False) -> Dict[str, Any]:
    """Pipeline quarterly: fetch -> ETL -> snapshot JSON + SQLite + textualize (+index)."""
    run_id = create_run("gho_diabetes_snapshot")
    result: Dict[str, Any] = {"snapshots": [], "sqlite_rows": 0, "lines": []}
    try:
        all_records: List[Dict[str, Any]] = []
        for code in GHO_INDICATORS:
            rows = fetch_indicator(code, countries)
            recs = etl_records(code, rows)
            if not recs and not force:
                log.warning(f"[gho] {code}: khong co ban ghi")
                continue
            p = save_snapshot(code, recs)
            result["snapshots"].append(str(p))
            all_records.extend(recs)
        result["sqlite_rows"] = upsert_sqlite(all_records)
        lines = textualize(all_records)
        save_texts(lines, countries[0])
        result["lines"] = lines
        try:
            result["indexed"] = index_gho_texts(lines, countries[0])
        except Exception as e:
            log.warning(f"[gho] index textualized that bai (se chay lai): {e}")
            result["indexed"] = {}
        finish_run(run_id, found_new=len(all_records))
    except Exception as e:
        finish_run(run_id, status="failed", error=str(e)[:300])
        raise
    return result


# ---------- Tool Calling (Luong B): readonly, template co dinh ----------
def looks_like_stats_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in STATS_KEYWORDS)


def query_gho_stats(question: str, country: str = "VNM") -> Optional[str]:
    """Tra loi so lieu GHO bang SQL readonly (khong RAG so). Tra ve doan tieng Viet."""
    if not Path(GHO_DB_PATH).exists():
        return None
    conn = sqlite3.connect(f"file:{GHO_DB_PATH}?mode=ro", uri=True)
    try:
        out_lines = []
        for code, meta in GHO_INDICATORS.items():
            rows = conn.execute(
                "SELECT year, value FROM gho_stats WHERE indicator_code=? AND country=? AND sex='SEX_BTSX' ORDER BY year",
                (code, country),
            ).fetchall()
            if not rows:
                continue
            latest = rows[-1]
            line = f"- {meta.get('vi', code)} tại Việt Nam năm {latest[0]}: {latest[1]}%"
            if len(rows) >= 2:
                first = rows[0]
                delta = round(latest[1] - first[1], 2)
                trend = "tăng" if delta > 0 else "giảm"
                line += f" ({trend} {abs(delta)} điểm phần trăm so với năm {first[0]}: {first[1]}%)"
            out_lines.append(line)
    finally:
        conn.close()
    if not out_lines:
        return None
    return "Số liệu WHO (Global Health Observatory):\n" + "\n".join(out_lines)
