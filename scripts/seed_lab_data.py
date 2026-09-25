"""Seed lab reports mo phong (Nura B1) — demo-only, deterministic.

Reuse synthetic patient IDs san co (sidecar doctors_patients.json hoac
DISTINCT glucose user_ids). Moi BN 2-4 phieu/90d tai moc 30/60/90
(+45 khi can phieu thu 4), jitter +-2d deterministic.

Khong cham episodic/glucose/B4 (D4 pin). DB rieng metadata/labs.db.

Usage:
    python scripts/seed_lab_data.py --dry-run
    python scripts/seed_lab_data.py --reset --seed 42
"""
import argparse
import json
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import src.labs.store as _ls  # noqa: E402
import src.labs.lab_thresholds as _th  # noqa: E402
import src.shared.config as _cfg  # noqa: E402

FACILITIES = ["BV Noi tiet TW", "BV Bach Mai", "PK Da khoa khu vuc"]
ANCHOR_SETS = {2: [90, 30], 3: [90, 60, 30], 4: [90, 60, 45, 30]}


def _sidecar_dir() -> Path:
    override = os.getenv("SYNTHETIC_SIDECAR_DIR")
    if override:
        return Path(override)
    return Path(_cfg.BASE_DIR) / "metadata"


def get_patient_ids(limit: int = 1000) -> list:
    """Reuse IDs san co: sidecar truoc, DISTINCT glucose sau, fallback 30 IDs."""
    sidecar = _sidecar_dir() / "doctors_patients.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            ids = sorted(data.keys())
            if ids:
                return ids[:limit]
        except Exception:
            pass
    try:
        gdb = Path(_cfg.GLUCOSE_DB_PATH)
        if gdb.exists():
            conn = sqlite3.connect(str(gdb))
            try:
                rows = conn.execute(
                    "SELECT DISTINCT user_id FROM glucose_logs ORDER BY user_id LIMIT ?",
                    (limit,)).fetchall()
                ids = [r[0] for r in rows if r[0]]
                if ids:
                    return ids
            finally:
                conn.close()
    except Exception:
        pass
    return [f"syn_patient_{i + 1:03d}" for i in range(30)][:limit]


def _archetype_of(pid: str) -> str:
    sidecar = _sidecar_dir() / "doctors_patients.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            arch = (data.get(pid) or {}).get("archetype")
            if arch:
                return str(arch)
        except Exception:
            pass
    return ""


def gen_panel_values(archetype: str, rng: random.Random) -> dict:
    """Gia tri quanh ref; high-risk/dawn lech xau hon (deterministic)."""
    shift = 1.0
    if archetype == "high_risk_comorbid":
        shift = 1.25
    elif archetype == "dawn_phenomenon":
        shift = 1.12
    vals = {
        "hba1c": round(rng.uniform(5.5, 7.0) * shift, 1),
        "fpg": round(rng.uniform(95, 135) * shift, 1),
        "tg": round(rng.uniform(110, 190), 1),
        "tc": round(rng.uniform(170, 220), 1),
        "ldl": round(rng.uniform(85, 130), 1),
        "hdl": round(rng.uniform(38, 60), 1),
        "creatinine": round(rng.uniform(0.7, 1.1), 2),
        "egfr": round(rng.uniform(70, 105), 1),
        "ast": round(rng.uniform(18, 38), 1),
        "alt": round(rng.uniform(18, 45), 1),
    }
    # ~12% dot co 1 chi so xau ro (trend demo), khong cham panic.
    if rng.random() < 0.12:
        pick = rng.choice(["hba1c", "fpg", "ldl", "tg"])
        vals[pick] = {"hba1c": round(rng.uniform(8.0, 9.5), 1),
                      "fpg": round(rng.uniform(160, 220), 1),
                      "ldl": round(rng.uniform(150, 180), 1),
                      "tg": round(rng.uniform(220, 320), 1)}[pick]
    return vals


def build_lab_plan(patient_ids: list, seed: int) -> list:
    """[(report_id, user_id, sampled_at, facility, visit_no, {key: value})]."""
    rng = random.Random(seed)
    today = datetime.now(timezone.utc).date()
    plan = []
    for idx, pid in enumerate(patient_ids):
        n = rng.randint(2, 4)
        anchors = ANCHOR_SETS[n]
        arch = _archetype_of(pid)
        for j, days_ago in enumerate(sorted(anchors, reverse=True)):
            jitter = rng.randint(-2, 2)
            day = today - timedelta(days=days_ago + jitter)
            sampled = datetime(day.year, day.month, day.day, 7, 0).isoformat()
            rid = f"{pid}-lab-{day.isoformat()}"
            plan.append((rid, pid, sampled, FACILITIES[(idx + j) % len(FACILITIES)],
                         f"V{idx + 1:04d}-{j + 1}", gen_panel_values(arch, rng)))
    return plan


def seed_all(seed=42, reset=False, dry_run=False, db_path=None,
             patient_ids: list | None = None) -> dict:
    pids = patient_ids if patient_ids is not None else get_patient_ids()
    plan = build_lab_plan(pids, seed)
    n_obs = len(plan) * len(_th.PANEL)
    result = {"patients": len(pids), "reports": len(plan),
              "observations": n_obs, "dry_run": dry_run}
    if dry_run:
        return result
    target = _ls._ensure_db(db_path)
    conn = sqlite3.connect(str(target))
    try:
        existing = conn.execute(
            "SELECT COUNT(*) FROM lab_reports WHERE user_id LIKE 'syn_patient_%'").fetchone()
        n_existing = int(existing[0]) if existing else 0
    finally:
        conn.close()
    if n_existing and not reset:
        raise SystemExit(
            f"Found {n_existing} lab rows — re-run with --reset (scoped to syn_patient_%).")
    if reset:
        conn = sqlite3.connect(str(target))
        try:
            conn.execute("DELETE FROM lab_observations WHERE report_id IN"
                         " (SELECT report_id FROM lab_reports WHERE user_id LIKE 'syn_patient_%')")
            conn.execute("DELETE FROM lab_reports WHERE user_id LIKE 'syn_patient_%'")
            conn.commit()
        finally:
            conn.close()
        result["wiped"] = n_existing
    for rid, pid, sampled, fac, visit, vals in plan:
        _ls.create_report(rid, pid, sampled, fac, visit, db_path=db_path)
        for key, value in vals.items():
            spec = _th.PANEL[key]
            _ls.add_observation(rid, spec["loinc"], spec["name"], value, spec["unit"],
                                spec["ref_low"], spec["ref_high"],
                                _th.flag_value(key, value), db_path=db_path)
    result["inserted_reports"] = len(plan)
    result["inserted_observations"] = n_obs
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Seed lab reports (demo only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--db-path", default=None)
    args = p.parse_args()
    if args.dry_run:
        r = seed_all(args.seed, dry_run=True)
        print(f"[dry-run] patients={r['patients']} reports={r['reports']} "
              f"observations={r['observations']}")
        return
    r = seed_all(args.seed, reset=args.reset, db_path=args.db_path)
    print(f"Seeded {r['inserted_reports']} reports / {r['inserted_observations']} observations "
          f"for {r['patients']} patients (seed={args.seed})")


if __name__ == "__main__":
    main()
