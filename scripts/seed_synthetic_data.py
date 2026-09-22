"""Seed 30 synthetic patients + 5 doctors for the multi-patient/doctor demo.

Demo-only. Deterministic via --seed. Synthetic IDs need no auth accounts
(doctor queue uses DISTINCT user_ids with fail-open display names).

Usage:
    python scripts/seed_synthetic_data.py --dry-run
    python scripts/seed_synthetic_data.py --reset --seed 42
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

from src.features.glucose_tracker import init_glucose_db, classify_glucose  # noqa: E402
import src.features.glucose_tracker as _gt  # noqa: E402
import src.config as _cfg  # noqa: E402

ARCHETYPES = ["well_controlled"] * 10 + ["dawn_phenomenon"] * 10 + ["high_risk_comorbid"] * 10
N_PATIENTS = 30
N_DOCTORS = 5
N_HOSPITALS = 1

# Deterministic hospital coords (Hanoi area) for geo-routing demo.
HOSPITAL_COORDS = [
    (21.007, 105.841),
    (21.028, 105.854),
    (20.990, 105.810),
]

SLOTS = [
    (6, 0, "fasting"),
    (13, 0, "post_meal_2h"),
    (18, 0, "pre_meal"),
    (22, 0, "bedtime"),
]
VALID_CONTEXTS = {"fasting", "pre_meal", "post_meal_2h", "bedtime", "random"}

# D3: forced ramp-week values (recorder math 100 -> 114 (+14%) -> 126 (+10.5%),
# both inside the detect_trend 10-15% band). Applied to 3 consecutive fasting
# days for every dawn/high-risk patient so trend coverage survives drift.
RAMP_VALUES = (100.0, 114.0, 126.0)

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "features" / "modules" / "diabetes_vn.json"

DOCTOR_SPECS = [
    ("syn_doctor_01", "Nội tiết chung", "Mon-Fri 08:00-17:00"),
    ("syn_doctor_02", "Nội tiết chung", "Mon-Fri 08:00-17:00; Sat 08:00-12:00"),
    ("syn_doctor_03", "Biến chứng ĐTĐ", "Mon-Fri 08:00-17:00"),
    ("syn_doctor_04", "Dinh dưỡng & Lối sống", "Mon-Fri 08:00-17:00; Sat 08:00-12:00"),
    ("syn_doctor_05", "Dinh dưỡng & Lối sống", "Mon-Fri 08:00-17:00"),
]

COMORBIDITIES = {
    "well_controlled": [[], [], ["Thừa cân"]],
    "dawn_phenomenon": [["Rối loạn giấc ngủ"], [], ["Thừa cân"]],
    "high_risk_comorbid": [["Tăng huyết áp"], ["Tăng huyết áp", "Rối loạn lipid máu"], ["Tăng huyết áp", "Thừa cân"]],
}

# Fallback Vietnamese names when Faker is unavailable (CI-safe).
_FAMILY = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Đặng", "Bùi"]
_GIVEN_M = ["Văn An", "Minh Đức", "Quốc Bảo", "Thành Long", "Hữu Phước", "Tuấn Anh", "Đức Huy", "Gia Khiêm"]
_GIVEN_F = ["Thị Lan", "Thu Hà", "Minh Châu", "Quỳnh Anh", "Thanh Tâm", "Ngọc Diệp", "Phương Thảo", "Kim Ngân"]
_DOCTOR_NAMES = ["BS. Nguyễn Thu Hà", "BS. Trần Minh Đức", "BS. Lê Quốc Bảo", "BS. Phạm Quỳnh Anh", "BS. Hoàng Thanh Tâm"]


def _name_fn(seed: int):
    """Faker vi_VN when available, else deterministic fallback lists."""
    try:
        from faker import Faker
        fake = Faker("vi_VN")
        fake.seed_instance(seed)
        return lambda rng, gender: fake.name_female() if gender == "Nữ" else fake.name_male()
    except Exception:
        def fallback(rng, gender):
            pool = _GIVEN_F if gender == "Nữ" else _GIVEN_M
            return f"{rng.choice(_FAMILY)} {rng.choice(pool)}"
        return fallback


def _archetype_for(i: int, n: int) -> str:
    """Block thirds (well/dawn/high); identical to ARCHETYPES when n=30."""
    block = n // 3
    if i < block:
        return "well_controlled"
    if i < 2 * block:
        return "dawn_phenomenon"
    return "high_risk_comorbid"


def build_doctors(name_fn, n_doctors: int = 5, hospitals: int = 1) -> list:
    rng = random.Random(7)  # fixed sub-seed so doctor names don't shift with --seed
    doctors = []
    specs = [DOCTOR_SPECS[i % len(DOCTOR_SPECS)] for i in range(n_doctors)]
    hosp_list = [
        {"hospital_id": f"hosp_{h + 1:02d}", "name": f"Bệnh viện {h + 1:02d}",
         "lat": HOSPITAL_COORDS[h % len(HOSPITAL_COORDS)][0],
         "lon": HOSPITAL_COORDS[h % len(HOSPITAL_COORDS)][1]}
        for h in range(max(1, hospitals))
    ]
    for i in range(n_doctors):
        _base_id, specialty, hours = specs[i]
        doc_id = f"syn_doctor_{i + 1:02d}"
        hosp = hosp_list[i % len(hosp_list)]
        try:
            name = name_fn(rng, "Nữ" if i % 2 else "Nam")
            name = f"BS. {name}" if not name.startswith("BS.") else name
        except Exception:
            name = _DOCTOR_NAMES[i % len(_DOCTOR_NAMES)]
        doctors.append({"doctor_id": doc_id, "name": name, "specialty": specialty,
                        "working_hours": hours, "hospital_id": hosp["hospital_id"],
                        "lat": hosp["lat"], "lon": hosp["lon"]})
    return doctors, hosp_list


def build_patients(seed: int, name_fn, n_patients: int = 30, doctor_ids: list | None = None) -> dict:
    rng = random.Random(seed + 1000)
    if doctor_ids is None:
        # Legacy round-robin over the default 5 roster ids (byte-identical PCPs).
        doctor_ids = [f"syn_doctor_{(k % 5) + 1:02d}" for k in range(5)]
    rng_geo = random.Random(seed + 2000)
    patients = {}
    for i in range(n_patients):
        pid = f"syn_patient_{i + 1:03d}"
        archetype = _archetype_for(i, n_patients)
        gender = rng.choice(["Nam", "Nữ"])
        pcp = doctor_ids[i % len(doctor_ids)] if doctor_ids else DOCTOR_SPECS[i % len(DOCTOR_SPECS)][0]
        patients[pid] = {
            "display_name": name_fn(rng, gender),
            "age": rng.randint(35, 75),
            "gender": gender,
            "comorbidities": rng.choice(COMORBIDITIES[archetype]),
            "pcp_doctor_id": pcp,  # round-robin over roster
            "archetype": archetype,
            # D5: ~1/3 near hospital (<5km greedy non-empty), rest spread city-wide.
            "lat": round((21.007 + rng_geo.uniform(-0.02, 0.02)) if i % 3 == 0
                         else (21.01 + rng_geo.uniform(-0.25, 0.25)), 4),
            "lon": round((105.841 + rng_geo.uniform(-0.02, 0.02)) if i % 3 == 0
                         else (105.84 + rng_geo.uniform(-0.25, 0.25)), 4),
        }
    return patients


def gen_value(archetype: str, slot: str, rng: random.Random) -> float:
    """Base + noise + spikes per archetype (matches docs/SYNTHETIC_DATA.md)."""
    if archetype == "well_controlled":
        v = rng.uniform(*{"fasting": (95, 110), "post_meal_2h": (130, 160),
                           "pre_meal": (100, 140), "bedtime": (110, 150)}[slot])
        if rng.random() < 0.05:
            v = rng.uniform(260, 300)
    elif archetype == "dawn_phenomenon":
        if slot == "fasting":
            v = rng.uniform(185, 230) if rng.random() < 0.40 else rng.uniform(95, 115)
        else:
            v = rng.uniform(*{"post_meal_2h": (130, 160), "pre_meal": (100, 140), "bedtime": (110, 150)}[slot])
    else:  # high_risk_comorbid
        v = rng.uniform(*{"fasting": (100, 160), "post_meal_2h": (140, 220),
                           "pre_meal": (110, 180), "bedtime": (120, 190)}[slot])
        if rng.random() < 0.30:
            v = rng.uniform(260, 300) if rng.random() < 0.80 else rng.uniform(55, 68)
    return round(v, 1)


def build_log_plan(patients: dict, seed: int, days: int, end_offset: int,
                    trajectories: dict | None = None) -> list:
    """Deterministic (patient_id, timestamp, context, value) rows.

    Engine modulation (Synthea-method-inspired): daily values scatter around a
    weekly baseline that drifts <= +-2 mg/dL/week (cumulative per patient);
    dawn morning-spike rule stays inside gen_value untouched. One forced
    ramp-week per dawn/high-risk patient (RAMP_VALUES on 3 consecutive fasting
    days) reuses the B4 spike-week idea for guaranteed trend coverage.
    """
    rng = random.Random(seed)
    end = datetime.now(timezone.utc).date() - timedelta(days=end_offset)
    start = end - timedelta(days=days - 1)
    traj = trajectories if trajectories is not None else build_trajectories(
        patients, seed, days)
    pids = list(patients.keys())
    plan = []
    for i in range(days):
        day = start + timedelta(days=i)
        for pidx, pid in enumerate(pids):
            meta = patients[pid]
            cum = (traj.get(pid) or {}).get("weekly_cum") or [0.0]
            drift = cum[min(i // 7, len(cum) - 1)]
            for hour, minute, ctx in SLOTS:
                ts = datetime(day.year, day.month, day.day, hour, minute)
                plan.append((pid, ts, ctx, round(gen_value(meta["archetype"], ctx, rng) + drift, 1)))
    _apply_ramp_week(plan, patients, days)
    return plan


def build_trajectories(patients: dict, seed: int, days: int) -> dict:
    """Weekly engine run per patient -> {drift_cum, medication, complications}.

    Fail-open: module missing/broken -> {} (flat baseline, ramp still applies).
    """
    try:
        from src.features import disease_engine as _eng
    except ImportError:
        try:
            from features import disease_engine as _eng  # type: ignore
        except ImportError:
            return {}
    try:
        module = _eng.load_module(MODULE_PATH)
    except Exception:
        return {}
    weeks = max(1, (days + 6) // 7)
    out = {}
    for pidx, (pid, meta) in enumerate(patients.items()):
        try:
            engine = _eng.DiseaseEngine(module, seed=seed + pidx)
            # WARMUP=2: start -> base -> week consumes 2 hops before the
            # distributed week-loop can fire events; trim so cum aligns to weeks.
            res = engine.run({"archetype": meta["archetype"], "age": meta.get("age", 50)},
                             weeks=weeks + 2)
            cum, running = [], 0.0
            for rec in res["trajectory"][2:2 + weeks]:
                running += float(rec.get("drift_delta") or 0.0)
                cum.append(round(running, 2))
            ctx = res["ctx"]
            out[pid] = {"weekly_cum": cum or [0.0],
                        "medication": ctx.get("medication"),
                        "complications": list(ctx.get("conditions") or [])}
        except Exception:
            continue
    return out


def _apply_ramp_week(plan: list, patients: dict, days: int) -> None:
    """Override 3 consecutive fasting days per dawn/high-risk patient (in place)."""
    if days < 3:
        return
    fasting_idx: dict = {}
    for k, (pid, _ts, ctx, _v) in enumerate(plan):
        if ctx == "fasting":
            fasting_idx.setdefault(pid, []).append(k)
    for pidx, pid in enumerate(patients.keys()):
        if patients[pid]["archetype"] not in ("dawn_phenomenon", "high_risk_comorbid"):
            continue
        idxs = fasting_idx.get(pid) or []
        start = (pidx * 5) % max(1, len(idxs) - 2)
        for j, v in enumerate(RAMP_VALUES):
            if start + j < len(idxs):
                k = idxs[start + j]
                pid_, ts, ctx, _old = plan[k]
                plan[k] = (pid_, ts, ctx, v)


def _db_path(explicit=None) -> Path:
    return Path(explicit) if explicit else Path(_gt.GLUCOSE_DB_PATH)


def _sidecar_dir(explicit=None) -> Path:
    override = os.getenv("SYNTHETIC_SIDECAR_DIR")
    if explicit:
        return Path(explicit)
    if override:
        return Path(override)
    return Path(_cfg.BASE_DIR) / "metadata"


def _ensure_db(db_path=None) -> Path:
    """Ensure glucose_logs table exists on the TARGET db (not just global path)."""
    target = _db_path(db_path)
    try:
        from src.features.base_tracker import init_table
    except ImportError:
        from features.base_tracker import init_table  # type: ignore
    init_table(
        target,
        """
        CREATE TABLE IF NOT EXISTS glucose_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            value_mgdl REAL NOT NULL,
            measured_at TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'random',
            notes TEXT,
            classification TEXT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_glucose_user_time ON glucose_logs(user_id, measured_at)",
    )
    return target


def existing_synthetic_count(db_path=None) -> int:
    target = _ensure_db(db_path)
    conn = sqlite3.connect(str(target))
    try:
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM glucose_logs WHERE user_id LIKE 'syn_patient_%'").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0
    finally:
        conn.close()


def insert_logs_batch(plan: list, db_path=None) -> int:
    """Single-transaction batch insert (reuse classify_glucose, no core change)."""
    target = _ensure_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    rows = [(pid, v, ts.isoformat(), ctx, None, classify_glucose(v, ctx)[0], now)
            for pid, ts, ctx, v in plan]
    conn = sqlite3.connect(str(target))
    try:
        conn.executemany(
            "INSERT INTO glucose_logs (user_id, value_mgdl, measured_at, context, notes, classification, created_at)"
            " VALUES (?,?,?,?,?,?,?)", rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def write_sidecars(patients: dict, doctors: list, sidecar_dir=None, hospitals: list | None = None) -> tuple:
    d = _sidecar_dir(sidecar_dir)
    d.mkdir(parents=True, exist_ok=True)
    p1 = d / "doctors_patients.json"
    p2 = d / "synthetic_roster.json"
    p1.write_text(json.dumps(patients, ensure_ascii=False, indent=2), encoding="utf-8")
    roster: dict = {"doctors": doctors}
    if hospitals:
        roster["hospitals"] = hospitals
    p2.write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
    return p1, p2


def _phrase(text: str) -> str:
    """LLM phrasing when OPENAI_API_KEY exists, else deterministic template."""
    if not os.getenv("OPENAI_API_KEY"):
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                        timeout=15)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": "Diễn đạt lại ngắn gọn, tiếng Việt, giữ nguyên số liệu."},
                      {"role": "user", "content": text}],
            temperature=0.2, max_tokens=150)
        out = (resp.choices[0].message.content or "").strip()
        return out or text
    except Exception:
        return text


def _save_episodic(user_id: str, text: str, related_log_id, episodic=None) -> bool:
    # ponytail: skip heavy src.memory import when no object injected (notes row
    # already carries the data for SOAP S; episodic fills at runtime via API).
    if episodic is None:
        return False
    try:
        episodic.add_message("user", text, metadata={"user_id": user_id, "related_log_id": related_log_id})
        return True
    except Exception:
        return False  # fail-open


def run_b4(patients: dict, plan: list, db_path=None, use_llm=True, episodic=None) -> dict:
    """Pre-generate check-in notes for spike weeks (dawn + high-risk only).

    Dual-write: UPDATE notes on the spike log row (so SOAP S auto-fills) +
    episodic add_message only when an episodic object is injected
    (standalone skip avoids heavy src.memory model load; API fills it live).
    Deterministic templates when no OPENAI_API_KEY; never raises.
    """
    stats = {"notes": 0, "episodic": 0}
    try:
        # spike events per patient per ISO week (inserted row ids unknown pre-insert,
        # so locate rows back by (user_id, measured_at) after batch insert)
        events: dict = {}
        for pid, ts, ctx, v in plan:
            meta = patients[pid]
            if meta["archetype"] not in ("dawn_phenomenon", "high_risk_comorbid"):
                continue
            if v > 250 or v < 70 or (ctx == "fasting" and v > 180):
                monday = (ts.date() - timedelta(days=ts.weekday())).isoformat()
                events.setdefault(pid, {}).setdefault(monday, []).append((ts, ctx, v))
        conn = sqlite3.connect(str(_db_path(db_path)))
        try:
            for pid, weeks in events.items():
                top = sorted(weeks.items(), key=lambda kv: -len(kv[1]))[:2]  # ~1-2 weeks/patient
                for monday, evts in top:
                    ts, ctx, v = sorted(evts, key=lambda e: e[0])[-1]
                    raw = (f"Tuần {monday}: đường huyết {v} mg/dL ({ctx}) — "
                           f"bệnh nhân cho biết tối hôm trước đã ăn bánh ngọt.")
                    text = _phrase(raw) if use_llm else raw
                    row = conn.execute(
                        "SELECT id, notes FROM glucose_logs WHERE user_id=? AND measured_at=?",
                        (pid, ts.isoformat())).fetchone()
                    if not row:
                        continue
                    prev = row[1]
                    conn.execute("UPDATE glucose_logs SET notes=? WHERE id=?",
                                 (f"{prev} | {text}" if prev else text, row[0]))
                    stats["notes"] += 1
                    if _save_episodic(pid, text, row[0], episodic=episodic):
                        stats["episodic"] += 1
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    return stats


def apply_engine_annotations(patients: dict, trajectories: dict, db_path=None) -> int:
    """Meds timeline -> notes of latest log per patient (so SOAP S prefills).

    Complications already merged into sidecar comorbidities in seed_all.
    Fail-open, never raises. Returns patients annotated.
    """
    n = 0
    try:
        conn = sqlite3.connect(str(_db_path(db_path)))
    except Exception:
        return 0
    try:
        for pid, traj in trajectories.items():
            med = (traj or {}).get("medication")
            if not med:
                continue
            try:
                row = conn.execute(
                    "SELECT id, notes FROM glucose_logs WHERE user_id=? "
                    "ORDER BY measured_at DESC LIMIT 1", (pid,)).fetchone()
                if not row:
                    continue
                text = f"Đang dùng {med} (theo timeline điều trị)"
                conn.execute("UPDATE glucose_logs SET notes=? WHERE id=?",
                             (f"{row[1]} | {text}" if row[1] else text, row[0]))
                n += 1
            except Exception:
                continue
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return n


def seed_all(seed=42, days=90, end_offset=3, reset=False, dry_run=False,
             db_path=None, sidecar_dir=None, use_llm=True, episodic=None,
             n_patients: int = 30, n_doctors: int = 5, hospitals: int = 1,
             skip_b4: bool = False) -> dict:
    name_fn = _name_fn(seed)
    doctors, hosp_list = build_doctors(name_fn, n_doctors=n_doctors, hospitals=hospitals)
    patients = build_patients(seed, name_fn, n_patients=n_patients,
                              doctor_ids=[d["doctor_id"] for d in doctors])
    trajectories = build_trajectories(patients, seed, days)
    for pid, traj in trajectories.items():
        staged = (traj or {}).get("complications") or []
        if staged and pid in patients:
            patients[pid]["comorbidities"] = sorted(
                set(patients[pid]["comorbidities"]) | set(staged))
    plan = build_log_plan(patients, seed, days, end_offset, trajectories)
    result = {"patients": len(patients), "doctors": len(doctors),
              "logs": len(plan), "dry_run": dry_run,
              "hospitals": len(hosp_list)}
    if dry_run:
        return result
    n_existing = existing_synthetic_count(db_path)
    if n_existing and not reset:
        raise SystemExit(f"Found {n_existing} synthetic rows — re-run with --reset (scoped to syn_patient_%).")
    if reset:
        target = _ensure_db(db_path)
        conn = sqlite3.connect(str(target))
        conn.execute("DELETE FROM glucose_logs WHERE user_id LIKE 'syn_patient_%'")
        conn.commit()
        conn.close()
        result["wiped"] = n_existing
    result["inserted"] = insert_logs_batch(plan, db_path)
    result["engine_meds"] = apply_engine_annotations(patients, trajectories, db_path)
    p1, p2 = write_sidecars(patients, doctors, sidecar_dir, hospitals=hosp_list)
    result["sidecars"] = [str(p1), str(p2)]
    result["b4"] = {"notes": 0, "episodic": 0} if skip_b4 else run_b4(
        patients, plan, db_path, use_llm=use_llm, episodic=episodic)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Seed synthetic patients + doctors (demo only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--end-offset", type=int, default=3)
    p.add_argument("--n-patients", type=int, default=30)
    p.add_argument("--n-doctors", type=int, default=5)
    p.add_argument("--hospitals", type=int, default=1)
    p.add_argument("--skip-b4", action="store_true", help="skip LLM/template check-in notes (fast scale)")
    p.add_argument("--reset", action="store_true", help="wipe syn_patient_% rows first (idempotent)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="force deterministic templates even with OPENAI_API_KEY")
    args = p.parse_args()
    if args.dry_run:
        r = seed_all(args.seed, args.days, args.end_offset, dry_run=True,
                     n_patients=args.n_patients, n_doctors=args.n_doctors,
                     hospitals=args.hospitals)
        print(f"[dry-run] patients={r['patients']} doctors={r['doctors']} hospitals={r['hospitals']} "
              f"would_insert={r['logs']} days={args.days} end_offset={args.end_offset}")
        return
    r = seed_all(args.seed, args.days, args.end_offset, reset=args.reset, use_llm=not args.no_llm,
                 n_patients=args.n_patients, n_doctors=args.n_doctors,
                 hospitals=args.hospitals, skip_b4=args.skip_b4)
    print(f"Seeded {r['inserted']} logs for {r['patients']} patients + {r['doctors']} doctors "
          f"(seed={args.seed}, days={args.days}, end_offset={args.end_offset}, "
          f"notes={r['b4']['notes']}, episodic={r['b4']['episodic']})")


if __name__ == "__main__":
    main()
