"""Seed 90-day mock glucose baseline for the 3-5 minute demo video.

Baseline only (days 1..89 of a 90-day window ending 3 days ago).
The 3-record cascade (e.g. 110 -> 125 -> 142 fasting, ~+13%/day) is
NEVER in this seed — push it live at record time with explicit
``measured_at`` (today-2 / today-1 / today) so ``detect_trend`` fires
real-time on screen.

Usage:
    python scripts/seed_glucose_data.py --user demo_patient_01 --reset
    python scripts/seed_glucose_data.py --user demo_patient_01 --reset --dry-run
"""
import argparse
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.features.glucose_tracker import add_log, init_glucose_db
from src.config import GLUCOSE_DB_PATH

NOTES_POOL = ["ngủ ngon", "ăn bún chả", "hơi mệt"]


def existing_count(user_id: str) -> int:
    init_glucose_db()
    conn = sqlite3.connect(str(GLUCOSE_DB_PATH))
    try:
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM glucose_logs WHERE user_id=?", (user_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0
    finally:
        conn.close()


def build_plan(days: int, end_offset: int, seed: int):
    """Deterministic daily plan: 1 fasting + 1 post-meal + 30% extra random."""
    rng = random.Random(seed)
    end = datetime.now(timezone.utc).date() - timedelta(days=end_offset)
    start = end - timedelta(days=days - 1)
    plan = []
    for i in range(days):
        day = start + timedelta(days=i)
        plan.append((datetime(day.year, day.month, day.day, 7, 0), "fasting",
                     round(rng.uniform(95, 110), 1)))
        plan.append((datetime(day.year, day.month, day.day, 13, 0), "post_meal_2h",
                     round(rng.uniform(130, 160), 1)))
        if rng.random() < 0.30:
            plan.append((datetime(day.year, day.month, day.day, 20, 0), "random",
                         round(rng.uniform(100, 140), 1)))
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description="Seed mock glucose baseline (demo only)")
    p.add_argument("--user", default="demo_patient_01")
    p.add_argument("--days", type=int, default=89)
    p.add_argument("--end-offset", type=int, default=3,
                   help="seed ends this many days ago; cascade is pushed live")
    p.add_argument("--reset", action="store_true",
                   help="wipe this user's rows first (idempotent re-run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n_existing = existing_count(args.user)
    if args.dry_run:
        plan = build_plan(args.days, args.end_offset, args.seed)
        print(f"[dry-run] user={args.user} existing={n_existing} "
              f"would_insert={len(plan)} days={args.days} end_offset={args.end_offset}")
        return
    if n_existing and not args.reset:
        print(f"User {args.user} already has {n_existing} rows — "
              f"re-run with --reset to wipe-and-seed (scoped to this user only).")
        sys.exit(2)
    if args.reset:
        init_glucose_db()
        conn = sqlite3.connect(str(GLUCOSE_DB_PATH))
        conn.execute("DELETE FROM glucose_logs WHERE user_id=?", (args.user,))
        conn.commit()
        conn.close()
        print(f"Wiped {n_existing} rows for user={args.user}")
    plan = build_plan(args.days, args.end_offset, args.seed)
    n = 0
    for idx, (ts, ctx, val) in enumerate(plan):
        # ~10% of records carry a short VI note (distinct times keep ORDER BY stable)
        notes = NOTES_POOL[(idx // 10) % len(NOTES_POOL)] if idx % 10 == 0 else None
        add_log(user_id=args.user, value_mgdl=val, measured_at=ts, context=ctx, notes=notes)
        n += 1
    print(f"Seeded {n} logs for user={args.user} "
          f"({args.days} days ending {args.end_offset}d ago, db={GLUCOSE_DB_PATH})")


if __name__ == "__main__":
    main()
