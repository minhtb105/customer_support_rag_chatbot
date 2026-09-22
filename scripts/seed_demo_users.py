"""Seed 3 demo accounts for the demo-video recorder (demo-only credentials).

Usage:
    python scripts/seed_demo_users.py --force
    python scripts/seed_demo_users.py --force --password "Some@123"

Creates (or resets with --force):
    demo_patient_01 (role=user)
    demo_doctor_01  (role=doctor, is_verified=1 so require_expert passes)
    demo_admin_01   (role=admin,  is_verified=1 so require_admin passes)

Password defaults to env DEMO_PASSWORD, fallback "Demo@123".
WARNING demo-only: change the password when hosting publicly.
Glucose baseline is NOT here — use scripts/seed_glucose_data.py --reset.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.db import get_user_by_username, create_user, update_user, init_auth_db
from src.auth.security import hash_password

DEFAULT_PASSWORD = os.environ.get("DEMO_PASSWORD", "Demo@123")

ACCOUNTS = [
    ("demo_patient_01", "user", 1),
    ("demo_doctor_01", "doctor", 1),
    ("demo_admin_01", "admin", 1),
]


def main():
    parser = argparse.ArgumentParser(description="Seed demo users (demo-only)")
    parser.add_argument("--force", action="store_true",
                        help="reset password/role/verified if user exists")
    parser.add_argument("--password", default=DEFAULT_PASSWORD,
                        help="demo password (default env DEMO_PASSWORD or Demo@123)")
    args = parser.parse_args()
    print("WARNING demo-only credentials — change the password when hosting publicly.")
    init_auth_db()
    for username, role, verified in ACCOUNTS:
        existing = get_user_by_username(username)
        if existing:
            if not args.force:
                print(f"User {username} already exists (role={existing['role']}) — use --force to reset")
                continue
            u = update_user(existing["id"], hashed_password=hash_password(args.password),
                            role=role, is_active=1, is_verified=verified)
            print(f"{username} role={u['role']} id={u['id']} (reset)")
            continue
        u = create_user(username=username, email=None, hashed_password=hash_password(args.password),
                        full_name=username, role=role, is_verified=bool(verified))
        print(f"{username} role={u['role']} id={u['id']} (created)")


if __name__ == "__main__":
    main()
