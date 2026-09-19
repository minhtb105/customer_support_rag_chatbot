"""Seed admin CLI: python scripts/seed_admin.py --username admin --password Admin@123 --email admin@example.com"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.db import get_user_by_username, create_user, init_auth_db
from src.auth.security import hash_password

def main():
    parser = argparse.ArgumentParser(description="Create admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--email", default=None)
    parser.add_argument("--full-name", dest="full_name", default="Administrator")
    parser.add_argument("--force", action="store_true", help="if exists, reset password/role to admin")
    args = parser.parse_args()
    init_auth_db()
    existing = get_user_by_username(args.username)
    if existing:
        if not args.force:
            print(f"User {args.username} already exists (role={existing['role']}) — use --force to reset")
            return
        from src.auth.db import update_user
        hashed = hash_password(args.password)
        u = update_user(existing["id"], hashed_password=hashed, role="admin", is_active=1, is_verified=1, email=args.email or existing.get("email"))
        print(f"Updated user {args.username} to admin: {u}")
        return
    hashed = hash_password(args.password)
    user = create_user(username=args.username, email=args.email, hashed_password=hashed, full_name=args.full_name, role="admin", is_verified=True)
    print(f"Created admin {user['username']} id={user['id']}")

if __name__ == "__main__":
    main()
