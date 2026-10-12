"""DEPRECATED — use: python scripts/seed_demo_users.py --force [--password ...]."""
import runpy
import sys
from pathlib import Path

args = sys.argv[1:]
fwd = ["seed_demo_users.py", "--force"]
if "--password" in args:
    fwd += ["--password", args[args.index("--password") + 1]]
print("DEPRECATED seed_admin.py -> seed_demo_users.py --force")
sys.argv = fwd
runpy.run_path(str(Path(__file__).resolve().parent / "seed_demo_users.py"), run_name="__main__")
