"""
Wrapper chay GHO snapshot quarterly — Dual-Storage (Chroma textualized + SQLite tool).

Su dung:
  python scripts/fetch_gho_snapshot.py
  python scripts/fetch_gho_snapshot.py --countries VNM --force
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitors.gho_snapshot import run_snapshot


def main():
    parser = argparse.ArgumentParser(description="Fetch WHO GHO diabetes snapshot (VNM)")
    parser.add_argument("--countries", default="VNM", help="Comma-separated ISO codes")
    parser.add_argument("--force", action="store_true", help="Luu snapshot du rỗng")
    args = parser.parse_args()
    countries = tuple(c.strip() for c in args.countries.split(",") if c.strip())
    stats = run_snapshot(countries=countries, force=args.force)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
