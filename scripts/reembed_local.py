"""Compat shim — use scripts/reembed.py --provider local."""
from reembed import main
if __name__ == "__main__":
    main(["--provider", "local"])
