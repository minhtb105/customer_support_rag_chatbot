"""Compat shim — use scripts/reembed.py --provider nvidia."""
from reembed import main
if __name__ == "__main__":
    main(["--provider", "nvidia"])
