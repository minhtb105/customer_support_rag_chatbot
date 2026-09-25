"""Shared endpoint helpers used by multiple service routers."""
from __future__ import annotations

from fastapi import HTTPException

try:
    from src.shared.config import PDF_DIR
except ImportError:  # pragma: no cover
    from shared.config import PDF_DIR  # type: ignore


def _resolve_user_id(req_user_id: str, current_user: dict) -> str:
    """User thường chỉ được dùng own id, expert/admin được override"""
    role = current_user.get("role", "user")
    if role in ("admin", "doctor", "pharmacist", "specialist"):
        return req_user_id or current_user["id"]
    # user role: enforce own
    if req_user_id and req_user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden: user can only act on own user_id")
    return current_user["id"]


def _count_pdfs() -> tuple[int, dict]:
    if not PDF_DIR.exists():
        return 0, {}
    total = 0
    by_src: dict[str, int] = {}
    for sub in PDF_DIR.iterdir():
        if sub.is_dir():
            cnt = len(list(sub.glob("*.pdf")))
            if cnt:
                by_src[sub.name] = cnt
                total += cnt
        elif sub.suffix.lower() == ".pdf":
            total += 1
            by_src["_root"] = by_src.get("_root", 0) + 1
    return total, by_src
