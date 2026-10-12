"""Shared text helpers (dedup Group B)."""
from __future__ import annotations

import unicodedata


def unaccent(s: str) -> str:
    t = "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )
    return t.replace("đ", "d").replace("Đ", "D").lower()


def hit(text: str, kws) -> bool:
    lowered = (text or "").lower()
    plain = unaccent(lowered)
    return any(kw in lowered or unaccent(kw) in plain for kw in kws)
