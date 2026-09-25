"""Lab HITL tier-router (Nura B3, pure).

Mapping (D6 pins):
  T0 = auto answer, no review row
  T1 = admin (receptionist: booking/admin questions)
  T2 = pharmacist|specialist (drug-interaction -> pharmacist,
       mild-symptom -> specialist)
  T3 = doctor (dose-change/new-diagnosis + ALL panic cases)

Pure + deterministic, no-key safe. Panic flag always wins (fail-safe).
"""
from __future__ import annotations

import unicodedata
from typing import Dict, Optional

TIER_ROLES: Dict[str, Optional[str]] = {
    "T0": None,
    "T1": "admin",
    "T3": "doctor",
}

WHAT_HAPPENS_NEXT_VI: Dict[str, str] = {
    "T0": "AI trả lời ngay, không cần duyệt thêm.",
    "T1": "Câu hỏi đã chuyển tới lễ tân (admin), bạn sẽ nhận thông báo khi được phản hồi.",
    "T2": "Câu hỏi đã chuyển tới dược sĩ/chuyên gia, bạn sẽ nhận thông báo khi được duyệt.",
    "T3": "Câu hỏi đã chuyển tới bác sĩ, bạn sẽ nhận thông báo khi được duyệt. Nếu thấy nguy hiểm, gọi 115 ngay.",
}


def _unaccent(s: str) -> str:
    t = "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )
    return t.replace("đ", "d").replace("Đ", "D")


# NOTE: analyte names (creatinine/eGFR/AST/ALT/HbA1c...) are intentionally NOT
# here — labs carve-out: asking about kidney/liver values stays T0.
_T3_DOSE_DX = (
    "doi lieu", "tang lieu", "giam lieu", "chinh lieu",
    "ngung thuoc", "them thuoc", "doi thuoc", "bo thuoc",
    "chan doan", "benh moi", "mac them benh",
)
_T2_DRUG = (
    "tuong tac", "uong chung", "ket hop thuoc",
    "tac dung phu", "di ung thuoc",
)
_T2_MILD = (
    "te chan", "te tay", "mo mat nhe", "met moi nhe",
    "ho nhe", "nguoi met nhe", "chong mat nhe",
)
_T1_ADMIN = (
    "dat lich", "dat hen", "lich kham", "lich hen", "booking",
    "thu tuc", "giay to", "bao hiem y te", "vien phi",
    "gio kham", "dia chi", "phong kham", "hanh chinh",
)


def route_tier(question: str, panic: bool = False) -> Dict[str, Optional[str]]:
    """Pure mapping -> {tier, routed_role, reason}. Panic always T3."""
    if panic:
        return {"tier": "T3", "routed_role": "doctor",
                "reason": "panic: chi so vuot nguong nguy hiem"}
    lowered = (question or "").lower()
    plain = _unaccent(lowered)

    def _has(kws) -> bool:
        return any(kw in lowered or _unaccent(kw) in plain for kw in kws)

    if _has(_T3_DOSE_DX):
        return {"tier": "T3", "routed_role": "doctor",
                "reason": "doi lieu/chan doan moi can bac si"}
    if _has(_T2_DRUG):
        return {"tier": "T2", "routed_role": "pharmacist",
                "reason": "tuong tac thuoc can duoc si"}
    if _has(_T2_MILD):
        return {"tier": "T2", "routed_role": "specialist",
                "reason": "trieu chung nhe can chuyen gia"}
    if _has(_T1_ADMIN):
        return {"tier": "T1", "routed_role": "admin",
                "reason": "dat lich/hanh chinh can le tan"}
    return {"tier": "T0", "routed_role": None, "reason": "cau hoi thuong, AI tra loi ngay"}


def what_happens_next(tier: str) -> str:
    return WHAT_HAPPENS_NEXT_VI.get(tier, WHAT_HAPPENS_NEXT_VI["T0"])
