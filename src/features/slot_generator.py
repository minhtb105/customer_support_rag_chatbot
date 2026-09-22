"""Deterministic slot mock + roster matcher for the triage demo (pure, no DB).

- Parses free-text working_hours (seed-controlled formats) -> open weekdays.
- Fixed 30-min slots over a dynamic 7-day window from today.
- Busy subset via hashlib (NEVER builtin hash()) + DEMO_BUSY env override.
- Matcher fallback chain: requested -> PCP -> same specialty ->
  Endocrinology_General -> any free. Doctor names/ids always loaded from
  the real roster (never hardcoded).
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VI_LABEL = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]  # T2=Mon ... T7=Sat, CN=Sun

# NLU specialty codes -> roster specialty strings (seed-controlled).
SPECIALTY_MAP = {
    "Endocrinology_General": "Nội tiết chung",
    "Endocrinology_Complication": "Biến chứng ĐTĐ",
    "Nutrition_Lifestyle": "Dinh dưỡng & Lối sống",
}
GENERAL_SPECIALTY = "Nội tiết chung"

# Fail-open fallback when no sidecar exists (ids/specialties mirror seed
# order 2/1/2; names fall back to doctor_id — never invented names).
_FALLBACK_ORDER = [
    ("syn_doctor_01", "Nội tiết chung"),
    ("syn_doctor_02", "Nội tiết chung"),
    ("syn_doctor_03", "Biến chứng ĐTĐ"),
    ("syn_doctor_04", "Dinh dưỡng & Lối sống"),
    ("syn_doctor_05", "Dinh dưỡng & Lối sống"),
]
_FALLBACK_HOURS = "Mon-Fri 08:00-17:00"


def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


def expand_day_range(token: str) -> List[int]:
    """'Mon-Fri' -> [0..4]; 'Sat' -> [5]. Unknown -> []."""
    token = (token or "").strip()
    if "-" in token:
        a, b = [t.strip().title()[:3] for t in token.split("-", 1)]
        if a in DAY_ABBR and b in DAY_ABBR:
            i, j = DAY_ABBR.index(a), DAY_ABBR.index(b)
            if i <= j:
                return list(range(i, j + 1))
            return list(range(i, 7)) + list(range(0, j + 1))
        return []
    t = token.strip().title()[:3]
    return [DAY_ABBR.index(t)] if t in DAY_ABBR else []


def parse_working_hours(hours: str) -> Dict[int, List[tuple]]:
    """Free-text hours -> {weekday_idx: [(start_min, end_min)]}.

    Handles seed formats e.g. 'Mon-Fri 08:00-17:00' and
    'Mon-Fri 08:00-17:00; Sat 08:00-12:00'. Unparseable -> Mon-Fri 08-17.
    """
    out: Dict[int, List[tuple]] = {}
    try:
        for seg in str(hours or "").split(";"):
            seg = seg.strip()
            if not seg:
                continue
            days: List[int] = []
            for m in re.finditer(
                r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*(?:-\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun))?",
                seg,
                re.IGNORECASE,
            ):
                token = m.group(0).replace(" ", "")
                days.extend(expand_day_range(token))
            times = re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", seg)
            if days and times:
                for day in days:
                    for h1, m1, h2, m2 in times:
                        out.setdefault(day, []).append(
                            (int(h1) * 60 + int(m1), int(h2) * 60 + int(m2))
                        )
    except Exception:
        pass
    if not out:  # fail-open default
        out = {d: [(8 * 60, 17 * 60)] for d in range(5)}
    return out


def generate_slots(
    hours: str, days_ahead: int = 7, start: Optional[date] = None, slot_minutes: int = 30
) -> List[Dict[str, str]]:
    """Fixed slots for the next `days_ahead` days (dynamic weekdays)."""
    if start is None:
        start = date.today()
    open_map = parse_working_hours(hours)
    slots: List[Dict[str, str]] = []
    for i in range(max(1, days_ahead)):
        day = start + timedelta(days=i)
        for s, e in open_map.get(day.weekday(), []):
            t = s
            while t + slot_minutes <= e:
                slots.append(
                    {
                        "date": day.isoformat(),
                        "weekday": VI_LABEL[day.weekday()],
                        "time": f"{t // 60:02d}:{t % 60:02d}",
                    }
                )
                t += slot_minutes
    return slots


def _demo_busy_set() -> set:
    """DEMO_BUSY='doctor_id|YYYY-MM-DD|HH:MM, ...' forced-busy set (fail-open)."""
    try:
        raw = os.getenv("DEMO_BUSY", "") or ""
        return {p.strip() for p in raw.split(",") if p.strip().count("|") == 2}
    except Exception:
        return set()


def is_slot_busy(doctor_id: str, date_iso: str, time_str: str) -> bool:
    """Deterministic busy check: DEMO_BUSY override, else hashlib (stable)."""
    if f"{doctor_id}|{date_iso}|{time_str}" in _demo_busy_set():
        return True
    digest = hashlib.md5(f"{doctor_id}|{date_iso}|{time_str}".encode("utf-8")).digest()
    return digest[0] % 10 < 3  # ~30% busy


def free_slots(doctor: Dict[str, Any], excluded: Optional[set] = None,
               days_ahead: int = 7) -> List[Dict[str, str]]:
    """Slots minus busy minus excluded weekdays (English full names)."""
    excluded = excluded or set()
    out = []
    for s in generate_slots(str(doctor.get("working_hours") or ""), days_ahead=days_ahead):
        wd_full = DAY_FULL[DAY_ABBR.index(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                VI_LABEL.index(s["weekday"])
            ]
        )] if s["weekday"] in VI_LABEL else ""
        if wd_full in excluded:
            continue
        if is_slot_busy(str(doctor.get("doctor_id")), s["date"], s["time"]):
            continue
        out.append(s)
    return out


def load_doctors() -> List[Dict[str, Any]]:
    """Real roster first; fail-open fallback (no invented names)."""
    try:
        try:
            from src.features.synthetic_roster import _all_doctors
        except ImportError:
            from features.synthetic_roster import _all_doctors  # type: ignore
        doctors = [d for d in (_all_doctors() or []) if d.get("doctor_id")]
        if doctors:
            return doctors
    except Exception:
        pass
    return [
        {"doctor_id": did, "name": did, "specialty": spec, "working_hours": _FALLBACK_HOURS}
        for did, spec in _FALLBACK_ORDER
    ]


def resolve_doctor(query_text: str, doctors: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Fuzzy requested-doctor -> doctor_id via roster names/ids (accent-insensitive)."""
    text = _unaccent((query_text or "").lower())
    if not text.strip():
        return None
    for d in doctors or load_doctors():
        did = str(d.get("doctor_id") or "")
        name = str(d.get("name") or "")
        if did and did.lower() in (query_text or "").lower():
            return did
        nl = _unaccent(name.lower())
        if nl and nl in text:
            return did
        parts = nl.replace("bs.", "").strip().split()
        if len(parts) >= 2 and " ".join(parts[-2:]) in text:
            return did
    return None


def match_doctors(
    requested_doctor_id: Optional[str] = None,
    specialty_code: Optional[str] = None,
    excluded_weekdays: Optional[set] = None,
    pcp_doctor_id: Optional[str] = None,
    days_ahead: int = 7,
    per_doctor: int = 3,
    max_doctors: int = 3,
) -> List[Dict[str, Any]]:
    """Fallback chain: requested -> PCP -> same specialty -> General -> any free."""
    doctors = load_doctors()
    by_id = {str(d.get("doctor_id")): d for d in doctors}
    wanted_spec = SPECIALTY_MAP.get(specialty_code or "", "")
    ordered: List[Dict[str, Any]] = []

    def _push(did: Optional[str]):
        if did and did in by_id and all(
            str(x.get("doctor_id")) != did for x in ordered
        ):
            ordered.append(by_id[did])

    _push(requested_doctor_id)
    _push(pcp_doctor_id)
    if wanted_spec:
        for d in doctors:
            if d.get("specialty") == wanted_spec and all(
                str(x.get("doctor_id")) != str(d.get("doctor_id")) for x in ordered
            ):
                ordered.append(d)
    if wanted_spec != GENERAL_SPECIALTY:
        for d in doctors:
            if d.get("specialty") == GENERAL_SPECIALTY and all(
                str(x.get("doctor_id")) != str(d.get("doctor_id")) for x in ordered
            ):
                ordered.append(d)
    for d in doctors:
        if all(str(x.get("doctor_id")) != str(d.get("doctor_id")) for x in ordered):
            ordered.append(d)

    out: List[Dict[str, Any]] = []
    for d in ordered:
        slots = free_slots(d, excluded_weekdays, days_ahead)[:per_doctor]
        if slots:
            out.append(
                {
                    "doctor_id": d.get("doctor_id"),
                    "name": d.get("name"),
                    "specialty": d.get("specialty"),
                    "slots": slots,
                }
            )
        if len(out) >= max_doctors:
            break
    if not out and ordered:  # never return empty when doctors exist
        d = ordered[0]
        out.append(
            {"doctor_id": d.get("doctor_id"), "name": d.get("name"),
             "specialty": d.get("specialty"), "slots": []}
        )
    return out
