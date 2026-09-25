"""Hybrid NLU for Smart Triage (deterministic regex first, LLM only with key).

Output schema has NO `emergency` key (LLM-proof at type level — the
emergency path short-circuits in the endpoint BEFORE this module is even
imported). Keys: {symptoms, urgency, specialty, time_constraints
(excluded_weekdays + optional excluded_window), requested_doctor,
requested_doctors, has_bhyt, branch, tier, nearby, simulation, constraints,
intents}.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

# T2=Mon ... T7=Sat, CN=Sun (T4 = Wednesday!).
VN_DAY = {"t2": 0, "t3": 1, "t4": 2, "t5": 3, "t6": 4, "t7": 5, "cn": 6}
DAY_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SPECIALTIES = ("Endocrinology_General", "Endocrinology_Complication", "Nutrition_Lifestyle")

# (code, specialty, keywords matched on lowered raw OR unaccented text)
_SYMPTOMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("neuropathy", "Endocrinology_Complication",
     ("tê rần", "te ran", "tê bì", "te bi", "ngón chân", "ngon chan",
      "bàn chân", "ban chan", "kim châm", "kim cham", "tê tay", "te tay")),
    ("retinopathy", "Endocrinology_Complication",
     ("mắt mờ", "mat mo", "mờ mắt", "mo mat", "nhìn mờ", "nhin mo")),
    ("hyperglycemia", "Endocrinology_General",
     ("khát nhiều", "khat nhieu", "tiểu nhiều", "tieu nhieu",
      "sụt cân", "sut can", "tiểu đêm", "tieu dem")),
    ("hypoglycemia_sym", "Endocrinology_General",
     ("chóng mặt", "chong mat", "run rẩy", "run ray", "đói lả", "doi la",
      "toát mồ hôi", "toat mo hoi")),
    ("gi_side_effect", "Endocrinology_General",
     ("cồn ruột", "con ruot", "buồn nôn", "buon non",
      "đầy bụng", "day bung", "đau bụng", "dau bung")),
)

_BOOKING_WORDS = (
    "đặt lịch", "dat lich", "lịch khám", "lich kham", "lịch", "lich",
    "khám", "kham", "gặp bác sĩ", "gap bac si", "hẹn", "hen",
)

# Dynamic Solver Routing extensions (deterministic regex, ±dấu tolerant).
_BHYT_WORDS = (
    "bhyt", "bảo hiểm y tế", "bao hiem y te", "bảo hiểm", "bao hiem",
)

_BRANCH_CENTRAL = (
    "bạch mai", "bach mai", "trung ương", "trung uong",
    "tuyến trung ương", "tuyen trung uong", "tuyến trên", "tuyen tren",
)

_BRANCH_DISTRICT = (
    "tuyến huyện", "tuyen huyen", "huyện", "huyen",
    "trạm y tế", "tram y te",
)

_NEARBY_WORDS = (
    "gần nhà", "gan nha", "gần nhất", "gan nhat",
    "gần đây", "gan day",
)

_SIM_WORDS = (
    "mô phỏng", "mo phong", "giả lập", "gia lap",
)

_EXCLUDE_TRIGGERS = (
    "trừ", "ngoại trừ", "tránh", "không khám", "nghỉ",
)


def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


def _extract_excluded(text: str) -> List[str]:
    """Weekday exclusions, e.g. 'tuần sau trừ T4' -> ['Wednesday']."""
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    excluded: List[str] = []
    for m in re.finditer(r"\bt\s*([2-7])\b", plain):
        excluded.append(DAY_FULL[int(m.group(1)) - 2])
    if re.search(r"\bcn\b", plain):
        excluded.append("Sunday")
    if not excluded:
        return []
    triggers = [_unaccent(t) for t in _EXCLUDE_TRIGGERS]
    if any(t in plain for t in triggers):
        return sorted(set(excluded))
    return []


def _extract_symptoms(text: str) -> List[Dict[str, str]]:
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    out: List[Dict[str, str]] = []
    for code, _spec, keywords in _SYMPTOMS:
        for kw in keywords:
            if kw in lowered or _unaccent(kw) in plain:
                out.append({"code": code, "text": kw})
                break
    return out


def _extract_intents(text: str, symptoms: list, requested: Optional[str],
                     excluded: list, simulation: bool = False) -> List[str]:
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    intents: List[str] = []
    if requested or excluded or any(
        w in lowered or _unaccent(w) in plain for w in _BOOKING_WORDS
    ):
        intents.append("booking")
    if symptoms:
        intents.append("symptom_note")
    if simulation:
        intents.append("simulation")
    return intents


def _extract_bhyt(text: str) -> bool:
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    return any(w in lowered or _unaccent(w) in plain for w in _BHYT_WORDS)


def _extract_branch_tier(text: str) -> tuple:
    """-> (branch_label|None, tier|None). Central checked first."""
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    for w in _BRANCH_CENTRAL:
        if w in lowered or _unaccent(w) in plain:
            return ("Bạch Mai", "central")
    for w in _BRANCH_DISTRICT:
        if w in lowered or _unaccent(w) in plain:
            return ("Tuyến huyện", "district")
    return (None, None)


def _extract_nearby(text: str) -> bool:
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    return any(w in lowered or _unaccent(w) in plain for w in _NEARBY_WORDS)


def _extract_window(text: str) -> Optional[Dict[str, str]]:
    """Excluded time window, e.g. 'trừ 10h-11h' -> {start,end}. Needs a trigger."""
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    triggers = [_unaccent(t) for t in ("trừ", "tránh", "ngoại trừ", "không", "ngoài")]
    if not any(t in plain for t in triggers):
        return None
    m = re.search(r"(\d{1,2})\s*h?\s*[-–—]\s*(\d{1,2})\s*h\b", plain)
    if not m:
        m = re.search(r"tu\s*(\d{1,2})\s*h?\s*den\s*(\d{1,2})\s*h\b", plain)
    if not m:
        return None
    try:
        s, e = int(m.group(1)), int(m.group(2))
    except Exception:
        return None
    if not (0 <= s < e <= 24):
        return None
    return {"start": f"{s:02d}:00", "end": f"{e:02d}:00"}


def _extract_doctors(text: str) -> List[str]:
    """ALL roster doctors mentioned (roster order). Same predicate as resolve_doctor."""
    try:
        try:
            from src.scheduling.slot_generator import load_doctors
        except ImportError:
            from scheduling.slot_generator import load_doctors  # type: ignore
        doctors = load_doctors()
    except Exception:
        return []
    lowered_raw = (text or "").lower()
    lowered = _unaccent(lowered_raw)
    out: List[str] = []
    for d in doctors:
        did = str(d.get("doctor_id") or "")
        name = str(d.get("name") or "")
        hit = False
        if did and did.lower() in lowered_raw:
            hit = True
        nl = _unaccent(name.lower())
        if not hit and nl and nl in lowered:
            hit = True
        parts = nl.replace("bs.", "").strip().split()
        if not hit and len(parts) >= 2 and " ".join(parts[-2:]) in lowered:
            hit = True
        if hit and did and did not in out:
            out.append(did)
    return out


def _extract_simulation(text: str, requested: Optional[str]) -> Dict[str, Any]:
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    if not any(w in lowered or _unaccent(w) in plain for w in _SIM_WORDS):
        return {"is_simulation": False, "doctor_id": None, "days_off": 0}
    days = 3
    m = re.search(r"nghi\s*(\d+)\s*ngay", plain) or re.search(r"(\d+)\s*ngay", plain)
    if m:
        try:
            days = max(1, int(m.group(1)))
        except Exception:
            days = 3
    return {"is_simulation": True, "doctor_id": requested, "days_off": days}


def _extract_constraints(has_bhyt: bool, branch, tier, requested_list: list,
                         nearby: bool, excluded: list,
                         window: Optional[Dict[str, str]]) -> List[str]:
    out: List[str] = []
    if has_bhyt:
        out.append("bhyt")
    if branch or tier:
        out.append("branch")
    if requested_list:
        out.append("doctor")
    if nearby:
        out.append("nearby")
    if excluded:
        out.append("weekday")
    if window:
        out.append("window")
    return out


def parse_regex(text: str) -> Dict[str, Any]:
    """Deterministic fallback (also the CI path when no OPENAI_API_KEY)."""
    symptoms = _extract_symptoms(text)
    spec = "Endocrinology_General"
    for s in symptoms:
        for code, code_spec, _kw in _SYMPTOMS:
            if s["code"] == code and code_spec != "Endocrinology_General":
                spec = code_spec
    requested_list = _extract_doctors(text)
    requested = requested_list[0] if requested_list else None
    excluded = _extract_excluded(text)
    has_bhyt = _extract_bhyt(text)
    branch, tier = _extract_branch_tier(text)
    nearby = _extract_nearby(text)
    window = _extract_window(text)
    simulation = _extract_simulation(text, requested)
    constraints = _extract_constraints(has_bhyt, branch, tier, requested_list,
                                       nearby, excluded, window)
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    urgency = "routine"
    if "gấp" in lowered or re.search(r"\bgap\b", plain):
        urgency = "urgent"
    elif spec == "Endocrinology_Complication":
        urgency = "soon"
    tc: Dict[str, Any] = {"excluded_weekdays": excluded}
    if window:
        tc["excluded_window"] = window
    return {
        "symptoms": symptoms,
        "urgency": urgency,
        "specialty": spec if spec in SPECIALTIES else "Endocrinology_General",
        "time_constraints": tc,
        "requested_doctor": requested,
        "requested_doctors": requested_list,
        "has_bhyt": has_bhyt,
        "branch": branch,
        "tier": tier,
        "nearby": nearby,
        "simulation": simulation,
        "constraints": constraints,
        "intents": _extract_intents(text, symptoms, requested, excluded,
                                    simulation["is_simulation"]),
    }


def _validate_llm_doctor(raw: str) -> Optional[str]:
    """Validate an LLM-provided doctor via the roster predicate (same as regex path).

    Returns the resolved doctor_id, or None when unresolvable (phantom fragment).
    """
    try:
        try:
            from src.scheduling.slot_generator import resolve_doctor
        except ImportError:
            from scheduling.slot_generator import resolve_doctor  # type: ignore
        return resolve_doctor(raw)
    except Exception:
        return None


def _parse_llm(text: str) -> Optional[Dict[str, Any]]:
    """gpt-4o-mini refinement — only when OPENAI_API_KEY is set. Never adds emergency."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            timeout=20,
        )
        schema_hint = (
            '{"symptoms": [{"code": str, "text": str}], '
            '"urgency": "routine|soon|urgent", '
            '"specialty": "Endocrinology_General|Endocrinology_Complication|Nutrition_Lifestyle", '
            '"time_constraints": {"excluded_weekdays": ["Monday"...]}, '
            '"requested_doctor": str|null, "intents": ["booking", "symptom_note"]}'
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content":
                 "Trích xuất ý định đặt lịch khám tiểu đường sang JSON đúng schema, tiếng Việt. "
                 "Ngày: T2=Monday...T7=Saturday, CN=Sunday. Chỉ trả JSON.\n" + schema_hint},
                {"role": "user", "content": text},
            ],
            temperature=0, max_tokens=400,
        )
        data = json.loads((resp.choices[0].message.content or "").strip())
        if not isinstance(data, dict):
            return None
        data.pop("emergency", None)  # belt-and-braces: schema forbids it
        if data.get("specialty") not in SPECIALTIES:
            data["specialty"] = "Endocrinology_General"
        data.setdefault("symptoms", [])
        data.setdefault("urgency", "routine")
        data.setdefault("time_constraints", {"excluded_weekdays": []})
        data.setdefault("requested_doctor", None)
        data.setdefault("intents", [])
        return data
    except Exception:
        return None


def parse_triage(text: str) -> Dict[str, Any]:
    """Regex first; LLM only refines when a key exists (never clears anything)."""
    base = parse_regex(text or "")
    llm = _parse_llm(text or "")
    if not llm:
        return base
    # Merge: LLM fills specialty/intents only when regex found nothing.
    if not base["symptoms"] and llm.get("symptoms"):
        base["symptoms"] = llm["symptoms"]
    if base["specialty"] == "Endocrinology_General" and llm.get("specialty"):
        base["specialty"] = llm["specialty"]
    if not base["intents"] and llm.get("intents"):
        base["intents"] = [i for i in llm["intents"]
                           if i in ("booking", "symptom_note")]
    if llm.get("urgency") in ("routine", "soon", "urgent") and base["urgency"] == "routine":
        base["urgency"] = llm["urgency"]
    tc = llm.get("time_constraints") or {}
    if not base["time_constraints"]["excluded_weekdays"] and tc.get("excluded_weekdays"):
        base["time_constraints"]["excluded_weekdays"] = [
            d for d in tc["excluded_weekdays"] if d in DAY_FULL
        ]
    if not base["requested_doctor"] and llm.get("requested_doctor"):
        validated = _validate_llm_doctor(str(llm["requested_doctor"]))
        if validated is not None:
            base["requested_doctor"] = validated
            if validated not in base["requested_doctors"]:
                base["requested_doctors"] = [validated, *base["requested_doctors"]]
            tc0 = base.get("time_constraints") or {}
            base["constraints"] = _extract_constraints(
                base.get("has_bhyt", False), base.get("branch"),
                base.get("tier"), base["requested_doctors"],
                base.get("nearby", False),
                tc0.get("excluded_weekdays") or [],
                tc0.get("excluded_window"),
            )
        # else: unresolvable phantom -> drop (leave None; no "doctor" constraint)
    return base
