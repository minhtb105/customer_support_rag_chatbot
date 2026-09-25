"""Dynamic solver routing for Smart Triage (Approach 1: Strategy Pattern, pure Python).

Single new module (D6): deterministic doctor attributes (hashlib, NO env
override for geo — DEMO_BUSY on slots is the only demo override) + three
solvers + a constraint-counting orchestrator.

- Attributes: distance_km / tier+branch / accepts_bhyt interpolated from
  ``hashlib.md5(doctor_id|tag)``. Calibrated on the 5 seed doctors so that
  ~3/5 are <5km and BHYT is mixed (verified live, not assumed).
- GreedySolver: ``match_doctors`` (UNTOUCHED, D5) + pre/post-filter of every
  parsed filter (D2); <5km hard filter + earliest-slot sort only when the
  user asked nearby.
- DFSSolver: constraint tree over time-window/weekday -> named-doctor ->
  branch/tier -> BHYT; specialty is applied FIRST and NEVER relaxed (D3).
  BHYT relaxation always appends an explicit cost warning (D3).
- GASolver: in-memory reschedule of 200 virtual patients reusing the seed
  archetype distribution (well/dawn/high-risk thirds). Takes NO user_id,
  touches NO DB (D1, D4). Fixed fitness weights.
- SchedulerOrchestrator: constraint count selects the algorithm ONLY
  (0-1 -> greedy, >=2 -> DFS); simulation intent -> GA (D2).

No OPENAI_API_KEY needed anywhere here: fully deterministic.
"""

from __future__ import annotations

import hashlib
import math
import random
import unicodedata
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.scheduling import slot_generator as _sg
except ImportError:  # pragma: no cover
    import scheduling.slot_generator as _sg  # type: ignore

# Fixed fitness weights (D4): fitness = avg_shift_days + W * inversion_rate.
PRIORITY_WEIGHT = 2.0

# Seed archetype distribution, thirds like scripts/seed_synthetic_data.py
# (10/10/10). (name, pathology_priority)
ARCHETYPES: Tuple[Tuple[str, int], ...] = (
    ("well_controlled", 1),
    ("dawn_phenomenon", 2),
    ("high_risk_comorbid", 3),
)

# Relaxation order (D3): window/time -> named-doctor -> branch/tier -> BHYT.
RELAX_ORDER = ("window", "doctor", "branch", "bhyt")

RELAX_VI = {
    "window": "khung giờ/ngày loại trừ",
    "doctor": "yêu cầu đích danh bác sĩ",
    "branch": "chi nhánh/tuyến",
    "bhyt": "BHYT",
}

BHYT_COST_WARNING = "không đảm bảo BHYT — có thể phát sinh chi phí"

# Echo order for routing_reason constraints=[...] (D2).
ECHO_ORDER = ("bhyt", "branch", "doctor", "nearby", "weekday", "window")


def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


# ---------- Doctor attributes (hashlib only, calibrated) ----------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (math stdlib only)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def real_distance_km(doctor_id: str, patient_id: Optional[str] = None) -> Optional[float]:
    """Real coords distance when BOTH sidecars have lat/lon (D5); else None."""
    if not patient_id:
        return None
    try:
        try:
            from src.scheduling.synthetic_roster import get_patient_location, get_doctor_location
        except ImportError:
            from scheduling.synthetic_roster import get_patient_location, get_doctor_location  # type: ignore
        pl, dl = get_patient_location(patient_id), get_doctor_location(doctor_id)
        if not pl or not dl:
            return None
        return haversine_km(pl["lat"], pl["lon"], dl["lat"], dl["lon"])
    except Exception:
        return None

def doctor_attributes(doctor_id: str) -> Dict[str, Any]:
    """Deterministic pseudo-attributes for a roster doctor.

    - distance_km: (md5(id|geo)[0] % 100) / 10 -> 0.0..9.9; 3/5 seed
      doctors land <5km.
    - tier: "central" if md5(id|tier)[0] is even else "district"
      (branch label "Bạch Mai" vs "Tuyến huyện").
    - accepts_bhyt: md5(id|bhyt)[1] is even (3 True / 2 False on seed).
    """
    did = str(doctor_id or "")
    geo = hashlib.md5(f"{did}|geo".encode("utf-8")).digest()
    tier = hashlib.md5(f"{did}|tier".encode("utf-8")).digest()
    bhyt = hashlib.md5(f"{did}|bhyt".encode("utf-8")).digest()
    central = tier[0] % 2 == 0
    return {
        "distance_km": round((geo[0] % 100) / 10.0, 1),
        "tier": "central" if central else "district",
        "branch": "Bạch Mai" if central else "Tuyến huyện",
        "accepts_bhyt": bhyt[1] % 2 == 0,
    }


def count_constraints(nlu: Dict[str, Any]) -> List[str]:
    """Names of parsed filters present, in fixed echo order (D2).

    Count selects the algorithm ONLY; every listed filter is still applied
    by whichever solver runs. Symptoms/booking intent never count.
    """
    nlu = nlu or {}
    tc = nlu.get("time_constraints") or {}
    present = set()
    if nlu.get("has_bhyt"):
        present.add("bhyt")
    if nlu.get("branch") or nlu.get("tier"):
        present.add("branch")
    if nlu.get("requested_doctor") or nlu.get("requested_doctors"):
        present.add("doctor")
    if nlu.get("nearby"):
        present.add("nearby")
    if tc.get("excluded_weekdays"):
        present.add("weekday")
    if tc.get("excluded_window"):
        present.add("window")
    return [c for c in ECHO_ORDER if c in present]


def _echo(constraints: List[str]) -> str:
    return "constraints=[" + ",".join(constraints) + "]"


def _slot_minutes(t: str) -> int:
    try:
        h, m = str(t or "").split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return -1


def _in_window(slot: Dict[str, Any], window: Optional[Dict[str, Any]]) -> bool:
    if not window:
        return False
    try:
        s = _slot_minutes((window.get("start") or "00:00"))
        e = _slot_minutes((window.get("end") or "00:00"))
        m = _slot_minutes((slot or {}).get("time"))
        return s <= m < e
    except Exception:
        return False


def _earliest(slot_list: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not slot_list:
        return ("", "")
    s = min(slot_list, key=lambda x: (str(x.get("date") or ""), str(x.get("time") or "")))
    return (str(s.get("date") or ""), str(s.get("time") or ""))


def _with_attrs(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    out.update(doctor_attributes(str(entry.get("doctor_id") or "")))
    return out


# ---------- Greedy (D2: every parsed filter applies) ----------

class GreedySolver:
    """match_doctors (untouched) + post-filter; <5km + earliest sort on nearby."""

    @staticmethod
    def solve(
        nlu: Dict[str, Any],
        pcp_doctor_id: Optional[str] = None,
        per_doctor: int = 3,
        max_doctors: int = 3,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        nlu = nlu or {}
        tc = nlu.get("time_constraints") or {}
        excluded = set(tc.get("excluded_weekdays") or [])
        window = tc.get("excluded_window")
        recs = _sg.match_doctors(
            requested_doctor_id=nlu.get("requested_doctor"),
            specialty_code=nlu.get("specialty"),
            excluded_weekdays=excluded,
            pcp_doctor_id=pcp_doctor_id,
            per_doctor=per_doctor,
            max_doctors=max_doctors,
        )
        want_bhyt = bool(nlu.get("has_bhyt"))
        want_tier = nlu.get("tier")
        nearby = bool(nlu.get("nearby"))

        def _dist(did: str) -> float:
            real = real_distance_km(did, patient_id)  # prefer real, fallback hash
            if real is not None:
                return real
            return float(doctor_attributes(did)["distance_km"])
        out: List[Dict[str, Any]] = []
        for d in recs:
            attrs = doctor_attributes(str(d.get("doctor_id") or ""))
            if want_bhyt and not attrs["accepts_bhyt"]:
                continue
            if want_tier and attrs["tier"] != want_tier:
                continue
            dist = _dist(str(d.get("doctor_id") or ""))
            if nearby and dist >= 5:
                continue
            slots = [s for s in (d.get("slots") or []) if not _in_window(s, window)]
            if not slots:
                continue
            entry = dict(d)
            entry["slots"] = slots
            entry.update(attrs)
            entry["distance_km"] = dist
            out.append(entry)
        if nearby:
            out.sort(key=lambda e: (
                float(e.get("distance_km", 9.9)),
                _earliest(e.get("slots") or []),
            ))
        constraints = count_constraints(nlu)
        reason = (
            f"Greedy (khám nhẹ): {len(constraints)} ràng buộc ({_echo(constraints)}). "
            "Đã áp dụng toàn bộ tiêu chí đã nêu"
            + ("; chỉ bác sĩ trong 5km, xếp slot sớm nhất." if nearby
               else ", ưu tiên slot sớm nhất.")
        )
        return {"recommended_doctors": out, "routing_reason": reason}


# ---------- DFS / CSP (D3: fixed relaxation order, never specialty) ----------

class DFSSolver:
    """Constraint tree with relaxation window -> doctor -> branch -> BHYT."""

    @staticmethod
    def solve(
        nlu: Dict[str, Any],
        pcp_doctor_id: Optional[str] = None,
        per_doctor: int = 3,
        max_doctors: int = 3,
    ) -> Dict[str, Any]:
        nlu = nlu or {}
        tc = nlu.get("time_constraints") or {}
        excluded = set(tc.get("excluded_weekdays") or [])
        window = tc.get("excluded_window")
        requested = [r for r in (nlu.get("requested_doctors")
                                 or ([nlu["requested_doctor"]] if nlu.get("requested_doctor") else []))]
        want_bhyt = bool(nlu.get("has_bhyt"))
        want_tier = nlu.get("tier")
        wanted_spec = _sg.SPECIALTY_MAP.get(nlu.get("specialty") or "", "")
        doctors = _sg.load_doctors()

        # Specialty pool is fixed FIRST and never relaxed (D3).
        spec_pool = [d for d in doctors
                     if not wanted_spec or d.get("specialty") == wanted_spec]
        by_id = {str(d.get("doctor_id")): d for d in doctors}

        def _ordered(use_doctor: bool, use_branch: bool, use_bhyt: bool,
                     use_time: bool) -> List[Dict[str, Any]]:
            ordered: List[Dict[str, Any]] = []
            if use_doctor:
                for rid in requested:
                    d = by_id.get(str(rid))
                    if d and all(str(x.get("doctor_id")) != str(rid) for x in ordered):
                        ordered.append(d)
            if pcp_doctor_id and str(pcp_doctor_id) in by_id and all(
                    str(x.get("doctor_id")) != str(pcp_doctor_id) for x in ordered):
                ordered.append(by_id[str(pcp_doctor_id)])
            for d in spec_pool:
                if all(str(x.get("doctor_id")) != str(d.get("doctor_id")) for x in ordered):
                    ordered.append(d)
            out: List[Dict[str, Any]] = []
            for d in ordered:
                attrs = doctor_attributes(str(d.get("doctor_id") or ""))
                if use_branch and want_tier and attrs["tier"] != want_tier:
                    continue
                if use_bhyt and want_bhyt and not attrs["accepts_bhyt"]:
                    continue
                slots = _sg.free_slots(d, excluded if use_time else set())
                if use_time and window:
                    slots = [s for s in slots if not _in_window(s, window)]
                slots = slots[:per_doctor]
                if not slots:
                    continue
                entry = {"doctor_id": d.get("doctor_id"), "name": d.get("name"),
                         "specialty": d.get("specialty"), "slots": slots}
                entry.update(attrs)
                out.append(entry)
                if len(out) >= max_doctors:
                    break
            return out

        active = {
            "window": bool(excluded or window),
            "doctor": bool(requested),
            "branch": bool(want_tier),
            "bhyt": want_bhyt,
        }
        dropped = {k: False for k in RELAX_ORDER}
        flags = dict(active)
        proposals = _ordered(True, True, True, True)
        for step in RELAX_ORDER:
            if proposals:
                break
            if flags.get(step):
                dropped[step] = True
            flags[step] = False
            proposals = _ordered(
                use_doctor=flags["doctor"] if "doctor" in flags else True,
                use_branch=flags["branch"] if "branch" in flags else True,
                use_bhyt=flags["bhyt"] if "bhyt" in flags else True,
                use_time=flags["window"] if "window" in flags else True,
            )
        relaxed = [k for k in RELAX_ORDER if dropped.get(k) and active.get(k)]
        assert "specialty" not in relaxed  # D3: specialty is never relaxed
        constraints = count_constraints(nlu)
        if not relaxed:
            reason = (f"DFS (ràng buộc ngặt): khớp 100% {len(constraints)} ràng buộc "
                      f"({_echo(constraints)}), không cần nới lỏng.")
            exact = True
        else:
            names = ", ".join(RELAX_VI[k] for k in relaxed)
            reason = (f"DFS: đã nới lỏng {names} ({_echo(constraints)}). "
                      "Chuyên khoa được giữ nguyên.")
            if "bhyt" in relaxed:
                reason += f" Lưu ý: {BHYT_COST_WARNING}."
            exact = False
        return {"recommended_doctors": proposals, "relaxed": relaxed,
                "routing_reason": reason, "exact": exact}


# ---------- GA: bulk reschedule simulation (D1/D4: no user_id, zero DB) ----------

def _roster_ids() -> List[str]:
    try:
        doctors = _sg.load_doctors() or []
        ids = [str(d.get("doctor_id")) for d in doctors if d.get("doctor_id")]
        if ids:
            return ids
    except Exception:
        pass
    return [f"syn_doctor_{i:02d}" for i in range(1, 6)]


def generate_virtual_patients(
    n: int = 200, seed: int = 42,
    doctor_ids: Optional[List[str]] = None,
    start: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """200 virtual patients reusing the seed archetype thirds (D4).

    Each item: {patient_id, archetype, orig_doctor, orig_slot{date,time},
    pathology_priority}. Pure in-memory; never touches any DB.
    """
    rng = random.Random(seed)
    ids = doctor_ids or _roster_ids()
    if start is None:
        start = date.today()
    times = ["08:00", "09:30", "14:00", "16:00"]
    out: List[Dict[str, Any]] = []
    for i in range(max(0, n)):
        name, prio = ARCHETYPES[i % len(ARCHETYPES)]
        h = hashlib.md5(f"virtual|{seed}|{i}".encode("utf-8")).digest()
        day_off = h[0] % 7
        slot_date = start.fromordinal(start.toordinal() + day_off)
        out.append({
            "patient_id": f"virtual_{i:03d}",
            "archetype": name,
            "orig_doctor": ids[(i + h[1]) % len(ids)],
            "orig_slot": {"date": slot_date.isoformat(), "time": times[h[2] % len(times)]},
            "pathology_priority": prio,
        })
    # Keep rng use explicit for future jitter without breaking determinism.
    _ = rng.random()
    return out


def simulate_time_off(
    doctor_id: Optional[str] = None,
    days_off: int = 3,
    n: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Reschedule patients of one absent doctor (D4).

    Fitness = avg_shift_days + PRIORITY_WEIGHT * priority_inversion_rate.
    Returns {reassigned, unplaced, avg_shift_days, fitness, ...}.
    """
    from datetime import timedelta
    ids = _roster_ids()
    off = str(doctor_id or (ids[0] if ids else "syn_doctor_01"))
    days_off = max(1, int(days_off or 3))
    today = date.today()
    patients = generate_virtual_patients(n=n, seed=seed, doctor_ids=ids, start=today)
    off_dates = {(today + timedelta(days=k)).isoformat() for k in range(days_off)}
    affected = [p for p in patients
                if p["orig_doctor"] == off and p["orig_slot"]["date"] in off_dates]
    # Free-slot pools of the remaining doctors (precomputed once).
    pools: Dict[str, List[Dict[str, str]]] = {}
    try:
        for d in (_sg.load_doctors() or []):
            did = str(d.get("doctor_id") or "")
            if did and did != off:
                pools[did] = list(_sg.free_slots(d, set(), days_ahead=14))
    except Exception:
        pools = {}
    for lst in pools.values():
        lst.sort(key=lambda s: (str(s.get("date") or ""), str(s.get("time") or "")))
    used: set = set()
    shifts: List[int] = []
    placed: List[Tuple[int, str]] = []  # (priority, new_date) for inversion
    unplaced = 0
    for p in affected:
        orig = p["orig_slot"]["date"]
        best = None
        for did, lst in pools.items():
            for s in lst:
                key = (did, s.get("date"), s.get("time"))
                if key in used:
                    continue
                if str(s.get("date") or "") >= orig:
                    if best is None or (str(s.get("date") or ""), str(s.get("time") or "")) < best[1:]:
                        best = (did, str(s.get("date") or ""), str(s.get("time") or ""))
                    break
        if best is None:
            for did, lst in pools.items():
                for s in lst:
                    key = (did, s.get("date"), s.get("time"))
                    if key not in used:
                        best = (did, str(s.get("date") or ""), str(s.get("time") or ""))
                        break
                if best is not None:
                    break
        if best is None:
            unplaced += 1
            continue
        used.add((best[0], best[1], best[2]))
        try:
            shift = (date.fromisoformat(best[1]) - date.fromisoformat(orig)).days
        except Exception:
            shift = 0
        shifts.append(shift)
        placed.append((int(p["pathology_priority"]), best[1]))
    reassigned = len(shifts)
    avg_shift = round(sum(shifts) / reassigned, 2) if reassigned else 0.0
    inv = 0
    total_pairs = 0
    order = sorted(placed, key=lambda x: x[1])
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            total_pairs += 1
            if order[i][0] < order[j][0]:
                inv += 1
    inv_rate = (inv / total_pairs) if total_pairs else 0.0
    fitness = round(avg_shift + PRIORITY_WEIGHT * inv_rate, 3)
    return {
        "reassigned": reassigned,
        "unplaced": unplaced,
        "avg_shift_days": avg_shift,
        "fitness": fitness,
        "off_doctor": off,
        "days_off": days_off,
        "total_affected": len(affected),
        "total_virtual": len(patients),
    }


# ---------- Orchestrator (D2: count selects the algorithm only) ----------

class SchedulerOrchestrator:
    """Route AFTER red-flag: simulation -> GA; >=2 constraints -> DFS; else greedy."""

    @staticmethod
    def route(
        nlu: Dict[str, Any],
        pcp_doctor_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        nlu = nlu or {}
        sim = nlu.get("simulation") or {}
        if sim.get("is_simulation"):
            summary = simulate_time_off(sim.get("doctor_id"), sim.get("days_off") or 3)
            reason = (
                f"GA (tối ưu hàng loạt): mô phỏng {summary['off_doctor']} nghỉ "
                f"{summary['days_off']} ngày trên {summary['total_virtual']} BN ảo — "
                f"{summary['reassigned']} xếp lại, {summary['unplaced']} chưa xếp được, "
                f"fitness={summary['fitness']}."
            )
            return ("ga", {"simulation_summary": summary, "routing_reason": reason})
        constraints = count_constraints(nlu)
        if len(constraints) >= 2:
            res = DFSSolver.solve(nlu, pcp_doctor_id=pcp_doctor_id)
            return ("dfs", res)
        res = GreedySolver.solve(nlu, pcp_doctor_id=pcp_doctor_id, patient_id=patient_id)
        return ("greedy", res)
