"""Read-only helpers for synthetic multi-patient/multi-doctor sidecars (fail-open).

Sidecars (written by ``scripts/seed_synthetic_data.py``):
- ``metadata/doctors_patients.json``: patient_id -> {display_name, age, gender,
  comorbidities, pcp_doctor_id, archetype}
- ``metadata/synthetic_roster.json``: {"doctors": [{doctor_id, name, specialty,
  working_hours}]}

All helpers return None/[] when files are missing or malformed — never raise,
so the doctor queue falls back to ``uid[:8]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.shared.config import BASE_DIR as _BASE_DIR
except ImportError:  # pragma: no cover
    _BASE_DIR = Path(__file__).resolve().parents[2]  # type: ignore


def _sidecar_dir() -> Path:
    override = os.getenv("SYNTHETIC_SIDECAR_DIR")
    if override:
        return Path(override)
    return Path(_BASE_DIR) / "metadata"


def _load_json(name: str) -> Any:
    try:
        path = _sidecar_dir() / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_patient_display_name(patient_id: str) -> Optional[str]:
    """Pretty Vietnamese name for a synthetic patient, or None (fail-open)."""
    try:
        mapping = _load_json("doctors_patients.json") or {}
        entry = mapping.get(patient_id) or {}
        name = (entry.get("display_name") or "").strip()
        return name or None
    except Exception:
        return None


def get_patient_pcp(patient_id: str) -> Optional[str]:
    """PCP doctor_id for a synthetic patient, or None (fail-open)."""
    try:
        mapping = _load_json("doctors_patients.json") or {}
        entry = mapping.get(patient_id) or {}
        pcp = (entry.get("pcp_doctor_id") or "").strip()
        return pcp or None
    except Exception:
        return None


def get_patient_location(patient_id: str) -> Optional[Dict[str, float]]:
    """Sidecar lat/lon origin for distance (D5); None -> hash fallback."""
    try:
        mapping = _load_json("doctors_patients.json") or {}
        entry = mapping.get(patient_id) or {}
        lat, lon = entry.get("lat"), entry.get("lon")
        if lat is None or lon is None:
            return None
        return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        return None


def get_doctor_location(doctor_id: str) -> Optional[Dict[str, float]]:
    """Roster doctor lat/lon; None -> hash fallback."""
    try:
        for d in _all_doctors():
            if d.get("doctor_id") == doctor_id:
                lat, lon = d.get("lat"), d.get("lon")
                if lat is None or lon is None:
                    return None
                return {"lat": float(lat), "lon": float(lon)}
        return None
    except Exception:
        return None


def get_hospitals() -> List[Dict[str, Any]]:
    """Hospital list from roster sidecar (fail-open -> [])."""
    try:
        roster = _load_json("synthetic_roster.json") or {}
        hospitals = roster.get("hospitals") or []
        return [h for h in hospitals if isinstance(h, dict)]
    except Exception:
        return []


def _all_doctors() -> List[Dict[str, Any]]:
    try:
        roster = _load_json("synthetic_roster.json") or {}
        doctors = roster.get("doctors") or []
        return [d for d in doctors if isinstance(d, dict)]
    except Exception:
        return []


def query_doctor_availability(
    doctor_id: Optional[str] = None, day: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Working-hours lookup for the AI triage/scheduling demo (fail-open -> []).

    - doctor_id given: return [that doctor's entry] (or [] if unknown).
    - day given (e.g. "Mon", "Saturday"): only doctors working that day
      (substring match against working_hours, case-insensitive).
    """
    try:
        doctors = _all_doctors()
        if doctor_id:
            doctors = [d for d in doctors if d.get("doctor_id") == doctor_id]
        if day:
            needle = day.strip().lower()
            doctors = [
                d for d in doctors
                if needle in str(d.get("working_hours") or "").lower()
            ]
        return doctors
    except Exception:
        return []
