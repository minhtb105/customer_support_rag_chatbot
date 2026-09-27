"""Per-patient voice prefs — JSON sidecar, fail-open (D4 pin).

No SQLite table/migration for 2 keys (required_slots, tts_voice_region).
metadata/voice_prefs.json via module-namespace + env override
(VOICE_PREFS_PATH, precedent TRIAGE_EVENTS_PATH/LAB_DB_PATH) so tests
monkeypatch to tmp_path. Missing file -> defaults, never 500.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.shared.config import BASE_DIR
except ImportError:  # pragma: no cover
    from shared.config import BASE_DIR  # type: ignore

try:
    from src.voice.providers import PERSONA_REGIONS, norm_persona
except ImportError:  # pragma: no cover
    try:
        from voice.providers import PERSONA_REGIONS, norm_persona  # type: ignore
    except ImportError:  # pragma: no cover
        PERSONA_REGIONS = {"lan": "Bac", "huong": "Trung", "sau": "Nam"}

        def norm_persona(pid: str = "lan") -> str:  # type: ignore
            s = (pid or "").strip().lower()
            if s in PERSONA_REGIONS:
                return s
            alias = {"bac": "lan", "north": "lan",
                     "trung": "huong", "central": "huong",
                     "nam": "sau", "south": "sau"}
            return alias.get(s, "lan")

VOICE_PREFS_PATH = Path(os.getenv("VOICE_PREFS_PATH", str(BASE_DIR / "metadata" / "voice_prefs.json")))

DEFAULTS: Dict[str, Any] = {
    "required_slots": ["fasting", "post_meal_2h"],
    "tts_voice_region": "Bac",
    "selected_persona": "lan",  # canonical (D5 pin); skip -> lan (Bac voice)
}


def _load_all() -> Dict[str, Any]:
    try:
        p = Path(str(VOICE_PREFS_PATH))
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def get_prefs(user_id: str) -> Dict[str, Any]:
    d = dict(DEFAULTS)
    d.update(_load_all().get(user_id or "", {}))
    slots = d.get("required_slots") or DEFAULTS["required_slots"]
    d["required_slots"] = [s for s in slots if isinstance(s, str)][:4] or list(DEFAULTS["required_slots"])
    # selected_persona is canonical; unknown/empty -> lan (D5 pin).
    d["selected_persona"] = norm_persona(d.get("selected_persona", "lan"))
    return d


def set_prefs(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    allp = _load_all()
    cur = dict(DEFAULTS)
    cur.update(allp.get(user_id or "", {}))
    if "required_slots" in patch:
        slots = [s for s in (patch["required_slots"] or []) if isinstance(s, str)][:4]
        if slots:
            cur["required_slots"] = slots
    if "tts_voice_region" in patch and isinstance(patch["tts_voice_region"], str):
        cur["tts_voice_region"] = patch["tts_voice_region"][:16]
    if "selected_persona" in patch and isinstance(patch["selected_persona"], str):
        pid = norm_persona(patch["selected_persona"])
        cur["selected_persona"] = pid
        cur["tts_voice_region"] = PERSONA_REGIONS[pid]  # sync region from persona
    allp[user_id or ""] = cur
    try:
        p = Path(str(VOICE_PREFS_PATH))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(allp, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return cur


def required_slots(user_id: str) -> List[str]:
    return list(get_prefs(user_id).get("required_slots", DEFAULTS["required_slots"]))
