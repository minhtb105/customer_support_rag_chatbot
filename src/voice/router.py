"""Voice router — POST /v1/voice/transcribe|speak, GET /v1/voice/daily-check.

Safety ordering (D7): STT transcript runs check_red_flag BEFORE confirm/save
(even when the number is not parsed yet); numeric panic still flows through
POST /v1/glucose anomaly on save, like manual entry. Keys server-side only
(grep OPENAI_API_KEY in frontend/ must be 0).
Transcript-only (D6): audio hits tmp + deleted immediately, never persisted,
never logged. No duplicate write path: save goes via POST /v1/glucose.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from src.shared.config import API_PREFIX
    from src.auth.dependencies import get_current_user
    AUTH_ENABLED = True
except ImportError:  # pragma: no cover
    from shared.config import API_PREFIX  # type: ignore
    from auth.dependencies import get_current_user  # type: ignore
    AUTH_ENABLED = True

try:
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from api.deps import _resolve_user_id  # type: ignore

router = APIRouter(prefix=API_PREFIX, tags=["voice"])

MAX_AUDIO_BYTES = 5 * 1024 * 1024
ALLOWED_AUDIO = ("audio/webm", "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3")
VN_TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh, no pytz needed

_SLOT_LABEL = {
    "fasting": "lúc đói", "post_meal_2h": "sau ăn 2 giờ",
    "pre_meal": "trước ăn", "bedtime": "trước ngủ", "random": "ngẫu nhiên",
}


class SpeakRequest(BaseModel):
    text: str = ""
    voice_region: str = "Bac"
    selected_persona: Optional[str] = None


class PrefsPatch(BaseModel):
    required_slots: Optional[List[str]] = None
    tts_voice_region: Optional[str] = None
    selected_persona: Optional[str] = None


@router.post("/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...),
                           current_user=Depends(get_current_user)):
    ctype = (audio.content_type or "").split(";")[0].strip().lower()
    if ctype not in ALLOWED_AUDIO:
        raise HTTPException(status_code=422, detail=f"unsupported audio type: {ctype or 'unknown'}")
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=422, detail="audio oversize (max 5MB)")
    try:
        from src.voice.providers import get_stt_provider
    except ImportError:  # pragma: no cover
        from voice.providers import get_stt_provider  # type: ignore
    try:
        text = get_stt_provider().transcribe(data, ctype) or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcribe failed: {e}") from e
    # tmp-free by construction (bytes only, never written); transcript-only.
    out: Dict[str, Any] = {"text": text, "emergency": False}
    try:
        try:
            from src.voice.vn_parse import parse_vi
        except ImportError:
            from voice.vn_parse import parse_vi  # type: ignore
        out["parse"] = parse_vi(text)
    except Exception:
        out["parse"] = {"value": None, "context": None, "ambiguous": True,
                        "reason": "parse_error", "confirm_text": ""}
    try:
        try:
            from src.triage.red_flag import check_red_flag
        except ImportError:
            from triage.red_flag import check_red_flag  # type: ignore
        rf = check_red_flag(text)
        if rf.get("emergency"):
            out.update({"emergency": True, "red_flag_type": rf.get("red_flag_type"),
                        "message": rf.get("message", "")})
    except Exception:
        pass
    return out


@router.post("/voice/speak")
def speak_text(payload: SpeakRequest, current_user=Depends(get_current_user)):
    text = (payload.text or "")[:500]
    if not text.strip():
        raise HTTPException(status_code=422, detail="empty text")
    try:
        from src.voice.providers import synthesize, PERSONA_REGIONS, norm_persona
    except ImportError:  # pragma: no cover
        from voice.providers import synthesize, PERSONA_REGIONS, norm_persona  # type: ignore
    # Server-side persona -> region resolve (D1 pin): persona wins over any
    # client-sent voice_region; synthesize(text, region) signature unchanged.
    region = payload.voice_region or "Bac"
    if payload.selected_persona:
        region = PERSONA_REGIONS.get(norm_persona(payload.selected_persona), "Bac")
    audio_bytes = None
    try:
        audio_bytes = synthesize(text, region)
    except Exception:
        audio_bytes = None
    if not audio_bytes:
        return {"audio_available": False, "text": text}  # fail-open text-only
    return {"audio_available": True, "mime": "audio/wav", "text": text,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii")}


@router.get("/voice/daily-check")
def daily_check(user_id: str = "", current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        try:
            user_id = _resolve_user_id(user_id, current_user)
        except HTTPException as e:
            raise e
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id required")
    try:
        from src.voice.prefs import required_slots
    except ImportError:  # pragma: no cover
        from voice.prefs import required_slots  # type: ignore
    required = required_slots(user_id)
    today = datetime.now(VN_TZ).date().isoformat()
    have: List[str] = []
    try:
        try:
            from src.diabetes.glucose_tracker import get_logs
        except ImportError:
            from diabetes.glucose_tracker import get_logs  # type: ignore
        for log in get_logs(user_id, limit=100) or []:
            try:
                # Compare in VN wall-clock (aware UTC ISO -> +07:00 date).
                dt = datetime.fromisoformat(str(log.get("measured_at", "")))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.astimezone(VN_TZ).date().isoformat() == today \
                        and log.get("context") in required:
                    if log["context"] not in have:
                        have.append(log["context"])
            except Exception:
                continue
    except Exception:
        pass
    missing = [s for s in required if s not in have]
    if missing:
        names = ", ".join(_SLOT_LABEL.get(s, s) for s in missing)
        greeting = f"Chào bác! Hôm nay bác còn thiếu chỉ số {names}. Bác bấm mic và đọc giúp con nhé."
    else:
        greeting = "Chào bác! Hôm nay bác đã đo đủ chỉ số. Cảm ơn bác!"
    return {"user_id": user_id, "date": today, "required_slots": required,
            "missing_slots": missing, "greeting": greeting}


@router.get("/voice/prefs")
def get_voice_prefs(user_id: str = "", current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(user_id, current_user)
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id required")
    try:
        from src.voice.prefs import get_prefs
    except ImportError:  # pragma: no cover
        from voice.prefs import get_prefs  # type: ignore
    return {"user_id": user_id, **get_prefs(user_id)}


@router.post("/voice/prefs")
def set_voice_prefs(payload: PrefsPatch, user_id: str = "",
                    current_user=Depends(get_current_user)):
    if AUTH_ENABLED and current_user:
        user_id = _resolve_user_id(user_id, current_user)
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id required")
    try:
        from src.voice.prefs import set_prefs
    except ImportError:  # pragma: no cover
        from voice.prefs import set_prefs  # type: ignore
    patch = {k: v for k, v in payload.model_dump().items() if v is not None}
    return {"user_id": user_id, **set_prefs(user_id, patch)}
