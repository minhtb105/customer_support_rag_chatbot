"""Learn router — micro-curriculum delivery + quiz + adaptive + 7-day state.

Audio ONLY via /v1/voice/speak (playServerAudio + persona) — no new TTS path.
Quiz/adaptive pure + no-key (mock TTS). GET /v1/lessons/today is
server-driven (VN wall-clock, Asia/Ho_Chi_Minh): day = min(elapsed+1, 7);
missed days resolve to the earliest unfinished lesson (no skipping);
same-day re-entry returns the same lesson. Client only renders + anti-nag.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

try:
    from src.shared.config import API_PREFIX, BASE_DIR
    from src.auth.dependencies import get_current_user
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from shared.config import API_PREFIX, BASE_DIR  # type: ignore
    from auth.dependencies import get_current_user  # type: ignore
    from api.deps import _resolve_user_id  # type: ignore

try:
    from src.learn import store as _st
    from src.learn.recommend import (
        DAY_SEQUENCE, GROUPS, map_quiz_answer, recommend_next,
    )
except ImportError:  # pragma: no cover
    from learn import store as _st  # type: ignore
    from learn.recommend import (  # type: ignore
        DAY_SEQUENCE, GROUPS, map_quiz_answer, recommend_next,
    )

router = APIRouter(prefix=API_PREFIX, tags=["learn"])

VN_TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh (precedent: voice daily-check)


@lru_cache(maxsize=1)
def _lessons_path() -> Path:
    return Path(str(BASE_DIR)) / "data" / "curriculum" / "lessons.json"


@lru_cache(maxsize=1)
def load_lessons() -> list[dict[str, Any]]:
    with open(_lessons_path(), encoding="utf-8") as f:
        return list(json.load(f)["lessons"])


def _lesson_map() -> dict[str, dict[str, Any]]:
    return {str(l["id"]): l for l in load_lessons()}


class OnboardingIn(BaseModel):
    age: Optional[int] = None
    meds: Optional[str] = None
    has_meter: Optional[int] = None


class QuizIn(BaseModel):
    answer: Optional[int] = None
    answer_text: Optional[str] = None


def _uid(user_id: str, current_user: dict) -> str:
    return _resolve_user_id(user_id or "", current_user)


def _today_vn() -> str:
    return datetime.now(VN_TZ).date().isoformat()


def _day_number(day1: str) -> int:
    try:
        d1 = datetime.fromisoformat(day1).date()
        n = (datetime.now(VN_TZ).date() - d1).days + 1
        return max(1, min(n, 7))
    except Exception:
        return 1


def _with_progress(user_id: str) -> tuple[list[dict], list[dict], dict]:
    lessons = load_lessons()
    progress = _st.get_progress(user_id)
    profile = _st.get_profile(user_id)
    by_id = {p["lesson_id"]: p for p in progress}
    out = []
    for l in lessons:
        p = by_id.get(l["id"], {})
        out.append({**l, "status": p.get("status", "not_started"),
                    "attempts": p.get("attempts", 0)})
    return out, progress, profile


@router.get("/lessons")
def list_lessons(user_id: str = "", current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    lessons, _, profile = _with_progress(uid)
    groups: dict[str, list] = {g: [] for g in GROUPS}
    for l in lessons:
        groups.setdefault(l.get("group", ""), []).append(l["id"])
    return {"lessons": lessons, "groups": groups,
            "day1_date": profile.get("day1_date", "")}


@router.get("/lessons/next")
def next_lesson(user_id: str = "", current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    _, progress, profile = _with_progress(uid)
    nxt = recommend_next(profile, progress, load_lessons())
    if nxt is None:
        return {"done": True, "lesson": None}
    return {"done": False, "lesson": nxt}


@router.get("/lessons/today")
def today_lesson(user_id: str = "", current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    lessons, progress, profile = _with_progress(uid)
    day = _day_number(profile.get("day1_date") or _today_vn())
    done = {p["lesson_id"] for p in progress if p.get("status") == "done"}
    # Missed-day pin: earliest unfinished lesson in day-sequence order (no skipping).
    lesson: dict[str, Any] | None = None
    for lid in DAY_SEQUENCE[:day]:
        if lid not in done:
            lesson = _lesson_map().get(lid)
            break
    if lesson is None:
        lesson = _lesson_map().get(DAY_SEQUENCE[day - 1])
    # Reuse glucose-slot check (single get_logs read, fail-open; no 2nd query).
    missing_slots: list[str] = []
    try:
        try:
            from src.voice.prefs import required_slots
        except ImportError:
            from voice.prefs import required_slots  # type: ignore
        try:
            from src.diabetes.glucose_tracker import get_logs
        except ImportError:
            from diabetes.glucose_tracker import get_logs  # type: ignore
        required = required_slots(uid)
        today = _today_vn()
        have: list[str] = []
        for log in get_logs(uid, limit=100) or []:
            try:
                dt = datetime.fromisoformat(str(log.get("measured_at", "")))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.astimezone(VN_TZ).date().isoformat() == today \
                        and log.get("context") in required \
                        and log.get("context") not in have:
                    have.append(log["context"])
            except Exception:
                continue
        missing_slots = [s for s in required if s not in have]
    except Exception:
        missing_slots = []
    return {"day": day, "lesson": lesson, "missing_slots": missing_slots,
            "day1_date": profile.get("day1_date", "")}


@router.get("/lessons/{lesson_id}")
def get_lesson(lesson_id: str, user_id: str = "",
               current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    lesson = _lesson_map().get(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="unknown lesson")
    lessons, progress, _ = _with_progress(uid)
    mine = next((p for p in progress if p["lesson_id"] == lesson_id), {})
    return {**lesson, "status": mine.get("status", "not_started")}


@router.post("/lessons/{lesson_id}/quiz")
def submit_quiz(lesson_id: str, payload: QuizIn, user_id: str = "",
                current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    lesson = _lesson_map().get(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="unknown lesson")
    idx = payload.answer
    if idx is None and payload.answer_text:
        mapped = map_quiz_answer(payload.answer_text)
        if mapped is None:
            return {"mapped": None,
                    "message": "Con nghe chưa rõ. Bác bấm nút A, B hoặc C giúp con nhé."}
        idx = {"A": 0, "B": 1, "C": 2}[mapped]
    if idx not in (0, 1, 2):
        raise HTTPException(status_code=422, detail="answer must be 0, 1 or 2")
    correct = idx == int(lesson["quiz"]["correct"])
    row = _st.record_quiz(uid, lesson_id, correct)
    # Update per-group mastery on the profile (fail-open).
    try:
        from src.learn.recommend import _lesson_group as _grp
    except ImportError:  # pragma: no cover
        from learn.recommend import _lesson_group as _grp  # type: ignore
    try:
        progress = _st.get_progress(uid)
        mastery = {g: round(sum(int(p.get("last_correct", 0))
                                for p in progress if _grp(str(p.get("lesson_id", ""))) == g)
                            / max(1, sum(1 for p in progress
                                         if _grp(str(p.get("lesson_id", ""))) == g)), 3)
                    for g in GROUPS}
        _st.upsert_profile(uid, {"mastery": mastery})
    except Exception:
        pass
    return {"correct": correct, "explanation": lesson["quiz"].get("explanation", ""),
            "attempts": row.get("attempts", 1)}


@router.post("/lessons/onboarding")
def save_onboarding(payload: OnboardingIn, user_id: str = "",
                    current_user=Depends(get_current_user)):
    uid = _uid(user_id, current_user)
    patch: dict[str, Any] = {}
    if payload.age is not None:
        try:
            patch["age"] = max(0, min(130, int(payload.age)))
        except Exception:
            pass
    if payload.meds is not None:
        patch["meds"] = str(payload.meds)[:500]
    if payload.has_meter is not None:
        patch["has_meter"] = 1 if payload.has_meter else 0
    cur = _st.get_profile(uid)  # skippable: empty patch keeps defaults
    if not cur.get("day1_date"):
        patch["day1_date"] = _today_vn()
    return {"user_id": uid, **_st.upsert_profile(uid, patch)}
