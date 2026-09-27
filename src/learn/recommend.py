"""Pure recommend + quiz-answer mapping (no-key deterministic, D3/D5 pins).

weakest-group adaptive only (+ engagement-low -> short:true only when such a
lesson exists, else the branch is dropped). No spaced-repetition (no
longitudinal data yet). map_quiz_answer is letters-only, no guessing:
digits/words ("mot", "1") NEVER map -> None -> buttons.
"""
from __future__ import annotations

import re
import unicodedata

GROUPS = ["hieu-benh", "dinh-duong", "van-dong",
          "thuoc-theo-doi", "bien-chung", "tam-ly"]

# 7-day voice script: one anchor lesson per day (day1 = onboarding + 1.1).
DAY_SEQUENCE = ["1.1", "1.2", "2.1", "2.2", "3.1", "4.1", "5.1"]


def _unaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


# Letters-only patterns (accented + unaccented). Digits/number-words excluded by design.
_QUIZ_RES = [
    re.compile(r"\bdap an ([abc])\b"),
    re.compile(r"\bcau ([abc])\b"),
    re.compile(r"\bchon ([abc])\b"),
    re.compile(r"\b([abc])\b"),
]


def map_quiz_answer(text: str) -> str | None:
    """Map spoken/written quiz reply -> 'A'|'B'|'C', else None (use buttons).

    No-guess: 'mot'/'1'/number-words never map; ambiguous -> None.
    """
    t = _unaccent((text or "").lower().strip())
    if not t:
        return None
    for rx in _QUIZ_RES:
        m = rx.search(t)
        if m:
            return m.group(1).upper()
    return None


def group_mastery(group: str, progress: list[dict]) -> float:
    """Mean last_correct over done lessons of the group (unattempted groups -> 0.0)."""
    vals = [int(p.get("last_correct", 0)) for p in progress
            if _lesson_group(str(p.get("lesson_id", ""))) == group]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _lesson_group(lesson_id: str) -> str:
    prefix = str(lesson_id).split(".")[0]
    return {"1": GROUPS[0], "2": GROUPS[1], "3": GROUPS[2],
            "4": GROUPS[3], "5": GROUPS[4], "6": GROUPS[5]}.get(prefix, "")


def recommend_next(profile: dict, progress: list[dict],
                   lessons: list[dict]) -> dict | None:
    """Weakest-group first; engagement-low prefers short:true within that group
    only when such a lesson exists (else the branch is dropped). Cold-start ->
    first unfinished lesson in DAY_SEQUENCE order, tie-break smallest group.
    Pure + deterministic (no-key)."""
    done = {p["lesson_id"] for p in progress if p.get("status") == "done"}
    todo = [l for l in lessons if l["id"] not in done]
    if not todo:
        return None
    if not done and not any(p.get("attempts") for p in progress):
        for lid in DAY_SEQUENCE:  # cold-start: day order
            hit = next((l for l in todo if l["id"] == lid), None)
            if hit:
                return hit
    # Weakest group (mastery asc, tie-break group order).
    order = {g: i for i, g in enumerate(GROUPS)}
    scored = sorted(GROUPS, key=lambda g: (group_mastery(g, progress), order[g]))
    weak = scored[0]
    in_weak = [l for l in todo if l.get("group") == weak]
    if in_weak:
        if str((profile or {}).get("engagement", "normal")).lower() in ("low", "thap"):
            shorts = [l for l in in_weak if l.get("short") is True]
            if shorts:  # branch exists -> use it; else dropped (fall through)
                return shorts[0]
        return in_weak[0]
    return todo[0]
