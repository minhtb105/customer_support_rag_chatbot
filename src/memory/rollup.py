"""
Rolling weekly summarization: episodic -> long-term (1 fact/week).

- rollup_weekly(user_id, week_key=None): idempotent on key `weekly-<monday>`.
- Reads episodic summaries/facts for the week, summarizes via gpt-4o-mini
  (fallback: join notes), stores ONE fact into long-term Chroma.
- Does NOT modify core src/memory/* modules; uses public APIs only:
  get_episodic_memory() / get_long_term_memory(user).store_fact(...).
- week_key: "YYYY-MM-DD" Monday of the target week. Defaults to current Monday (UTC).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _current_monday() -> str:
    today = datetime.now(timezone.utc).date()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def _normalize_week_key(week_key: Optional[str]) -> str:
    if week_key:
        return week_key.strip()
    return _current_monday()


def _rollup_key(monday: str) -> str:
    return f"weekly-{monday}"


def _collect_episodic_texts(episodic: Any, monday: str) -> List[str]:
    """Collect episodic summary/fact texts roughly belonging to the week.

    Best-effort: prefer summaries whose timestamp falls in [monday, monday+7d);
    fall back to all summaries/facts when timestamps are unavailable (e.g. fakes).
    """
    texts: List[str] = []
    try:
        start = datetime.fromisoformat(monday).timestamp()
    except Exception:
        start = 0.0
    end = start + 7 * 24 * 3600 if start else float("inf")

    summaries = list(getattr(episodic, "summaries", []) or [])
    in_week = [s for s in summaries if start <= float(getattr(s, "timestamp", 0) or 0) < end]
    pool = in_week or summaries
    for s in pool:
        t = getattr(s, "summary_text", "") or ""
        if t.strip():
            texts.append(t.strip())
        for f in getattr(s, "structured_facts", []) or []:
            ft = (f.get("text") if isinstance(f, dict) else getattr(f, "text", "")) or ""
            if ft.strip():
                texts.append(ft.strip())
    if not texts:
        try:
            for f in episodic.get_all_facts() or []:
                ft = (f.get("text") if isinstance(f, dict) else getattr(f, "text", "")) or ""
                if ft.strip():
                    texts.append(ft.strip())
        except Exception:
            pass
    # buffer messages as last resort
    if not texts:
        for m in list(getattr(episodic, "conversation_buffer", []) or []):
            c = (m.get("content") if isinstance(m, dict) else "") or ""
            if c.strip():
                texts.append(c.strip())
    return texts


def _already_rolled_up(long_term: Any, user_id: str, key: str) -> bool:
    """Check whether a fact for this weekly key already exists (idempotency)."""
    try:
        store = getattr(long_term, "vector_store", None)
        if store is None:
            return False
        res = store.get(filter={"user_id": user_id})
        docs = (res.get("documents") or []) if isinstance(res, dict) else []
        metas = (res.get("metadatas") or []) if isinstance(res, dict) else []
        for d, m in zip(docs, metas):
            m = m or {}
            if m.get("rollup_key") == key or m.get("week_key") == key.replace("weekly-", ""):
                return True
            if key in str(d or ""):
                return True
    except Exception:
        return False
    return False


def _summarize(texts: List[str], monday: str) -> str:
    joined = "\n".join(f"- {t[:200]}" for t in texts[:20])
    prompt_body = (
        f"Tóm tắt thói quen đường huyết tuần bắt đầu {monday} thành 1-2 câu tiếng Việt "
        f"(vd 'Tuần 3/9: ăn ngọt tối nhiều'). Chỉ dùng dữ liệu sau:\n{joined}"
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            )
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Bạn tóm tắt thói quen sức khỏe thành 1-2 câu ngắn gọn, tiếng Việt."},
                    {"role": "user", "content": prompt_body},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception:
            pass
    # Fallback: deterministic join (no LLM needed for demo/tests)
    head = "; ".join(t[:80] for t in texts[:5])
    return f"Tuần {monday}: {head}" if head else f"Tuần {monday}: chưa có dữ liệu episodic."


def rollup_weekly(
    user_id: str,
    week_key: Optional[str] = None,
    episodic: Any = None,
    long_term: Any = None,
) -> Dict[str, Any]:
    """Roll up one week of episodic memory into a single long-term fact.

    Idempotent: second call with the same week_key returns {"skipped": True}.
    """
    monday = _normalize_week_key(week_key)
    key = _rollup_key(monday)

    if episodic is None:
        try:
            from src.memory import get_episodic_memory
        except ImportError:
            from memory import get_episodic_memory  # type: ignore
        episodic = get_episodic_memory()
    if long_term is None:
        try:
            from src.memory import get_long_term_memory
        except ImportError:
            from memory import get_long_term_memory  # type: ignore
        long_term = get_long_term_memory(user_id)

    if _already_rolled_up(long_term, user_id, key):
        return {"skipped": True, "reason": "duplicate", "rollup_key": key, "user_id": user_id}

    texts = _collect_episodic_texts(episodic, monday)
    summary = _summarize(texts, monday)
    fact_text = f"[{key}] {summary}"

    store = getattr(long_term, "store_fact", None) or getattr(long_term, "add_fact", None)
    if store is None:
        raise AttributeError("long_term memory has neither store_fact nor add_fact")
    try:
        fact_id = store(fact_text, source="weekly_rollup", metadata={"week_key": monday, "rollup_key": key})
    except TypeError:
        # fake signatures without metadata
        fact_id = store(fact_text)
    return {
        "skipped": False,
        "fact_id": fact_id,
        "rollup_key": key,
        "user_id": user_id,
        "text": fact_text,
        "source_texts": len(texts),
    }
