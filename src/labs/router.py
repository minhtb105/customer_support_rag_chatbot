"""Labs Q&A router — POST /v1/labs/ask (Nura B2).

Ordering (D2 pins, precedent triage red_flag short-circuit):
  1. red_flag (text trieu chung) chay DAU TIEN o muc call — emergency thi
     audit-append + return ngay, KHONG import NLU/RAG/LLM/scope_guard.
  2. lab-panic (numeric tren observations, in-panel only via
     lab_thresholds.is_panic) chay THU HAI — panic thi emergency + "115"
     + audit-append, skip RAG/LLM (khong import gi them).
  3. Non-panic: fetch report + history trend (don gian) -> RAG guideline
     grounding (fail-open, chi khi co OPENAI_API_KEY) -> segments rule-based.

Labs carve-out (D2 blocker): KHONG BAO GIO goi scope_guard tren path nay —
scope_guard flag chu de than/gan (suy than, men gan...) trong khi panel lab
CHINH LA than/gan (creatinine/eGFR/AST/ALT). Cau hoi hop le se bi nuot.

Read-only (D9): endpoint chi SELECT lab tables, KHONG dual-write.
Auth (D9, precedent triage auth-gate): Depends(get_current_user_optional) +
_resolve_user_id; anonymous response-only; panic path khong can auth de fire.
"""
from __future__ import annotations

import os
import unicodedata
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

try:
    from src.shared.config import API_PREFIX
    from src.labs.schemas import LabAskRequest, LabAskResponse, LabAnswerSegment, LabCitation
    from src.auth.dependencies import get_current_user_optional
    AUTH_ENABLED = True
except ImportError:  # pragma: no cover
    from shared.config import API_PREFIX  # type: ignore
    from labs.schemas import LabAskRequest, LabAskResponse, LabAnswerSegment, LabCitation  # type: ignore
    from auth.dependencies import get_current_user_optional  # type: ignore
    AUTH_ENABLED = True

try:
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from api.deps import _resolve_user_id  # type: ignore

router = APIRouter(prefix=API_PREFIX, tags=["labs"])

LAB_PANIC_MESSAGE = (
    "🚨 CẢNH BÁO CẤP CỨU: chỉ số xét nghiệm vượt ngưỡng nguy hiểm. "
    "Gọi 115 ngay lập tức hoặc đến cơ sở y tế gần nhất. "
    "Không tự ý đổi liều/tăng liều, không chờ chat trả lời."
)

GUIDELINE_FALLBACK = "BYT QD5481 (2020) + ADA 2024"


def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


# Analyte keywords (co/khong dau) -> LOINC. Than/gan la hop le o day
# (carve-out khoi scope_guard) — chi dung de uu tien sap xep segments.
_ANALYTE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("4548-4", ("hba1c", "hb1ac", "duong huyet trung binh")),
    ("1558-6", ("glucose doi", "duong doi", "fpg", "glucose luc doi")),
    ("2571-8", ("triglyceride", "triglycerid", "tg", "mo mau")),
    ("2093-3", ("cholesterol toan phan", "cholesterol", "tc")),
    ("18262-6", ("ldl", "mo xau")),
    ("18263-4", ("hdl", "mo tot")),
    ("2160-0", ("creatinine", "creatinin", "chuc nang than", "suy than")),
    ("33914-3", ("egfr", "loc cau than", "do loc")),
    ("1920-8", ("ast", "got", "men gan")),
    ("1742-6", ("alt", "gpt", "men gan")),
)


def _parse_question(text: str) -> Dict[str, Any]:
    """Parse nhe cau hoi -> {topics, loincs}. KHONG co key panic/emergency
    (LLM-proof o muc type: khong ai co the override panic qua parse output)."""
    lowered = (text or "").lower()
    plain = _unaccent(lowered)
    loincs: List[str] = []
    for loinc, kws in _ANALYTE_KEYWORDS:
        for kw in kws:
            if kw in lowered or _unaccent(kw) in plain:
                loincs.append(loinc)
                break
    return {"topics": list(loincs), "loincs": loincs}


def _audit_lab(user_id: Optional[str], excerpt: str, red_flag_type: Optional[str],
               emergency: bool) -> None:
    """Audit-append fail-open (red_flag audit + triage event). Never raises."""
    try:
        try:
            from src.triage.red_flag import append_red_flag_audit
        except ImportError:
            from triage.red_flag import append_red_flag_audit  # type: ignore
        append_red_flag_audit(str(red_flag_type or "lab_panic"), user_id=user_id, excerpt=excerpt)
    except Exception:
        pass
    try:
        try:
            from src.triage.triage_events import append_triage_event
        except ImportError:
            from triage.triage_events import append_triage_event  # type: ignore
        append_triage_event(user_id=user_id, message=excerpt, emergency=emergency,
                            red_flag_type=red_flag_type, specialty="lab")
    except Exception:
        pass


def _handoff_review(requester_id: Optional[str], question: str, draft: str,
                    contexts: Any, tier: str, routed_role: Optional[str],
                    reason: str) -> Optional[str]:
    """Tao review row cho T1-T3 (reuse review_requests + notify + fallback).
    Fail-open -> review_id hoac None. T0 khong bao gio goi."""
    if not requester_id or not routed_role:
        return None
    try:
        try:
            from src.reviews.service import create_review_request
        except ImportError:
            from reviews.service import create_review_request  # type: ignore
        rev = create_review_request(
            query=f"[LAB {tier}] {question}",
            draft_answer=draft,
            contexts=list(contexts or []) if isinstance(contexts, list) else [],
            evaluation={"confidence": None,
                        "comments": {"tier": tier, "reason": reason},
                        "failed_metrics": [],
                        "routed_role": routed_role},
            requester_id=requester_id,
            disease="lab",
        )
        if isinstance(rev, dict) and rev.get("id"):
            return str(rev["id"])
    except Exception:
        pass
    return None


def _next_step(tier: str) -> str:
    try:
        try:
            from src.labs.handoff import what_happens_next
        except ImportError:
            from labs.handoff import what_happens_next  # type: ignore
        return str(what_happens_next(tier))
    except Exception:
        return ""


def _is_panic_safe(loinc: str, value: float) -> bool:
    """In-panel only: loinc la -> False (khong phan hoi gia tri ngoai panel)."""
    try:
        try:
            from src.labs import lab_thresholds as _th
        except ImportError:
            from labs import lab_thresholds as _th  # type: ignore
        return bool(_th.is_panic(loinc, float(value)))
    except Exception:
        return False


@router.post("/labs/ask", response_model=LabAskResponse)
def labs_ask(payload: LabAskRequest, current_user=Depends(get_current_user_optional)):
    question = (payload.question or "").strip()
    # Step 1 — red_flag FIRST (only import allowed on the emergency path).
    # NEVER scope_guard here (carve-out): than/gan la lab hop le.
    try:
        try:
            from src.triage.red_flag import check_red_flag
        except ImportError:
            from triage.red_flag import check_red_flag  # type: ignore
        rf = check_red_flag(question)
    except Exception:
        rf = {"emergency": False, "red_flag_type": None, "message": ""}
    if rf.get("emergency"):
        _audit_lab(payload.user_id, question, rf.get("red_flag_type"), True)
        _rid = _handoff_review(payload.user_id, question, str(rf.get("message") or ""),
                               [], "T3", "doctor", "red_flag emergency")
        return LabAskResponse(
            panic=True, emergency=True,
            red_flag_type=rf.get("red_flag_type"), message=rf.get("message"),
            report_id=payload.report_id, user_id=payload.user_id,
            answer_segments=[], handoff_tier="T3",
            review_id=_rid, what_happens_next=_next_step("T3"),
        )
    # Auth gate (triage auth-gate precedent). Panic text o tren da fire
    # ma khong can auth; con lai: anonymous phai kem user_id/report_id.
    if current_user is None and not payload.user_id and not payload.report_id:
        raise HTTPException(status_code=401, detail="Not authenticated: kem user_id hoac report_id, hoac dang nhap")
    try:
        try:
            import src.labs.store as _ls
        except ImportError:
            import labs.store as _ls  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"labs store error: {e}") from e
    owner: Optional[str] = None
    target_report: Optional[Dict[str, Any]] = None
    if payload.report_id:
        try:
            target_report = _ls.get_report(payload.report_id)
        except Exception:
            target_report = None
        if target_report is None:
            raise HTTPException(status_code=404, detail="lab report not found")
        owner = str(target_report.get("user_id") or "")
    owner = owner or payload.user_id or (current_user.get("id") if isinstance(current_user, dict) and current_user else None)
    if not owner:
        raise HTTPException(status_code=401, detail="Not authenticated: kem user_id hoac report_id, hoac dang nhap")
    if current_user is not None:
        try:
            eff = _resolve_user_id(owner, current_user)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
    else:
        eff = owner  # anonymous response-only (read-only, zero writes)
    try:
        obs = _ls.get_observations(target_report["report_id"]) if target_report else []
    except Exception:
        obs = []
    if not target_report:
        try:
            reps = _ls.get_reports(eff)
        except Exception:
            reps = []
        if not reps:
            return LabAskResponse(
                panic=False, emergency=False, report_id=None, user_id=eff,
                answer_segments=[], message="Chua co phieu xet nghiem nao cho user nay.",
                handoff_tier="T0", review_id=None,
                what_happens_next=_next_step("T0"),
            )
        target_report = reps[0]
        try:
            obs = _ls.get_observations(target_report["report_id"])
        except Exception:
            obs = []
    rid = str(target_report.get("report_id") or payload.report_id or "")
    # Step 2 — lab-panic SECOND (in-panel numeric only), BEFORE any
    # NLU/RAG/LLM import. Panic -> emergency + "115" + audit, skip RAG/LLM.
    panics: List[Dict[str, Any]] = []
    for ob in obs or []:
        try:
            if _is_panic_safe(str(ob.get("loinc") or ""), float(ob.get("value"))):
                panics.append(ob)
        except (TypeError, ValueError):
            continue
    if panics:
        _audit_lab(eff, question, "lab_panic", True)
        _rid = _handoff_review(eff, question, LAB_PANIC_MESSAGE, panics,
                               "T3", "doctor", "panic: chi so vuot nguong")
        return LabAskResponse(
            panic=True, emergency=True, red_flag_type="lab_panic",
            message=LAB_PANIC_MESSAGE, report_id=rid, user_id=eff,
            answer_segments=[], handoff_tier="T3",
            review_id=_rid, what_happens_next=_next_step("T3"),
        )
    # Step 3 — non-panic only (lazy imports keep panic paths clean).
    parsed = _parse_question(question)
    mentioned = set(parsed.get("loincs") or [])
    # History trend (don gian): so latest vs diem truoc do theo tung loinc.
    trends: Dict[str, str] = {}
    try:
        for ob in obs or []:
            loinc = str(ob.get("loinc") or "")
            if not loinc:
                continue
            try:
                hist = _ls.get_history(eff, loinc)
            except Exception:
                hist = []
            vals = []
            for h in hist or []:
                try:
                    vals.append(float(h.get("value")))
                except (TypeError, ValueError):
                    continue
            if len(vals) >= 2 and vals[-1]:
                pct = (vals[-1] - vals[-2]) / abs(vals[-1])
                trends[loinc] = "tang" if pct > 0.05 else ("giam" if pct < -0.05 else "on dinh")
            else:
                trends[loinc] = "on dinh"
    except Exception:
        pass
    # RAG guideline grounding (fail-open, chi khi co key; khong anh huong
    # tinh deterministic cua segments khi thieu key).
    guideline = GUIDELINE_FALLBACK
    if os.getenv("OPENAI_API_KEY"):
        try:
            try:
                from src.chat.rag_pipeline import rag_chat
            except ImportError:
                from chat.rag_pipeline import rag_chat  # type: ignore
            _summary = "; ".join(
                f"{ob.get('name')} {ob.get('value')} {ob.get('unit')}" for ob in (obs or [])
            )
            _r = rag_chat(f"Giai thich phieu xet nghiem tieu duong: {question}. Chi so: {_summary}",
                          top_k=3, user_id=eff)
            _raw = _r.get("raw_answer") if isinstance(_r, dict) else None
            _srcs = (_raw.get("cited_sources") or []) if isinstance(_raw, dict) else []
            _names = [str(s.get("source") or s.get("title") or "") for s in _srcs
                      if isinstance(s, dict)][:2]
            _names = [n for n in _names if n]
            if _names:
                guideline = "; ".join(_names)
        except Exception:
            pass
    # Rule-based segments (deterministic): moi claim kem citation
    # report_id + loinc + guideline source (D9 pin).
    try:
        try:
            from src.labs import lab_thresholds as _th2
        except ImportError:
            from labs import lab_thresholds as _th2  # type: ignore
        _panel = _th2.get_panel()
        _l2k = _th2.LOINC_TO_KEY
    except Exception:
        _panel, _l2k = {}, {}
    ordered = sorted(obs or [], key=lambda o: (0 if str(o.get("loinc") or "") in mentioned else 1,
                                               str(o.get("loinc") or "")))
    segments: List[LabAnswerSegment] = []
    for ob in ordered:
        loinc = str(ob.get("loinc") or "")
        name = str(ob.get("name") or loinc)
        try:
            val = float(ob.get("value"))
            val_s = ("%g" % val)
        except (TypeError, ValueError):
            continue
        unit = str(ob.get("unit") or "")
        key = _l2k.get(loinc, loinc) if isinstance(_l2k, dict) else loinc
        spec = _panel.get(key, {}) if isinstance(_panel, dict) else {}
        ref = ""
        try:
            if spec and spec.get("ref_low") is not None and spec.get("ref_high") is not None:
                ref = f" (nguong {spec['ref_low']}-{spec['ref_high']} {unit})"
        except Exception:
            ref = ""
        flag = str(ob.get("flag") or "")
        trend = trends.get(loinc, "on dinh")
        note = ""
        if loinc == "4548-4":
            note = " Muc tieu HbA1c<7% theo BYT QD5481 + ADA 2024."
        if flag == "high":
            state = f"cao hon nguong{ref}; xu huong {trend}."
        elif flag == "low":
            state = f"thap hon nguong{ref}; xu huong {trend}."
        else:
            state = f"trong nguong{ref}; xu huong {trend}."
        segments.append(LabAnswerSegment(
            text=f"{name}: {val_s} {unit} — {state}{note}",
            citations=[LabCitation(report_id=rid, loinc=loinc, source=guideline)],
        ))
    # B3 tier-routing (D6): T0 = auto answer (no review row);
    # T1-T3 = review row + notify (reuse create_review_request) + single
    # triage event-log (same jsonl, via _audit_lab).
    try:
        try:
            from src.labs.handoff import route_tier
        except ImportError:
            from labs.handoff import route_tier  # type: ignore
        _route = route_tier(question, panic=False)
        _tier = str(_route.get("tier") or "T0")
        _role = _route.get("routed_role")
        _reason = str(_route.get("reason") or "")
    except Exception:
        _tier, _role, _reason = "T0", None, ""
    _rid2: Optional[str] = None
    if _tier != "T0" and _role:
        try:
            _draft = "\n".join(s.text for s in segments)
        except Exception:
            _draft = ""
        _rid2 = _handoff_review(eff, question, _draft, obs, _tier, str(_role), _reason)
    try:
        _audit_lab(eff, question, None, False)
    except Exception:
        pass
    return LabAskResponse(
        panic=False, emergency=False, report_id=rid or None, user_id=eff,
        answer_segments=segments, handoff_tier=_tier,
        review_id=_rid2, what_happens_next=_next_step(_tier),
    )
