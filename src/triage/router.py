"""Triage service router — /v1/triage, /v1/doctor/patients, /v1/admin/triage/events."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

try:
    from src.shared.config import API_PREFIX
    from src.triage.schemas import TriageRequest, TriageResponse
    from src.diabetes.glucose_tracker import should_escalate_to_doctor
    from src.auth.dependencies import get_current_user_optional, require_expert, require_admin
    AUTH_ENABLED = True
except ImportError:
    from shared.config import API_PREFIX  # type: ignore
    from triage.schemas import TriageRequest, TriageResponse  # type: ignore
    from diabetes.glucose_tracker import should_escalate_to_doctor  # type: ignore
    from auth.dependencies import get_current_user_optional, require_expert, require_admin  # type: ignore
    AUTH_ENABLED = True

try:
    from src.api.deps import _resolve_user_id
except ImportError:  # pragma: no cover
    from api.deps import _resolve_user_id  # type: ignore

router = APIRouter(prefix=API_PREFIX, tags=["triage"])


@router.post("/triage", response_model=TriageResponse)
def triage_endpoint(payload: TriageRequest, current_user=Depends(get_current_user_optional)):
    """Functional sequential pipeline: RedFlag -> NLU -> Matcher -> Response.

    Emergency short-circuits at CALL level: audit-append + return immediately
    WITHOUT importing/calling triage_nlu, slot_generator, matcher,
    scope_guard, or any LLM. Runs BEFORE any scope_guard reuse (cardio
    keywords overlap scope_guard and would otherwise return a canned
    refusal instead of the 115 alert).
    """
    # Step 1 — red-flag FIRST (only import allowed on the emergency path).
    try:
        try:
            from src.triage.red_flag import check_red_flag, append_red_flag_audit
        except ImportError:
            from triage.red_flag import check_red_flag, append_red_flag_audit  # type: ignore
        rf = check_red_flag(payload.message or "")
    except Exception:
        rf = {"emergency": False, "red_flag_type": None, "message": ""}
    if rf.get("emergency"):
        try:
            append_red_flag_audit(
                str(rf.get("red_flag_type") or "unknown"),
                user_id=payload.user_id,
                excerpt=payload.message or "",
            )
        except Exception:
            pass
        try:
            try:
                from src.triage.triage_events import append_triage_event
            except ImportError:
                from triage.triage_events import append_triage_event  # type: ignore
            append_triage_event(
                user_id=payload.user_id, message=payload.message or "",
                emergency=True, red_flag_type=rf.get("red_flag_type"),
            )
        except Exception:
            pass
        return TriageResponse(
            emergency=True,
            red_flag_type=rf.get("red_flag_type"),
            message=rf.get("message"),
            recommended_doctors=[],
        )
    # Step 2+ — non-emergency only (lazy imports keep the emergency path clean).
    try:
        try:
            from src.triage.triage_nlu import parse_triage
        except ImportError:
            from triage.triage_nlu import parse_triage  # type: ignore
        try:
            from src.scheduling import solvers as _solvers
        except ImportError:
            import scheduling.solvers as _solvers  # type: ignore
        nlu = parse_triage(payload.message or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"triage NLU error: {e}") from e
    symptoms = nlu.get("symptoms") or []
    intents = nlu.get("intents") or []
    tc = nlu.get("time_constraints") or {}
    excluded = set(tc.get("excluded_weekdays") or [])
    # Simulation intent (admin demo command, NOT a medical emergency):
    # GA in-memory summary, response-only. SKIPS dual-write entirely (D1):
    # the generator takes no user_id and never touches glucose/episodic DB.
    sim = nlu.get("simulation") or {}
    if sim.get("is_simulation"):
        _solver, _payload = _solvers.SchedulerOrchestrator.route(nlu, patient_id=payload.user_id)
        try:
            try:
                from src.triage.triage_events import append_triage_event
            except ImportError:
                from triage.triage_events import append_triage_event  # type: ignore
            append_triage_event(
                user_id=payload.user_id, message=payload.message or "",
                emergency=False, specialty=nlu.get("specialty"), solver_used="ga",
            )
        except Exception:
            pass
        return TriageResponse(
            emergency=False,
            urgency=nlu.get("urgency"),
            symptoms=symptoms,
            suggested_specialty=nlu.get("specialty"),
            recommended_doctors=[],
            solver_used="ga",
            routing_reason=_payload.get("routing_reason"),
            simulation_summary=_payload.get("simulation_summary"),
        )
    constraints = _solvers.count_constraints(nlu)
    # Missing core info -> one clarification question, no slots.
    if not symptoms and not nlu.get("requested_doctor") and not excluded and "booking" not in intents and not constraints:
        try:
            try:
                from src.triage.triage_events import append_triage_event
            except ImportError:
                from triage.triage_events import append_triage_event  # type: ignore
            append_triage_event(
                user_id=payload.user_id, message=payload.message or "",
                emergency=False, specialty=nlu.get("specialty"),
            )
        except Exception:
            pass
        return TriageResponse(
            emergency=False,
            urgency=nlu.get("urgency"),
            symptoms=[],
            suggested_specialty=nlu.get("specialty"),
            recommended_doctors=[],
            followup_question=(
                "Bạn có thể mô tả rõ hơn triệu chứng đang gặp "
                "(ví dụ: tê chân, mắt mờ, khát nhiều) và thời gian muốn khám không?"
            ),
        )
    # PCP lookup for the matcher chain (fail-open, synthetic sidecar aware).
    pcp_id = None
    if payload.user_id:
        try:
            try:
                from src.scheduling.synthetic_roster import get_patient_pcp
            except ImportError:
                from scheduling.synthetic_roster import get_patient_pcp  # type: ignore
            pcp_id = get_patient_pcp(payload.user_id)
        except Exception:
            pcp_id = None
    # Dynamic solver routing (D2): count selects the algorithm ONLY;
    # every parsed filter is applied by whichever solver runs.
    try:
        solver_used, solved = _solvers.SchedulerOrchestrator.route(
            nlu, pcp_doctor_id=pcp_id, patient_id=payload.user_id)
    except Exception:
        solver_used, solved = "greedy", {"recommended_doctors": [], "routing_reason": ""}
    recommended = solved.get("recommended_doctors") or []
    routing_reason = solved.get("routing_reason")
    previsit_notes = None
    if "symptom_note" in intents and (payload.message or "").strip():
        previsit_notes = (payload.message or "").strip()
        # Dual-write only when authenticated AND authorized; otherwise response-only.
        # Anonymous (current_user is None) -> zero DB writes. Authenticated
        # role=user writing to another id -> _resolve_user_id raises -> response-only.
        if payload.user_id and current_user is not None:
            try:
                user_id = _resolve_user_id(payload.user_id, current_user)
            except HTTPException:
                user_id = None
            if user_id:
                try:
                    try:
                        from src.diabetes.followup_notes import save_followup_notes
                    except ImportError:
                        from diabetes.followup_notes import save_followup_notes  # type: ignore
                    save_followup_notes(user_id, previsit_notes, source="triage")
                except Exception:
                    pass
    try:
        try:
            from src.triage.triage_events import append_triage_event
        except ImportError:
            from triage.triage_events import append_triage_event  # type: ignore
        append_triage_event(
            user_id=payload.user_id, message=payload.message or "",
            emergency=False, specialty=nlu.get("specialty"), solver_used=solver_used,
        )
    except Exception:
        pass
    return TriageResponse(
        emergency=False,
        urgency=nlu.get("urgency"),
        symptoms=symptoms,
        suggested_specialty=nlu.get("specialty"),
        recommended_doctors=recommended,
        previsit_notes=previsit_notes,
        solver_used=solver_used,
        routing_reason=routing_reason,
    )


@router.get("/doctor/patients")
def list_doctor_patients(
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user=Depends(require_expert),
):
    """Smart queue for doctors: DISTINCT glucose user_ids, triage server-side. Glucose-only."""
    try:
        from src.diabetes import glucose_tracker as _gt
    except ImportError:
        import diabetes.glucose_tracker as _gt  # type: ignore
    try:
        from src.vitals.base_tracker import get_conn as _get_conn
    except ImportError:
        from vitals.base_tracker import get_conn as _get_conn  # type: ignore
    try:
        from src.diabetes.anomaly_detector import detect_spike, detect_trend, SPIKE_HIGH, SPIKE_LOW
    except ImportError:
        from diabetes.anomaly_detector import detect_spike, detect_trend, SPIKE_HIGH, SPIKE_LOW  # type: ignore
    try:
        from src.auth.db import get_user_by_id as _get_user
    except ImportError:
        from auth.db import get_user_by_id as _get_user  # type: ignore
    try:
        from src.scheduling.synthetic_roster import get_patient_display_name as _syn_name
    except ImportError:
        try:
            from scheduling.synthetic_roster import get_patient_display_name as _syn_name  # type: ignore
        except ImportError:
            _syn_name = lambda _uid: None  # type: ignore
    _gt.init_glucose_db()
    try:
        conn = _get_conn(_gt.GLUCOSE_DB_PATH)
        try:
            total = conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT user_id FROM glucose_logs)").fetchone()[0]
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM glucose_logs ORDER BY user_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        finally:
            conn.close()
        _uids = [r[0] for r in rows]
    except Exception:
        return {"patients": [], "total": 0, "limit": limit, "offset": offset}
    # ponytail: slice 100, pagination khi >100 patients
    _rank = {"critical": 0, "trend": 1, "watch": 2, "safe": 3}
    out: list = []
    for _uid in _uids:
        try:
            _logs = _gt.get_logs(_uid, limit=50, days=90)
            if not _logs:
                continue
            _last = _logs[0]
            try:
                _spike = detect_spike(float(_last.get("value_mgdl")))
            except (TypeError, ValueError):
                _spike = {"type": "none"}
            _trend = detect_trend(_logs)
            _esc = should_escalate_to_doctor(_uid)
            _cls = str(_last.get("classification") or "")
            if _spike.get("type") != "none" or _cls in ("critical", "low"):
                _level = "critical"
            elif _trend.get("type") == "trend_cascade":
                _level = "trend"
            elif _esc or _cls == "high":
                _level = "watch"
            else:
                _level = "safe"
            try:
                _uname = _syn_name(_uid)  # synthetic sidecar pretty name (fail-open None)
            except Exception:
                _uname = None
            if not _uname:
                try:
                    _u = _get_user(_uid)
                    _uname = (_u.get("username") if _u else None) or _uid[:8]
                except Exception:
                    _uname = _uid[:8]
            _acount = 0
            for _l in _logs:
                try:
                    _vv = float(_l.get("value_mgdl"))
                except (TypeError, ValueError):
                    continue
                if _vv > SPIKE_HIGH or _vv < SPIKE_LOW:
                    _acount += 1
            out.append({"user_id": _uid, "username": _uname, "last_value": _last.get("value_mgdl"), "last_classification": _cls, "last_measured_at": _last.get("measured_at"), "level": _level, "should_escalate": bool(_esc), "anomaly_count": _acount})
        except Exception:
            # fail-open per-user: broken user -> safe row, never break the list
            try:
                out.append({"user_id": _uid, "username": _uid[:8], "last_value": None, "last_classification": "", "last_measured_at": "", "level": "safe", "should_escalate": False, "anomaly_count": 0})
            except Exception:
                pass
    out.sort(key=lambda p: str(p.get("last_measured_at") or ""), reverse=True)
    out.sort(key=lambda p: _rank.get(p["level"], 3))
    return {"patients": out, "total": total, "limit": limit, "offset": offset}


@router.get("/admin/triage/events")
def list_triage_events_admin(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    specialty: Optional[str] = Query(None),
    emergency: Optional[bool] = Query(None),
    q: Optional[str] = Query(None),
    current_user=Depends(require_admin),
):
    """Paged triage event log (admin doubles as receptionist in demo; no new roles)."""
    try:
        try:
            from src.triage.triage_events import list_triage_events
        except ImportError:
            from triage.triage_events import list_triage_events  # type: ignore
        return list_triage_events(
            page=page, limit=limit, specialty=specialty or None,
            emergency=emergency, q=(q or "").strip() or None,
        )
    except Exception:
        return {"events": [], "total": 0, "page": page, "limit": limit}
