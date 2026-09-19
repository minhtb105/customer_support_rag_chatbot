"""Admin Prompts — 2-step approval + dry-run with 5 queries"""
from __future__ import annotations
import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

try:
    from src.auth.dependencies import require_admin, get_current_user
    from src.observability.tracing_db import (
        list_prompts, get_active_prompt, get_prompt_by_version, create_prompt, approve_prompt, reject_prompt
    )
    from src.prompt_manager import PROMPT_REGISTRY
    from src.rag_pipeline import rag_chat
except ImportError:
    from auth.dependencies import require_admin, get_current_user  # type: ignore
    from observability.tracing_db import list_prompts, get_active_prompt, get_prompt_by_version, create_prompt, approve_prompt, reject_prompt  # type: ignore
    from prompt_manager import PROMPT_REGISTRY  # type: ignore
    from rag_pipeline import rag_chat  # type: ignore

router = APIRouter(prefix="/v1/admin/prompts", tags=["admin-prompts"])

class PromptCreateRequest(BaseModel):
    tone: str
    text: str
    description: Optional[str] = None

class PromptApproveRequest(BaseModel):
    tone: str
    version: str

class DryRunRequest(BaseModel):
    tone: str
    text: str  # draft prompt to test
    queries: List[str]  # custom queries, or if empty use golden 5
    top_k: Optional[int] = 5

# Golden dataset 5 samples per tone (fallback)
GOLDEN_QUERIES = {
    "diabetes": [
        "Ngưỡng chẩn đoán đái tháo đường theo WHO là bao nhiêu?",
        "Dấu hiệu đái tháo đường type 2 là gì?",
        "Chế độ ăn cho người tiền đái tháo đường theo WHO?",
        "Biến chứng của đái tháo đường không kiểm soát?",
        "Khi nào cần đo HbA1c và tần suất tái khám?"
    ],
    "hypertension": [
        "Ngưỡng huyết áp bình thường theo AHA 2025 là bao nhiêu?",
        "Tăng huyết áp độ 1 cần làm gì?",
        "Khi nào gọi cấp cứu tăng huyết áp?",
        "Thuốc huyết áp có tác dụng phụ gì?",
        "Đo huyết áp tại nhà đúng cách thế nào?"
    ],
    "respiratory": [
        "Phân biệt hen và COPD theo GOLD 2024?",
        "Peak flow màu vàng nghĩa là gì?",
        "Kỹ thuật hít đúng cho hen suyễn?",
        "CAT score 15 nghĩa là gì?",
        "Khi nào cần đi khám hô hấp?"
    ],
    "mental": [
        "Dấu hiệu trầm cảm theo mhGAP?",
        "PHQ-9 15 nghĩa là gì?",
        "Khi nào cần gọi hotline tâm thần?",
        "Lo âu kéo dài có nguy hiểm không?",
        "Làm sao hỗ trợ người có ý nghĩ tự hại?"
    ],
    "balanced": [
        "Why does dehydration cause headaches?",
        "How to improve sleep quality?",
        "What is diabetes classification?",
        "Explain hypertension stages",
        "What are asthma triggers?"
    ],
    "strict": ["What are WHO diagnostic thresholds for diabetes?", "Is metformin safe?", "Explain COPD gold stages", "Define hypertension crisis", "List diabetes complications"],
    "friendly": ["I feel tired and stressed, advice?", "How to diet for diabetes prevention?", "Sleep advice", "Exercise for hypertension", "Healthy lifestyle tips"],
    "soap": ["Generate SOAP for glucose logs", "SOAP for hypertension", "SOAP for respiratory", "SOAP for mental", "Pre-visit summary"],
}

@router.get("")
def list_all_prompts(status: Optional[str] = None, tone: Optional[str] = None, current_user=Depends(require_admin)):
    rows = list_prompts(tone=tone, status=status)
    # enrich active flag
    return {"prompts": rows, "total": len(rows)}

@router.get("/{tone}")
def get_prompt_tone(tone: str, current_user=Depends(require_admin)):
    if tone not in PROMPT_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown tone")
    active = get_active_prompt(tone)
    versions = list_prompts(tone=tone)
    return {"tone": tone, "active": active, "versions": versions, "description": PROMPT_REGISTRY[tone]["description"]}

@router.post("/draft")
def create_draft(req: PromptCreateRequest, current_user=Depends(require_admin)):
    if req.tone not in PROMPT_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown tone")
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Prompt text too short")
    # create as pending_approval
    created = create_prompt(req.tone, req.text, req.description, created_by=current_user["id"], status="pending_approval")
    return created

@router.post("/{tone}/approve/{version}")
def approve(tone: str, version: str, current_user=Depends(require_admin)):
    if tone not in PROMPT_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown tone")
    p = get_prompt_by_version(tone, version)
    if not p:
        raise HTTPException(status_code=404, detail="Version not found")
    if p["status"] == "active":
        return p
    approved = approve_prompt(tone, version)
    # invalidate cache in prompt_manager
    try:
        from src.prompt_manager import _cache
        _cache.pop(tone, None)
    except Exception:
        pass
    return approved

@router.post("/{tone}/reject/{version}")
def reject(tone: str, version: str, current_user=Depends(require_admin)):
    p = get_prompt_by_version(tone, version)
    if not p:
        raise HTTPException(status_code=404, detail="Version not found")
    reject_prompt(tone, version)
    return {"msg": "rejected"}

@router.post("/dry-run")
def dry_run(req: DryRunRequest, current_user=Depends(require_admin)):
    if req.tone not in PROMPT_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown tone")
    queries = req.queries
    if not queries or len(queries)==0:
        queries = GOLDEN_QUERIES.get(req.tone, GOLDEN_QUERIES["balanced"])[:5]
    # limit to 5 for cost control, but allow up to 10 if admin provides custom
    queries = queries[:10]
    # temporarily create a draft prompt version but not active, then test each query via rag_chat with monkey patch prompt
    # We will temporarily override prompt_manager cache for this tone
    from src.prompt_manager import _cache
    import hashlib
    # save original active
    original_active = get_active_prompt(req.tone)
    # temporary set cache to draft text
    draft_version = hashlib.sha256(req.text.encode()).hexdigest()[:8]
    prev = _cache.get(req.tone)
    _cache[req.tone] = {"text": req.text, "fetched_at": __import__("time").time(), "version": draft_version}
    results = []
    try:
        for q in queries:
            try:
                # rag_chat will use the draft prompt via get_system_prompt
                r = rag_chat(q, top_k=req.top_k or 5, user_id=current_user["id"], username=current_user.get("username"))
                results.append({"query": q, "answer": r.get("raw_answer",{}).get("answer","")[:800] if isinstance(r.get("raw_answer"),dict) else str(r.get("formatted_answer",""))[:800], "trace_id": r.get("trace_id"), "status": "ok"})
            except Exception as e:
                results.append({"query": q, "answer": "", "error": str(e)[:500], "status": "failed"})
    finally:
        # restore cache
        if prev:
            _cache[req.tone] = prev
        else:
            _cache.pop(req.tone, None)
    return {"tone": req.tone, "draft_version": draft_version, "results": results, "note": "Dry-run dùng prompt draft tạm thời, chưa active. Admin phải bấm Approve để áp dụng."}
