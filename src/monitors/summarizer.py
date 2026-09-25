"""LLM summarizer for guideline diff & safety alerts — tiếng Việt mặc định"""
from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional

try:
    from src.shared.prompt_manager import get_system_prompt
    from src.shared.config import DEFAULT_MODEL, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, GROQ_API_KEY, GROQ_BASE_URL
except ImportError:
    from shared.prompt_manager import get_system_prompt  # type: ignore
    from shared.config import DEFAULT_MODEL, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, GROQ_API_KEY, GROQ_BASE_URL  # type: ignore

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
    try:
        from openai import OpenAI
        if LLM_PROVIDER == "groq":
            client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        else:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        # normalize openai/ prefix
        if model.startswith("openai/"):
            model = model.split("/", 1)[1]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise e

def summarize_guideline_diff(
    source: str,
    title: str,
    old_text: Optional[str],
    new_text: str,
    version_hint: Optional[str] = None,
    publication_date_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Return structured diff JSON (VI). Falls back to heuristic if LLM unavailable."""
    system = get_system_prompt("guideline_diff")
    user = (
        f"Nguồn: {source}\nTiêu đề: {title}\n"
        f"Version gợi ý: {version_hint or 'không rõ'}\n"
        f"Ngày XB gợi ý: {publication_date_hint or 'không rõ'}\n\n"
        f"--- TRÍCH ĐOẠN CŨ (có thể rỗng nếu là bản đầu) ---\n{(old_text or '')[:6000]}\n\n"
        f"--- TRÍCH ĐOẠN MỚI ---\n{new_text[:8000]}\n\n"
        "Hãy trả về JSON đúng schema đã mô tả trong system prompt."
    )
    try:
        raw = _call_llm(system, user, max_tokens=1500)
        parsed = json.loads(raw)
        # ensure required keys
        parsed.setdefault("tom_tat_tieng_viet", "")
        parsed.setdefault("changed_sections", [])
        parsed.setdefault("dosage_changes", [])
        parsed.setdefault("new_recommendations", [])
        parsed.setdefault("removed_recommendations", [])
        parsed.setdefault("version_phat_hien", version_hint or "")
        parsed.setdefault("ngay_xuat_ban", publication_date_hint or "")
        parsed.setdefault("muc_do_quan_trong", "medium")
        return parsed
    except Exception as e:
        # heuristic fallback — still VI
        snippet = new_text[:500].replace("\n", " ")
        return {
            "tom_tat_tieng_viet": f"[Heuristic] Phát hiện bản mới {title} ({source}). Trích: {snippet[:200]}... (LLM lỗi: {str(e)[:120]})",
            "changed_sections": [],
            "dosage_changes": [],
            "new_recommendations": [],
            "removed_recommendations": [],
            "version_phat_hien": version_hint or "",
            "ngay_xuat_ban": publication_date_hint or "",
            "muc_do_quan_trong": "medium",
            "_fallback": True,
            "_error": str(e)[:500],
        }

def summarize_safety_alert(raw_json: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Summarize FDA/DAV/MOH raw alert into VI JSON."""
    system = get_system_prompt("safety_summary")
    user = f"Nguồn: {source}\nDữ liệu thô (JSON):\n{json.dumps(raw_json, ensure_ascii=False)[:7000]}\n\nTrả về JSON đúng schema safety_summary."
    try:
        raw = _call_llm(system, user, max_tokens=1000)
        parsed = json.loads(raw)
        parsed.setdefault("tieu_de_vi", raw_json.get("alert_title") or "Cảnh báo an toàn thuốc")
        parsed.setdefault("tom_tat_vi", "")
        parsed.setdefault("thuoc_lien_quan", [])
        parsed.setdefault("loai_canh_bao", "recall")
        parsed.setdefault("muc_do", "medium")
        parsed.setdefault("ly_do", "")
        parsed.setdefault("khuyen_cao", "")
        parsed.setdefault("nguon", source)
        parsed.setdefault("diem_rui_ro", 0.5)
        return parsed
    except Exception as e:
        # heuristic
        title = raw_json.get("alert_title") or raw_json.get("title") or "Cảnh báo"
        drug = raw_json.get("drug_name") or raw_json.get("product_description") or ""
        return {
            "tieu_de_vi": title,
            "tom_tat_vi": f"[Heuristic] Cảnh báo từ {source}: {title} — thuốc: {drug} (LLM lỗi: {str(e)[:120]})",
            "thuoc_lien_quan": [drug] if drug else [],
            "loai_canh_bao": raw_json.get("alert_type", "recall"),
            "muc_do": raw_json.get("severity", "medium"),
            "ly_do": str(raw_json.get("reason_for_recall") or raw_json.get("reason") or "")[:300],
            "khuyen_cao": "Tham khảo dược sĩ/bác sĩ trước khi tiếp tục sử dụng.",
            "nguon": source,
            "diem_rui_ro": 0.5,
            "_fallback": True,
            "_error": str(e)[:500],
        }
