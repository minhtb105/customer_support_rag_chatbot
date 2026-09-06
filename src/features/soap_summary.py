"""
Hướng B — Công cụ chuẩn bị hồ sơ trước tái khám (SOAP/ADA).

- Input: glucose logs + long_term memory profile
- Output: SOAP JSON (Subjective/Objective/Assessment/Plan) + markdown
- Có thể export PDF qua weasyprint/reportlab nếu cài, fallback markdown.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List

try:
    from src.features.glucose_tracker import get_logs, get_stats
    from src.memory import get_long_term_memory
    from src.config import DEFAULT_MODEL
except ImportError:  # pragma: no cover
    from features.glucose_tracker import get_logs, get_stats  # type: ignore
    from memory import get_long_term_memory  # type: ignore
    from config import DEFAULT_MODEL  # type: ignore


SOAP_SYSTEM_PROMPT = """Bạn là trợ lý y tế tạo bản tóm tắt trước tái khám theo cấu trúc SOAP,
dựa trên khuyến cáo ADA Standards of Care và WHO PEN.

YÊU CẦU:
- Chỉ dùng dữ liệu được cung cấp (logs, profile). Không bịa thêm chỉ số.
- Viết ngắn gọn, chuẩn lâm sàng, có thể gửi trực tiếp cho bác sĩ.
- Mỗi phần 2-4 bullet, tiếng Việt, kèm disclaimer.
- Nếu thiếu dữ liệu, ghi rõ "chưa có dữ liệu".
"""


def _build_soap_context(user_id: str, days: int) -> str:
    logs = get_logs(user_id, limit=200, days=days)
    stats = get_stats(user_id)
    # profile
    try:
        mem = get_long_term_memory(user_id)
        profile = mem.get_user_profile()
    except Exception:
        profile = {"message": "No profile"}

    ctx = f"=== DỮ LIỆU ĐO ĐƯỜNG HUYẾT ({days} ngày gần nhất) ===\n"
    if not logs:
        ctx += "Chưa có log nào.\n"
    else:
        for l in logs[:20]:  # giới hạn để prompt gọn
            ctx += f"- {l['measured_at'][:16]} | {l['value_mgdl']} mg/dL | {l['context']} | {l['classification']}"
            if l.get("notes"):
                ctx += f" | notes: {l['notes']}"
            ctx += "\n"
        ctx += f"\nThống kê: avg={stats['avg_mgdl']} mg/dL, 7d_avg={stats['last_7_days_avg']}, logs/tuần={stats['logs_per_week']}, streak={stats['streak_days']} ngày\n"
        ctx += f"Phân loại: {stats['classification_counts']}\n"

    ctx += "\n=== HỒ SƠ NGƯỜI DÙNG (Long-term memory) ===\n"
    if "message" in profile:
        ctx += profile["message"] + "\n"
    else:
        for key in ["medications", "conditions", "symptoms", "allergies"]:
            items = profile.get(key, [])
            ctx += f"{key}: {', '.join([x.get('text','')[:80] for x in items[:3]]) if items else '—'}\n"

    return ctx


def generate_soap(user_id: str, days: int = 14, language: str = "vi") -> Dict[str, Any]:
    """
    Gọi LLM để tạo SOAP. Nếu không có OPENAI_API_KEY, trả về template rule-based.
    """
    context = _build_soap_context(user_id, days)
    logs = get_logs(user_id, limit=200, days=days)
    stats = get_stats(user_id)

    # Rule-based fallback (không cần LLM)
    def rule_based():
        # Subjective: tóm tắt triệu chứng / tuân thủ
        subj = f"- Người bệnh ghi nhận {len(logs)} lần đo trong {days} ngày (trung bình {stats['logs_per_week']}/tuần). "
        if stats["logs_per_week"] < 3:
            subj += "Tần suất chưa đạt KPI ≥3 lần/tuần (Hướng A). "
        subj += f"Lần đo gần nhất {logs[0]['value_mgdl']} mg/dL ({logs[0]['classification']}) lúc {logs[0]['measured_at'][:16]}." if logs else "Chưa có dữ liệu."
        obj = f"- Chỉ số trung bình: {stats['avg_mgdl'] or '—'} mg/dL; 7 ngày gần nhất: {stats['last_7_days_avg'] or '—'} mg/dL. "
        obj += f"Phân bố: {stats['classification_counts'] or '—'}. "
        obj += f"Chuỗi ngày đo liên tiếp: {stats['streak_days']} ngày."
        assess = "- Đánh giá tuân thủ tự theo dõi: "
        if stats["logs_per_week"] >= 3:
            assess += "Đạt KPI. "
        else:
            assess += "Chưa đạt — cần nhắc nhở. "
        if stats["classification_counts"].get("high", 0) >= 3 or stats["classification_counts"].get("critical"):
            assess += "Có dấu hiệu tăng đường huyết lặp lại — cần bác sĩ xem xét điều chỉnh."
        else:
            assess += "Chưa ghi nhận chuỗi bất thường kéo dài."
        plan = "- Rà soát lại chế độ ăn, vận động, tuân thủ thuốc. "
        plan += "- Duy trì đo ≥3 lần/tuần, ưu tiên fasting + post-meal 2h. "
        plan += "- Mang bản tóm tắt này và máy đo đến buổi tái khám. "
        plan += "(Lưu ý: tóm tắt AI hỗ trợ, không thay thế chỉ định bác sĩ.)"
        return {"subjective": subj, "objective": obj, "assessment": assess, "plan": plan}

    # Thử gọi LLM nếu có key
    try:
        import os
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        user_prompt = f"{context}\n\nHãy tạo bản tóm tắt SOAP cho bác sĩ, trả về JSON với 4 keys: subjective, objective, assessment, plan."
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SOAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        import json
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        # chuẩn hoá keys — coerce list → string (LLM sometimes returns bullet arrays)
        def _coerce(v):
            if isinstance(v, list):
                return "\n".join(f"- {x}" for x in v)
            return v or ""
        soap = {
            "subjective": _coerce(data.get("subjective") or data.get("Subjective") or ""),
            "objective": _coerce(data.get("objective") or data.get("Objective") or ""),
            "assessment": _coerce(data.get("assessment") or data.get("Assessment") or ""),
            "plan": _coerce(data.get("plan") or data.get("Plan") or ""),
        }
        # fallback nếu thiếu
        if not all(soap.values()):
            rb = rule_based()
            for k in soap:
                if not soap[k]:
                    soap[k] = rb[k]
        return {
            "user_id": user_id,
            "generated_at": datetime.utcnow(),
            "period": f"{days} ngày",
            "soap": soap,
            "stats": stats,
        }
    except Exception as e:
        # fallback rule-based
        rb = rule_based()
        return {
            "user_id": user_id,
            "generated_at": datetime.utcnow(),
            "period": f"{days} ngày",
            "soap": rb,
            "stats": stats,
            "_fallback_reason": str(e)[:200],
        }


def soap_to_markdown(soap_data: Dict[str, Any]) -> str:
    soap = soap_data["soap"]
    stats = soap_data["stats"]
    md = f"# Tóm tắt trước tái khám — {soap_data['user_id']}\n\n"
    md += f"_Tạo lúc: {soap_data['generated_at'].isoformat()[:19]}Z | Khoảng: {soap_data['period']}_\n\n"
    md += f"**Thống kê:** {stats['total_logs']} logs, avg {stats['avg_mgdl']} mg/dL, 7d {stats['last_7_days_avg']} mg/dL, {stats['logs_per_week']}/tuần, streak {stats['streak_days']} ngày\n\n"
    md += "## S — Subjective (Chủ quan)\n" + soap["subjective"] + "\n\n"
    md += "## O — Objective (Khách quan)\n" + soap["objective"] + "\n\n"
    md += "## A — Assessment (Đánh giá)\n" + soap["assessment"] + "\n\n"
    md += "## P — Plan (Kế hoạch)\n" + soap["plan"] + "\n\n"
    md += "---\n*Lưu ý: Tài liệu do AI hỗ trợ tổng hợp từ dữ liệu tự theo dõi, không thay thế chẩn đoán y khoa. Vui lòng tham khảo bác sĩ.*\n"
    return md
