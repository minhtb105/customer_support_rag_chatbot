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
    from src.features.bp_tracker import get_bp_logs, get_bp_stats
    from src.features.respiratory_tracker import get_respiratory_logs, get_respiratory_stats
    from src.features.mood_tracker import get_mood_logs, get_mood_stats
    from src.memory import get_long_term_memory
    from src.config import DEFAULT_MODEL
except ImportError:  # pragma: no cover
    from features.glucose_tracker import get_logs, get_stats  # type: ignore
    from features.bp_tracker import get_bp_logs, get_bp_stats  # type: ignore
    from features.respiratory_tracker import get_respiratory_logs, get_respiratory_stats  # type: ignore
    from features.mood_tracker import get_mood_logs, get_mood_stats  # type: ignore
    from memory import get_long_term_memory  # type: ignore
    from config import DEFAULT_MODEL  # type: ignore


SOAP_SYSTEM_PROMPT = """Bạn là trợ lý y tế tạo bản tóm tắt trước tái khám theo cấu trúc SOAP,
dựa trên QĐ 5481/QĐ-BYT 2020 (ĐTĐ type 2 VN) và ADA Standards of Care 2024,
mục tiêu HbA1c < 7%.

YÊU CẦU BẮT BUỘC:
- Chỉ dùng dữ liệu được cung cấp (logs, profile). Không bịa thêm chỉ số.
- Viết ngắn gọn, chuẩn lâm sàng, có thể gửi trực tiếp cho bác sĩ.
- Mỗi phần 2-4 bullet, tiếng Việt, kèm disclaimer.
- Nếu thiếu dữ liệu, ghi rõ "chưa có dữ liệu".
- Mỗi nhận định ở S/O/A phải kèm audit link dạng [Xem log #<id>] trỏ về log thô.
- Phần A CHỈ được nói đạt/không đạt mục tiêu HbA1c < 7% theo BYT5481/ADA2024.
  TUYỆT ĐỐI KHÔNG đưa ra nhận định mới về bệnh (không dùng các cụm từ chẩn đoán mới).
- Phần plan LUÔN để chuỗi rỗng "" — bác sĩ là người duy nhất chỉ định.
  Không ghi gợi ý điều trị, không ghi liều thuốc, không ghi "cần điều chỉnh".
"""


def _build_soap_context(user_id: str, days: int, disease: str = "diabetes") -> str:
    # Select logs/stats per disease
    if disease == "hypertension":
        logs = get_bp_logs(user_id, limit=200, days=days)
        stats = get_bp_stats(user_id)
        header = f"=== DỮ LIỆU HUYẾT ÁP ({days} ngày) ==="
        line_fmt = lambda l: f"- {l['measured_at'][:16]} | {l['systolic']}/{l['diastolic']} mmHg | {l['context']} | {l['classification']}"
        stats_line = f"Avg: {stats.get('avg_sys')}/{stats.get('avg_dia')} mmHg, at_target_rate={stats.get('at_target_rate')}, logs/tuần={stats.get('logs_per_week')}"
    elif disease == "respiratory":
        logs = get_respiratory_logs(user_id, limit=200, days=days)
        stats = get_respiratory_stats(user_id)
        header = f"=== DỮ LIỆU HÔ HẤP ({days} ngày) ==="
        line_fmt = lambda l: f"- {l['measured_at'][:16]} | peak {l.get('peak_flow_percent')}% | GOLD {l.get('gold_stage')} | {l['classification']}"
        stats_line = f"Avg peak {stats.get('avg_peak_flow')}%, red_rate={stats.get('red_rate')}, incorrect inhaler {stats.get('incorrect_inhaler_rate')}"
    elif disease == "mental":
        logs = get_mood_logs(user_id, limit=200, days=days)
        stats = get_mood_stats(user_id)
        header = f"=== DỮ LIỆU SỨC KHỎE TÂM THẦN ({days} ngày) — PII redacted ==="
        line_fmt = lambda l: f"- {l['measured_at'][:16]} | PHQ9 {l.get('phq9_score')} GAD7 {l.get('gad7_score')} | {l['classification']} | crisis={l.get('crisis_flag')}"
        stats_line = f"Avg PHQ9 {stats.get('avg_phq9')} GAD7 {stats.get('avg_gad7')}, crisis={stats.get('crisis_count')}"
    else:
        logs = get_logs(user_id, limit=200, days=days)
        stats = get_stats(user_id)
        header = f"=== DỮ LIỆU ĐO ĐƯỜNG HUYẾT ({days} ngày gần nhất) ==="
        line_fmt = lambda l: f"- {l['measured_at'][:16]} | {l['value_mgdl']} mg/dL | {l['context']} | {l['classification']}"
        stats_line = f"avg={stats['avg_mgdl']} mg/dL, 7d_avg={stats['last_7_days_avg']}, logs/tuần={stats['logs_per_week']}, streak={stats['streak_days']} ngày"

    # profile
    try:
        mem = get_long_term_memory(user_id)
        profile = mem.get_user_profile()
    except Exception:
        profile = {"message": "No profile"}

    ctx = header + "\n"
    if not logs:
        ctx += "Chưa có log nào.\n"
    else:
        for l in logs[:20]:
            ctx += line_fmt(l)
            if l.get("notes") or l.get("mood_notes"):
                ctx += f" | notes: {(l.get('notes') or l.get('mood_notes') or '')[:60]}"
            ctx += "\n"
        ctx += f"\nThống kê: {stats_line}\n"
        ctx += f"Phân loại: {stats.get('classification_counts')}\n"

    ctx += "\n=== HỒ SƠ NGƯỜI DÙNG (Long-term memory) ===\n"
    if "message" in profile:
        ctx += profile["message"] + "\n"
    else:
        for key in ["medications", "conditions", "symptoms", "allergies"]:
            items = profile.get(key, [])
            ctx += f"{key}: {', '.join([x.get('text','')[:80] for x in items[:3]]) if items else '—'}\n"

    return ctx


def generate_soap(user_id: str, days: int = 14, language: str = "vi", disease: str = "diabetes") -> Dict[str, Any]:
    """
    Gọi LLM để tạo SOAP. Nếu không có OPENAI_API_KEY, trả về template rule-based.
    disease: diabetes|hypertension|respiratory|mental|all
    """
    # Select appropriate stats for rule-based
    if disease == "hypertension":
        logs_r = get_bp_logs(user_id, limit=200, days=days)
        stats_r = get_bp_stats(user_id)
    elif disease == "respiratory":
        logs_r = get_respiratory_logs(user_id, limit=200, days=days)
        stats_r = get_respiratory_stats(user_id)
    elif disease == "mental":
        logs_r = get_mood_logs(user_id, limit=200, days=days)
        stats_r = get_mood_stats(user_id)
    else:
        logs_r = get_logs(user_id, limit=200, days=days)
        stats_r = get_stats(user_id)
    context = _build_soap_context(user_id, days, disease=disease)
    logs = logs_r
    stats = stats_r

    # Rule-based fallback (không cần LLM) — disease-aware
    # Diabetes path: S/O/A có audit link [Xem log #id]; A chỉ HbA1c<7%; P="".
    def _log_link(l: Dict[str, Any]) -> str:
        try:
            return f"[Xem log #{int(l.get('id'))}]"
        except (TypeError, ValueError):
            return ""

    def rule_based():
        subj = f"- Người bệnh ghi nhận {len(logs)} lần đo trong {days} ngày (trung bình {stats.get('logs_per_week',0)}/tuần). "
        if stats.get("logs_per_week", 0) < 3:
            subj += "Tần suất chưa đạt KPI ≥3 lần/tuần (Hướng A). "
        if not logs:
            subj += "Chưa có dữ liệu."
        else:
            last = logs[0]
            if disease == "hypertension":
                subj += f"Lần đo gần nhất {last.get('systolic')}/{last.get('diastolic')} mmHg ({last.get('classification')}) lúc {last.get('measured_at','')[:16]} {_log_link(last)}."
            elif disease == "respiratory":
                subj += f"Lần đo gần nhất peak {last.get('peak_flow_percent')}% GOLD {last.get('gold_stage')} ({last.get('classification')}) lúc {last.get('measured_at','')[:16]} {_log_link(last)}."
            elif disease == "mental":
                subj += f"Lần đo gần nhất PHQ-9 {last.get('phq9_score')} GAD-7 {last.get('gad7_score')} ({last.get('classification')}) lúc {last.get('measured_at','')[:16]} {_log_link(last)}."
                if last.get("crisis_flag"):
                    subj += " ⚠️ Crisis flag."
            else:
                subj += f"Lần đo gần nhất {last.get('value_mgdl')} mg/dL ({last.get('classification')}) lúc {last.get('measured_at','')[:16]} {_log_link(last)}."
                # FQG notes context (symptoms from chat/FQG notes)
                _notes = [f"{(l.get('notes') or '').strip()} {_log_link(l)}".strip() for l in logs[:5] if (l.get("notes") or "").strip()]
                if _notes:
                    subj += " Triệu chứng/ghi chú tự báo: " + "; ".join(_notes[:3]) + "."
                else:
                    subj += " Không có ghi chú triệu chứng kèm theo."
        # Objective
        if disease == "hypertension":
            obj = f"- Trung bình: {stats.get('avg_sys') or '—'}/{stats.get('avg_dia') or '—'} mmHg; 7 ngày: {stats.get('last_7_days_avg') or '—'}. "
            obj += f"Phân bố: {stats.get('classification_counts') or '—'}. At-target: {stats.get('at_target_rate',0)*100:.0f}%. "
            obj += f"Chuỗi ngày đo: {stats.get('streak_days',0)} ngày."
        elif disease == "respiratory":
            obj = f"- Avg peak flow: {stats.get('avg_peak_flow') or '—'}%; red_rate: {stats.get('red_rate',0)}; incorrect_inhaler: {stats.get('incorrect_inhaler_rate',0)}. "
            obj += f"Phân bố: {stats.get('classification_counts') or '—'}."
        elif disease == "mental":
            obj = f"- Avg PHQ-9: {stats.get('avg_phq9') or '—'}; GAD-7: {stats.get('avg_gad7') or '—'}; Crisis: {stats.get('crisis_count',0)}. "
            obj += f"Phân bố: {stats.get('classification_counts') or '—'}."
        else:
            obj = f"- Chỉ số trung bình: {stats.get('avg_mgdl') or '—'} mg/dL; 7 ngày gần nhất: {stats.get('last_7_days_avg') or '—'} mg/dL. "
            obj += f"Phân bố: {stats.get('classification_counts') or '—'}. "
            obj += f"Chuỗi ngày đo liên tiếp: {stats.get('streak_days',0)} ngày."
            _links = " ".join(_log_link(l) for l in logs[:3] if _log_link(l))
            if _links:
                obj += f" Nguồn số liệu: {_links}."
        if disease != "diabetes":
            assess = "- Đánh giá tuân thủ tự theo dõi: "
            if stats.get("logs_per_week", 0) >= 3:
                assess += "Đạt KPI. "
            else:
                assess += "Chưa đạt — cần nhắc nhở. "
            # Disease-specific escalation signals (non-diabetes keeps prior wording)
            cc = stats.get("classification_counts", {})
            if disease == "hypertension":
                if cc.get("crisis") or cc.get("stage2", 0) >= 3:
                    assess += "Có dấu hiệu tăng huyết áp nặng — cần bác sĩ xem xét (crisis hoặc 3×stage2)."
                else:
                    assess += "Chưa ghi nhận chuỗi bất thường kéo dài."
            elif disease == "respiratory":
                if cc.get("red", 0) >= 1:
                    assess += "Có lần đo vùng đỏ — cần can thiệp theo action plan."
                else:
                    assess += "Chưa ghi nhận vùng đỏ."
            elif disease == "mental":
                if cc.get("crisis") or stats.get("crisis_count", 0) > 0:
                    assess += "Có crisis flag — cần chuyển ngay tới chuyên gia, cung cấp hotline 1800-1567/1900-1267/115."
                elif cc.get("severe", 0) > 0:
                    assess += "Có mức severe — khuyến nghị đánh giá chuyên khoa trong vài ngày."
                else:
                    assess += "Chưa ghi nhận mức nghiêm trọng kéo dài."
        else:
            # Diabetes A: ONLY HbA1c<7% target check per BYT5481/ADA2024. No new diagnoses.
            cc = stats.get("classification_counts", {})
            _links_a = " ".join(_log_link(l) for l in logs[:3] if _log_link(l))
            assess = "- Đối chiếu mục tiêu HbA1c < 7% theo QĐ 5481/QĐ-BYT 2020 và ADA 2024 (proxy từ dữ liệu tự đo, không thay thế xét nghiệm HbA1c tại lab): "
            n_high = int(cc.get("high", 0) or 0) + int(cc.get("critical", 0) or 0)
            if not logs:
                assess += "chưa có dữ liệu để đối chiếu. "
            elif n_high >= 3 or cc.get("critical"):
                assess += "chưa đạt mục tiêu (ghi nhận tăng đường huyết lặp lại trong kỳ theo dõi). "
            elif (stats.get("avg_mgdl") or 0) and stats["avg_mgdl"] >= 154:
                assess += "chưa đạt mục tiêu (trung bình ước tính tương đương HbA1c ≥ 7%). "
            else:
                assess += "đạt mục tiêu trong kỳ theo dõi (không ghi nhận chuỗi bất thường kéo dài). "
            assess += "(Tóm tắt AI hỗ trợ, không thay thế chỉ định bác sĩ.)"
            if _links_a:
                assess += f" Nguồn: {_links_a}."
        # P is ALWAYS empty — the doctor decides. Frontend renders the muted placeholder.
        plan = ""
        return {"subjective": subj, "objective": obj, "assessment": assess, "plan": plan}

    # Thử gọi LLM nếu có key
    try:
        import os
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        user_prompt = (
            f"{context}\n\nHãy tạo bản tóm tắt SOAP cho bác sĩ, trả về JSON với 4 keys: "
            "subjective, objective, assessment, plan. "
            "Mỗi nhận định ở subjective/objective/assessment phải kèm audit link [Xem log #<id>] "
            "(lấy id từ dữ liệu logs). "
            "Assessment CHỈ được nói đạt/không đạt mục tiêu HbA1c < 7% theo BYT5481/ADA2024, "
            "không đưa ra nhận định mới về bệnh. "
            'Plan LUÔN là chuỗi rỗng "".'
        )
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
        # P is always empty (doctor decides) — enforce even if LLM returns content.
        soap["plan"] = ""
        # fallback nếu thiếu (trừ plan luôn rỗng)
        if not soap.get("subjective") or not soap.get("objective") or not soap.get("assessment"):
            rb = rule_based()
            for k in ("subjective", "objective", "assessment"):
                if not soap[k]:
                    soap[k] = rb[k]
        # Guardrail: LLM must not drop audit links / HbA1c target line (diabetes).
        # Append source links when the model ignores the prompt instruction.
        if disease == "diabetes" and logs:
            _src_links = " ".join(_log_link(l) for l in logs[:3] if _log_link(l))
            if _src_links:
                for k in ("subjective", "objective", "assessment"):
                    if "[Xem log #" not in (soap.get(k) or ""):
                        soap[k] = (soap[k] + f" Nguồn: {_src_links}.").strip()
                if "HbA1c" not in (soap.get("assessment") or ""):
                    soap["assessment"] = (
                        soap["assessment"]
                        + " Đối chiếu mục tiêu HbA1c < 7% theo QĐ 5481/QĐ-BYT 2020 và ADA 2024."
                    ).strip()
        return {
            "user_id": user_id,
            "generated_at": datetime.utcnow(),
            "period": f"{days} ngày",
            "soap": soap,
            "stats": stats,
            "disease": disease,
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
            "disease": disease,
            "_fallback_reason": str(e)[:200],
        }


def soap_to_markdown(soap_data: Dict[str, Any]) -> str:
    soap = soap_data["soap"]
    stats = soap_data["stats"]
    disease = soap_data.get("disease", "diabetes")
    md = f"# Tóm tắt trước tái khám — {soap_data['user_id']} ({disease})\n\n"
    md += f"_Tạo lúc: {soap_data['generated_at'].isoformat()[:19]}Z | Khoảng: {soap_data['period']}_\n\n"
    # Generic stats line
    if disease == "diabetes":
        md += f"**Thống kê:** {stats.get('total_logs',0)} logs, avg {stats.get('avg_mgdl')} mg/dL, 7d {stats.get('last_7_days_avg')} mg/dL, {stats.get('logs_per_week')}/tuần, streak {stats.get('streak_days')} ngày\n\n"
    elif disease == "hypertension":
        md += f"**Thống kê:** {stats.get('total_logs',0)} logs, avg {stats.get('avg_sys')}/{stats.get('avg_dia')} mmHg, at_target {stats.get('at_target_rate')}, {stats.get('logs_per_week')}/tuần\n\n"
    elif disease == "respiratory":
        md += f"**Thống kê:** {stats.get('total_logs',0)} logs, avg peak {stats.get('avg_peak_flow')}%, red_rate {stats.get('red_rate')}\n\n"
    elif disease == "mental":
        md += f"**Thống kê:** {stats.get('total_logs',0)} logs, avg PHQ9 {stats.get('avg_phq9')} GAD7 {stats.get('avg_gad7')}, crisis {stats.get('crisis_count')}\n\n"
    else:
        md += f"**Thống kê:** {stats}\n\n"
    md += "## S — Subjective (Chủ quan)\n" + soap["subjective"] + "\n\n"
    md += "## O — Objective (Khách quan)\n" + soap["objective"] + "\n\n"
    md += "## A — Assessment (Đánh giá)\n" + soap["assessment"] + "\n\n"
    md += "## P — Plan (Kế hoạch)\n" + soap["plan"] + "\n\n"
    md += "---\n*Lưu ý: Tài liệu do AI hỗ trợ tổng hợp từ dữ liệu tự theo dõi, không thay thế chẩn đoán y khoa. Vui lòng tham khảo bác sĩ.*\n"
    if disease == "mental":
        md += "\n**Crisis:** Nếu có ý nghĩ tự hại, vui lòng liên hệ hotline 1800-1567 / 1900-1267 / 115 ngay và tìm chuyên gia.\n"
    return md
