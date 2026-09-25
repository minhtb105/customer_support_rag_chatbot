"""
Safeguard guardrail (Diabetes demo): hard keyword gate + canned template.

- is_out_of_scope(text): pure function, keyword match (case-insensitive).
- SAFEGUARD_TEMPLATE: exact canned response; [X] = short topic label.
- Must be checked BEFORE any RAG/LLM call (see src/api/main.py::rag_query
  early-return when disease == diabetes).
- LLM intent classifier is future-work (see README).
"""

from __future__ import annotations

SAFEGUARD_TEMPLATE = (
    "Hệ thống này được thiết kế riêng để theo dõi đường huyết. "
    "Vấn đề {topic} nằm ngoài phạm vi hỗ trợ và an toàn y khoa. "
    "Vui lòng tham vấn bác sĩ tại buổi khám tới."
)

# Out-of-diabetes-scope keyword groups -> short topic label for [X].
_OUT_OF_SCOPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kháng sinh", ("kháng sinh", "khang sinh", "antibiotic", "amoxicillin", "augmentin", "cephalosporin")),
    ("tim mạch", ("tim mạch", "tim mach", "nhồi máu", "nhoi mau", "đau tim", "dau tim", "heart attack", "suy tim", "cardiology")),
    ("huyết áp", ("huyết áp", "huyet ap", "tăng huyết áp", "tang huyet ap", "cao huyết áp", "cao huyet ap", "hypertension", "hạ huyết áp", "ha huyet ap")),
    ("ung thư", ("ung thư", "ung thu", "cancer", "khối u", "khoi u", "hóa trị", "hoa tri", "xạ trị", "xa tri")),
    ("thận", ("suy thận", "suy than", "chạy thận", "chay than", "lọc máu", "loc mau", "nephro")),
    ("gan", ("viêm gan", "viem gan", "xơ gan", "xo gan", "hepatitis", "men gan")),
    ("phổi/hô hấp", ("hen suyễn", "hen suyen", "copd", "phổi", "phoi", "lao phổi", "lao phoi", "asthma")),
    ("thai sản", ("mang thai", "thai kỳ", "thai ky", "sinh con", "pregnancy", "cho con bú", "cho con bu")),
    ("tâm thần", ("trầm cảm", "tram cam", "tự tử", "tu tu", "suicide", "tâm thần", "tam than", "lo âu nặng", "rối loạn tâm", "roi loan tam")),
    ("xương khớp", ("gãy xương", "gay xuong", "thoái hóa khớp", "thoai hoa khop", "gout", "viêm khớp", "viem khop")),
    ("da liễu", ("vảy nến", "vay nen", "chàm", "cham", "mụn nặng", "nấm da", "nam da")),
    ("mắt", ("đục thủy tinh thể", "duc thuy tinh the", "glaucoma", "glôcôm", "mù mắt", "mu mat")),
    ("tiêm chủng", ("vắc xin", "vac xin", "vaccine", "tiêm phòng", "tiem phong")),
    ("covid/cúm", ("covid", "cúm a", "cum a", "sars", "h5n1")),
    ("thuốc ngoài ĐTĐ", ("thuốc tránh thai", "thuoc tranh thai", "thuốc ngủ", "thuoc ngu", "thuốc giảm đau mạnh", "morphine", "corticoid")),
)


def detect_topic(text: str) -> str | None:
    """Return short topic label if out-of-scope keywords hit, else None."""
    if not text:
        return None
    lowered = text.lower()
    for topic, keywords in _OUT_OF_SCOPE_KEYWORDS:
        for kw in keywords:
            if kw.lower() in lowered:
                return topic
    return None


def is_out_of_scope(text: str) -> bool:
    """Pure gate: True when text is outside diabetes scope."""
    return detect_topic(text) is not None


def safeguard_response(text: str, topic: str | None = None) -> str:
    """Render the exact canned safeguard template."""
    label = topic or detect_topic(text or "") or "này"
    return SAFEGUARD_TEMPLATE.format(topic=label)
