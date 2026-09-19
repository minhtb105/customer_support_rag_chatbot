import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


def _env_first(*names: str, default: str = "") -> str:
    """Return the value of the first env var that is set and non-empty."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "docling_chunks"
PDF_DB_DIR = BASE_DIR / "embeddings" / "pdf_db"
META_DB_PATH = BASE_DIR / "metadata" / "metadata_store.db"

MAX_TOKENS = 480
CHUNK_OVERLAP = 200
BATCH_SIZE = 2000

SLIDING_WINDOW_TOKENS = 350
SLIDING_OVERLAP = 64
SEMANTIC_SIM_THRESHOLD = 0.9128
ATOMIC_TOKEN_SIZE = 120
SENTENCE_GROUP = 4

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # generator model
# Embedding: ưu tiên OpenAI, fallback về local nếu chưa cấu hình
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # "openai" | "local"
EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openai" else "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
}
EMBEDDING_DIMENSION = EMBEDDING_DIMENSIONS_MAP.get(EMBEDDING_MODEL, 1536)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKENIZER_MODEL = "bert-base-uncased"

# ==============================
#  LLM provider selection
# ==============================
# "openai" (default) -> OPENAI_API_KEY + OPENAI_BASE_URL
# "groq"             -> GROQ_API_KEY + https://api.groq.com/openai/v1
LLM_PROVIDER = _env_first("LLM_PROVIDER", "PROVIDER", default="openai").lower()
OPENAI_API_KEY = _env_first("OPENAI_API_KEY")
OPENAI_BASE_URL = _env_first("OPENAI_BASE_URL", default="https://api.openai.com/v1")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

TOP_K = 10
HYBRID_ALPHA = 0.6
MERGE_PEERS = True

# ==============================
#  Guideline Corpus (data pipeline)
# ==============================
# Thư mục phân loại theo nguồn — indexer sẽ quét đệ quy
GUIDELINE_SOURCES = ["who_iris", "ada", "aha_acc", "gold", "gina", "mhgap", "byt"]
DIABETES_PDF_DIR = PDF_DIR / "diabetes"
WHO_IRIS_DIR = PDF_DIR / "who_iris"
ADA_DIR = PDF_DIR / "ada"
BYT_DIR = PDF_DIR / "byt"
# Map nguồn -> thư mục (dùng cho crawler + indexer)
# Hypertension reuses aha_acc (AHA/ACC) + who_iris (HEARTS HTN); keep alias for clarity
HYPERTENSION_DIR = PDF_DIR / "hypertension"
RESPIRATORY_DIR = PDF_DIR / "respiratory"
MENTAL_DIR = PDF_DIR / "mental"
GUIDELINE_SOURCE_DIRS = {
    "who_iris": WHO_IRIS_DIR,
    "ada": ADA_DIR,
    "aha_acc": PDF_DIR / "aha_acc",
    "gold": PDF_DIR / "gold",
    "gina": PDF_DIR / "gina",
    "mhgap": PDF_DIR / "mhgap",
    "byt": BYT_DIR,
    "diabetes": DIABETES_PDF_DIR,
    "hypertension": HYPERTENSION_DIR,
    "respiratory": RESPIRATORY_DIR,
    "mental": MENTAL_DIR,
}

# ==============================
#  Vitals tracking — Diabetes (A) + Hypertension / Respiratory / Mental (2B) — unified DB
# ==============================
GLUCOSE_THRESHOLDS_MGDL = {
    "fasting_normal_max": 100,
    "fasting_prediabetes_max": 125,
    "fasting_diabetes": 126,
    "postprandial_normal_max": 140,
    "postprandial_prediabetes_max": 199,
    "postprandial_diabetes": 200,
    "random_critical_high": 300,
    "hypoglycemia": 70,
}
# AHA/ACC 2025 Hypertension thresholds (mmHg)
BP_THRESHOLDS_MMHG = {
    "normal_sys_max": 120,
    "normal_dia_max": 80,
    "elevated_sys_min": 120, "elevated_sys_max": 129, "elevated_dia_max": 80,
    "stage1_sys_min": 130, "stage1_sys_max": 139, "stage1_dia_min": 80, "stage1_dia_max": 89,
    "stage2_sys_min": 140, "stage2_dia_min": 90,
    "crisis_sys": 180, "crisis_dia": 120,
    # Home BP target for treated patients
    "home_target_sys": 130, "home_target_dia": 80,
}
# Asthma/COPD — peak flow zones (% personal best) + GOLD stages
RESPIRATORY_THRESHOLDS = {
    "peak_flow_green_min": 80,  # % — Go zone
    "peak_flow_yellow_min": 50, # % — Caution
    "peak_flow_red_max": 50,    # % — Medical alert
    "gold_stages": ["GOLD_1_mild", "GOLD_2_moderate", "GOLD_3_severe", "GOLD_4_very_severe"],
    "cat_mild_max": 10, "cat_moderate_max": 20, # COPD Assessment Test
}
# Mental health — PHQ-9 / GAD-7 + crisis keywords (PII redact)
MENTAL_HEALTH_THRESHOLDS = {
    "phq9_minimal_max": 4, "phq9_mild_max": 9, "phq9_moderate_max": 14, "phq9_moderately_severe_max": 19, "phq9_severe_min": 20,
    "gad7_minimal_max": 4, "gad7_mild_max": 9, "gad7_moderate_max": 14, "gad7_severe_min": 15,
    "crisis_keywords": ["tự tử", "tu tu", "suicide", "kill myself", "tự hại", "self-harm", "tự làm hại", "muốn chết", "want to die"],
    "crisis_hotline_vn": "1800-1567 (Bảo vệ trẻ em) / 1900-1267 (Sức khỏe tâm thần) / 115 (Cấp cứu)",
}
# Unified vitals DB (stores glucose + bp + respiratory + mood with disease_type column)
VITALS_DB_PATH = BASE_DIR / "metadata" / "vitals.db"
# Legacy path kept for backward compatibility (glucose_logs.db → vitals.db migration)
GLUCOSE_DB_PATH = BASE_DIR / "metadata" / "glucose_logs.db"
# PII redact — fields to mask before logging/storage
PII_REDACT_FIELDS = {"name", "phone", "email", "address", "cmnd", "cccd"}

# ==============================
#  Auth & RBAC (mới)
# ==============================
AUTH_DB_PATH = BASE_DIR / "metadata" / "auth.db"
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-please-set-env")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
# Cookie settings cho httpOnly
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes")  # true khi https
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")  # lax | strict | none
# Roles: user, doctor, pharmacist, specialist, admin
VALID_ROLES = {"user", "doctor", "pharmacist", "specialist", "admin"}
EXPERT_ROLES = {"doctor", "pharmacist", "specialist"}

# ==============================
#  HILT — RAG Confidence & Review Routing
# ==============================
# Ngưỡng 0-5 cho từng metric (LLM-as-judge). Nếu < ngưỡng → low confidence → cần HILT
RAG_EVAL_THRESHOLDS = {
    "faithfulness": float(os.getenv("HILT_FAITHFULNESS_THRESHOLD", "3.5")),
    "context_precision": float(os.getenv("HILT_CONTEXT_PRECISION_THRESHOLD", "3.5")),
    "context_recall": float(os.getenv("HILT_CONTEXT_RECALL_THRESHOLD", "3.5")),
    "answer_relevance": float(os.getenv("HILT_ANSWER_RELEVANCE_THRESHOLD", "3.5")),
}
# Routing: metric thấp → role nào duyệt
HILT_ROUTING = {
    "faithfulness": os.getenv("HILT_ROUTE_FAITHFULNESS", "doctor"),      # factual → doctor
    "context_precision": os.getenv("HILT_ROUTE_PRECISION", "specialist"), # retrieval precision → specialist
    "context_recall": os.getenv("HILT_ROUTE_RECALL", "specialist"),
    "answer_relevance": os.getenv("HILT_ROUTE_RELEVANCE", "doctor"),
    # fallback khi không xác định
    "default": os.getenv("HILT_ROUTE_DEFAULT", "doctor"),
}
# Nếu có từ khoá dược → ưu tiên pharmacist
PHARMACIST_KEYWORDS = ["thuốc", "medication", "drug", "dosage", "liều", "tác dụng phụ", "side effect", "pharmacist", "dược"]

# ==============================
#  Monitoring — Guideline & Safety Agents
# ==============================
MONITORING_DB_PATH = BASE_DIR / "metadata" / "monitoring.db"
STAGING_DIR = BASE_DIR / "data" / "raw" / "staging"
# intervals
MONITOR_GUIDELINE_HEAD_INTERVAL = os.getenv("MONITOR_GUIDELINE_INTERVAL", "monthly")  # monthly HEAD, quarterly deep
MONITOR_SAFETY_DAILY_AT = os.getenv("MONITOR_SAFETY_DAILY_AT", "02:00")
MONITOR_SAFETY_WEEKLY_AT = os.getenv("MONITOR_SAFETY_WEEKLY_AT", "monday-03:00")
MONITOR_BYT_INTERVAL = os.getenv("MONITOR_BYT_INTERVAL", "weekly")
# FDA anonymous, cache
FDA_API_BASE = os.getenv("FDA_API_BASE", "https://api.fda.gov")
FDA_CACHE_TTL_SECONDS = int(os.getenv("FDA_CACHE_TTL_SECONDS", "3600"))  # 1h
FDA_RATE_LIMIT_PER_MIN = int(os.getenv("FDA_RATE_LIMIT_PER_MIN", "240"))
# superseded retention
MONITOR_SUPERSEDED_RETENTION_DAYS = int(os.getenv("MONITOR_SUPERSEDED_RETENTION_DAYS", "30"))
# Allowed domains for SSRF allowlist (mirrors db.ALLOWED_MONITOR_DOMAINS)
ALLOWED_MONITOR_DOMAINS = {
    "iris.who.int",
    "diabetesjournals.org",
    "ada-journals.cld.bz",
    "www.ahajournals.org",
    "ahajournals.org",
    "goldcopd.org",
    "ginasthma.org",
    "api.fda.gov",
    "www.fda.gov",
    "moh.gov.vn",
    "dav.gov.vn",
    "thuvienphapluat.vn",
    "kcb.vn",
    "daithaoduong.kcb.vn",
    "bvcdn.org.vn",
    "ghoapi.azureedge.net",
    "www.who.int",
}

# ==============================
#  WHO GHO OData snapshot — Dual-Storage (textualized RAG + SQLite tool)
# ==============================
GHO_API_BASE = os.getenv("GHO_API_BASE", "https://ghoapi.azureedge.net/api")
GHO_INDICATORS = {
    "NCD_DIABETES_PREVALENCE_CRUDE": {"vi": "tỷ lệ lưu hành bệnh đái tháo đường (ước tính thô)", "unit": "%"},
    "NCD_DIABETES_PREVALENCE_AGESTD": {"vi": "tỷ lệ lưu hành bệnh đái tháo đường (chuẩn hóa tuổi)", "unit": "%"},
    "NCD_DIABETES_TREATMENT_CRUDE": {"vi": "độ bao phủ điều trị đái tháo đường (ước tính thô)", "unit": "%"},
    "NCD_DIABETES_TREATMENT_AGESTD": {"vi": "độ bao phủ điều trị đái tháo đường (chuẩn hóa tuổi)", "unit": "%"},
}
GHO_SNAPSHOT_DIR = BASE_DIR / "data" / "raw" / "gho"
GHO_TEXT_DIR = BASE_DIR / "data" / "raw" / "gho_text"
GHO_DB_PATH = BASE_DIR / "data" / "processed" / "gho_stats.db"


# ==============================
#  API (Hướng C)
# ==============================
API_TITLE = "WHO-RAG Infrastructure API"
API_VERSION = "1.0.0"
API_PREFIX = "/v1"


# ==============================
#  CAG (Cache-Augmented Generation)
# ==============================
CAG_MAX_SIZE = 1024                  # number of maximum entry in cache
CAG_TTL_SECONDS = 60 * 60 * 24       # 24h
CAG_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CAG_SEMANTIC_THRESHOLD = 0.82


# ==============================
#  Local Tracing & Prompt Versioning (thay LangSmith)
# ==============================
TRACING_DB_PATH = BASE_DIR / "metadata" / "tracing.db"
TRACING_RETENTION_DAYS = int(os.getenv("TRACING_RETENTION_DAYS", "30"))
TRACING_PAGE_SIZE = int(os.getenv("TRACING_PAGE_SIZE", "10"))  # max 10 per spec
PROMPT_ACTIVE_CACHE_TTL_SECONDS = int(os.getenv("PROMPT_ACTIVE_CACHE_TTL_SECONDS", "300"))
# RAGAS on-demand per trace (admin bật mới tính)
RAGAS_ON_DEMAND = os.getenv("RAGAS_ON_DEMAND", "true").lower() in ("true","1","yes")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", DEFAULT_MODEL)

PROMPT_HUB_REPO_PREFIX = os.getenv("PROMPT_HUB_REPO_PREFIX", "medical-support")
PROMPT_HUB_CACHE_TTL_SECONDS = PROMPT_ACTIVE_CACHE_TTL_SECONDS
