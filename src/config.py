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
GUIDELINE_SOURCE_DIRS = {
    "who_iris": WHO_IRIS_DIR,
    "ada": ADA_DIR,
    "aha_acc": PDF_DIR / "aha_acc",
    "gold": PDF_DIR / "gold",
    "gina": PDF_DIR / "gina",
    "mhgap": PDF_DIR / "mhgap",
    "byt": BYT_DIR,
    "diabetes": DIABETES_PDF_DIR,
}

# ==============================
#  Glucose tracking (Hướng A)
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
GLUCOSE_DB_PATH = BASE_DIR / "metadata" / "glucose_logs.db"

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
#  LangSmith (tracing / observability / prompt hub)
# ==============================
# Supports both new (LANGSMITH_*) and legacy (LANGCHAIN_*) variable names
LANGSMITH_TRACING = _env_first("LANGSMITH_TRACING", default="false").lower() in ("true", "1", "yes")
LANGSMITH_API_KEY = _env_first("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = _env_first("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", default="customer-support-rag")
LANGSMITH_ENDPOINT = _env_first("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT",
                                default="https://api.smith.langchain.com")

# Prompt Hub repository prefix: prompts are stored as "<owner>/<repo-prefix>-<prompt-name>"
PROMPT_HUB_REPO_PREFIX = os.getenv("PROMPT_HUB_REPO_PREFIX", "medical-support")
# How long (seconds) a prompt pulled from LangSmith Hub stays cached locally
PROMPT_HUB_CACHE_TTL_SECONDS = int(os.getenv("PROMPT_HUB_CACHE_TTL_SECONDS", "300"))
