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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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
