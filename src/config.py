import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
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

DEFAULT_MODEL = "llama-3.1-8b-instant"  # generator model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKENIZER_MODEL = "bert-base-uncased"

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
