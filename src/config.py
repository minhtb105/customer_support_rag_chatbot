import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "docling_chunks"
QA_DB_DIR = BASE_DIR / "embeddings" / "qa_db"
PDF_DB_DIR = BASE_DIR / "embeddings" / "pdf_db"
META_DB_PATH = BASE_DIR / "metadata" / "metadata_store.db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
BATCH_SIZE = 2000

DEFAULT_MODEL = "llama-3.1-8b-instant"  # generator model
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
TOKENIZER_MODEL = "bert-base-uncased"

TOP_K = 10
HYBRID_ALPHA=0.6

MERGE_PEERS = True
