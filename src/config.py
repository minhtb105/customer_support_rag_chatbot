import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "docling_chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "chroma_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

DEFAULT_MODEL = "llama-3.1-8b-instant"
TOKENIZER_MODEL = "bert-base-uncased"

TOP_K = 10
HYBRID_ALPHA=0.6

MERGE_PEERS = True
