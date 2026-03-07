"""
Embedding Configuration Module

Centralized configuration for embedding components.
"""

from src.config import EMBEDDING_MODEL

# Default configuration
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 32

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "text-embedding-ada-002": 1536
}

# Normalization settings
NORMALIZE_EMBEDDINGS = True
SIMILARITY_THRESHOLD = 0.8