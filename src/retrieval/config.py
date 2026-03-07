"""
Retrieval Configuration Module

Centralized configuration for retrieval components.
"""

from src.config import TOP_K, EMBEDDING_MODEL, RERANKER_MODEL

# Retrieval strategies
RETRIEVAL_STRATEGIES = [
    "structure",
    "sliding", 
    "semantic",
    "hybrid_section_semantic"
]

# Default configuration
DEFAULT_TOP_K = TOP_K
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL
DEFAULT_RERANKER_MODEL = RERANKER_MODEL

# Retrieval thresholds
MERGE_THRESHOLD = 0.8
TRIM_MAX_CHARS = 4000

# Cache settings
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 3600  # 1 hour