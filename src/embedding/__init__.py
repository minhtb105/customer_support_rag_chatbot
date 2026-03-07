"""
Embedding Module Interface

Provides clean API for embedding components with proper imports and exports.
Follows the same pattern as other modules for consistency.
"""

# Import core embedding components
from .adapter import EmbeddingAdapter

# Import utility functions
from .utils import (
    normalize_embedding,
    cosine_similarity,
    batch_embed_texts
)

# Import configuration
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BATCH_SIZE
)

__all__ = [
    # Core embedding classes
    'EmbeddingAdapter',
    
    # Utility functions
    'normalize_embedding',
    'cosine_similarity',
    'batch_embed_texts',
    
    # Configuration
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_BATCH_SIZE'
]