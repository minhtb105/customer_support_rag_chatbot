"""
Retrieval Module Interface

Provides clean API for retrieval components with proper imports and exports.
Follows the same pattern as memory module for consistency.
"""

# Import core retrieval components
from .vector_store import VectorStore
from .reranker import Reranker
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever

# Import utility functions and classes
from .utils import (
    should_merge, 
    trim_text, 
    make_section_key, 
    deserialize_metadata,
    normalize_docs
)

# Import configuration and types
from .config import (
    TOP_K, 
    EMBEDDING_MODEL, 
    RERANKER_MODEL,
    RETRIEVAL_STRATEGIES
)

__all__ = [
    # Core retrieval classes
    'VectorStore',
    'Reranker', 
    'BM25Retriever',
    'HybridRetriever',
    
    # Utility functions
    'should_merge',
    'trim_text',
    'make_section_key', 
    'deserialize_metadata',
    'normalize_docs',
    
    # Configuration
    'TOP_K',
    'EMBEDDING_MODEL',
    'RERANKER_MODEL',
    'RETRIEVAL_STRATEGIES'
]