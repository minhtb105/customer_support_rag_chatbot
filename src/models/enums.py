"""
Models Enums

Centralized enums for the models module.
"""

from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    STRUCTURE = "structure"
    SENTENCE = "sentence"
    SLIDING = "sliding"
    SEMANTIC = "semantic"
    HYBRID_SECTION_SEMANTIC = "hybrid_section_semantic"
    AUTO = "auto"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class MemoryType(str, Enum):
    """Available memory types."""
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"
    WORKING = "working"