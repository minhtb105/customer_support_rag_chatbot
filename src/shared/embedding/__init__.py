"""
Embedding Module — only EmbeddingAdapter is canonical.
Legacy utils/config (duplicate of src/config.EMBEDDING_*) removed 2026-09-07.
"""

from .adapter import EmbeddingAdapter

__all__ = [
    'EmbeddingAdapter',
]