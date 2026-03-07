"""
Embedding Utility Functions

Common utilities shared across embedding components.
"""

import numpy as np
from typing import List, Optional
from .config import NORMALIZE_EMBEDDINGS, EMBEDDING_DIMENSIONS


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector."""
    if not NORMALIZE_EMBEDDINGS:
        return embedding
    
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def batch_embed_texts(
    texts: List[str], 
    embed_fn,
    batch_size: int = 32
) -> List[np.ndarray]:
    """Batch embed texts for efficiency."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_fn(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings


def get_embedding_dimension(model_name: str) -> int:
    """Get embedding dimension for a model."""
    return EMBEDDING_DIMENSIONS.get(model_name, 384)


def truncate_embedding(
    embedding: np.ndarray, 
    target_dim: int
) -> np.ndarray:
    """Truncate embedding to target dimension."""
    if len(embedding) <= target_dim:
        return embedding
    
    return embedding[:target_dim]


def pad_embedding(
    embedding: np.ndarray, 
    target_dim: int
) -> np.ndarray:
    """Pad embedding to target dimension."""
    if len(embedding) >= target_dim:
        return embedding
    
    padding = target_dim - len(embedding)
    return np.pad(embedding, (0, padding), mode='constant')