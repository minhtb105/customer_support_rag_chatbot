"""
Reranker Component

Implements cross-encoder reranking for improved retrieval quality.
"""

from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from .config import DEFAULT_RERANKER_MODEL


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_n: int = 3
    ) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Create pairs for reranking
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])
        
        # Sort by score and return top N
        ranked_docs = sorted(
            documents, 
            key=lambda x: x.metadata.get("rerank_score", 0), 
            reverse=True
        )
        
        return ranked_docs[:top_n]
    
    def get_scores(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[float]:
        """Get reranking scores for documents."""
        if not documents:
            return []
        
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        return [float(score) for score in scores]