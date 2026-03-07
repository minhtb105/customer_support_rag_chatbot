"""
Hybrid Retriever Component

Implements hybrid retrieval combining BM25 and vector search with reranking.
"""

from typing import List, Optional
from langchain_core.documents import Document
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .reranker import Reranker
from .utils import (
    should_merge, normalize_docs, merge_documents_by_section
)
from .config import DEFAULT_TOP_K, MERGE_THRESHOLD


class HybridRetriever:
    """Hybrid retriever combining BM25, vector search, and reranking."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        documents: List[Document],
        reranker: Optional[Reranker] = None,
        top_k: int = DEFAULT_TOP_K,
        merge_threshold: float = MERGE_THRESHOLD
    ):
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever(documents, k=top_k)
        self.reranker = reranker or Reranker()
        self.top_k = top_k
        self.merge_threshold = merge_threshold
    
    def retrieve(
        self, 
        query: str, 
        strategy: str = "structure",
        top_n: int = 3
    ) -> List[Document]:
        """Perform hybrid retrieval."""
        # Get documents from both retrievers
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k)
        bm25_docs = self.bm25_retriever.retrieve(query)
        
        # Normalize metadata
        vector_docs = normalize_docs(vector_docs)
        bm25_docs = normalize_docs(bm25_docs)
        
        # Combine results with deduplication
        combined_docs = self._combine_results(vector_docs, bm25_docs)
        
        # Merge documents if query suggests it
        if should_merge(query):
            combined_docs = merge_documents_by_section(combined_docs, query)
        
        # Rerank final results
        reranked_docs = self.reranker.rerank(query, combined_docs, top_n=top_n)
        
        return reranked_docs
    
    def _combine_results(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document]
    ) -> List[Document]:
        """Combine results from vector and BM25 retrievers."""
        # Deduplicate by document identity
        seen = set()
        combined = []
        
        # Add vector results first (higher priority)
        for doc in vector_docs:
            key = self._get_doc_key(doc)
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        # Add BM25 results (avoid duplicates)
        for doc in bm25_docs:
            key = self._get_doc_key(doc)
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        return combined
    
    def _get_doc_key(self, doc: Document) -> tuple:
        """Get unique key for document deduplication."""
        return (
            doc.metadata.get("source_id"),
            doc.metadata.get("chunk_index"),
            doc.page_content[:50]  # Small fingerprint
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both vector store and BM25."""
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Add to BM25
        self.bm25_retriever.add_documents(documents)