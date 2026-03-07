"""
BM25 Retriever Component

Implements BM25 lexical retrieval for complementing semantic search.
"""

from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from .config import DEFAULT_TOP_K


class BM25Retriever:
    """BM25 lexical retriever for complementing semantic search."""
    
    def __init__(
        self, 
        documents: List[Document],
        k: int = DEFAULT_TOP_K
    ):
        self.k = k
        self.retriever = self._create_retriever(documents)
    
    def _create_retriever(self, documents: List[Document]) -> BM25Retriever:
        """Create BM25 retriever from documents."""
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = self.k
        return retriever
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using BM25."""
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve documents with BM25: {e}")
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to BM25 index."""
        try:
            self.retriever.add_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to BM25: {e}")