"""
Vector Store Component

Encapsulates vector database operations with proper error handling and configuration.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K


class VectorStore:
    """Vector store wrapper with configuration management."""
    
    def __init__(
        self, 
        persist_directory: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize vector store
        self.vectorstore = self._create_vectorstore()
    
    def _create_vectorstore(self) -> Chroma:
        """Create or load vector store."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def add_documents(
        self, 
        documents: List[Document],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            return self.vectorstore.add_documents(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to vector store: {e}")
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search."""
        search_k = k or self.top_k
        try:
            return self.vectorstore.similarity_search(
                query=query,
                k=search_k,
                filter=filter
            )
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")
    
    def as_retriever(self, **kwargs):
        """Get retriever interface."""
        return self.vectorstore.as_retriever(**kwargs)
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.vectorstore.delete(ids=ids)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete documents: {e}")
    
    def get(self, ids: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Get documents from vector store."""
        try:
            return self.vectorstore.get(ids=ids, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to get documents: {e}")
    
    def persist(self):
        """Persist vector store to disk."""
        try:
            self.vectorstore.persist()
        except Exception as e:
            raise RuntimeError(f"Failed to persist vector store: {e}")