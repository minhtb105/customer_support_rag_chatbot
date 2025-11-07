from functools import lru_cache
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from config import QA_DB_DIR, PDF_DB_DIR, TOP_K, EMBEDDING_MODEL
from models.llm_io import ContextItem
from typing import List

 
def load_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    qa_db = Chroma(persist_directory=QA_DB_DIR, embedding_function=embeddings)
    pdf_db = Chroma(persist_directory=PDF_DB_DIR, embedding_function=embeddings)
    
    return qa_db, pdf_db

@lru_cache(maxsize=1)
def cached_documents():
    qa_db, pdf_db = load_vectorstores()
    docs = []
    for db in [qa_db, pdf_db]:
        data = db.get()
        for i in range(len(data["ids"])):
            docs.append(Document(
                page_content=data["documents"][i],
                metadata=data["metadatas"][i]
            ))
            
    return docs

def retrieve_context(query: str, top_k: int=TOP_K):
    qa_db, pdf_db = load_vectorstores()
    
    # Semantic retriever (vector)
    vector_retriever = qa_db.as_retriever(search_kwargs={"k": top_k})
    vector_retriever_pdf = pdf_db.as_retriever(search_kwargs={"k": top_k})
    
    # Lexical retriever (BM25)
    docs = cached_documents()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    
    # Ensemble retriever (Hybrid)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever, vector_retriever_pdf],
        weights=[0.3, 0.35, 0.35]
    )
    
    docs = hybrid_retriever.invoke(query)
    grouped = {}
    
    for doc in docs:
        sid = doc.metadata.get("source_id")
        section_path = tuple(doc.metadata.get("section_path", []))
        key = (sid, section_path)
        grouped.setdefault(key, []).append(doc)
        
    results = []
    for (sid, section_path), parts in grouped.items():
        merged = " <CHUNK_BREAK> ".join(part.page_content for part in parts)
        results.append({
            "source_id": sid,
            "section_path": section_path,
            "content": merged
        })

    return results
def retrieve_context(query: str, top_k: int = TOP_K) -> List[ContextItem]:
    """
    Retrieve relevant contexts for a given query.
    Combines BM25 + vector + PDF retrievers (hybrid retrieval).
    Returns a list of validated ContextItem objects.
    """
    qa_db, pdf_db = load_vectorstores()
    
    # Semantic retrievers
    vector_retriever = qa_db.as_retriever(search_kwargs={"k": top_k})
    vector_retriever_pdf = pdf_db.as_retriever(search_kwargs={"k": top_k})
    
    # Lexical retriever (BM25)
    docs = cached_documents()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    
    # Hybrid (ensemble) retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever, vector_retriever_pdf],
        weights=[0.3, 0.35, 0.35]
    )
    
    docs = hybrid_retriever.invoke(query)
    grouped = {}
    
    # Group documents by source and section
    for doc in docs:
        sid = doc.metadata.get("source_id")
        section_path = tuple(doc.metadata.get("section_path", []))
        key = (sid, section_path)
        grouped.setdefault(key, []).append(doc)
        
    # Merge grouped parts and build ContextItem objects
    results: List[ContextItem] = []
    for (sid, section_path), parts in grouped.items():
        merged = " <CHUNK_BREAK> ".join(part.page_content for part in parts)
        results.append(
            ContextItem(
                source_id=str(sid),
                content=merged,
                dataset=doc.metadata.get("dataset", None),
                score=None  # score will be assigned after reranking
            )
        )
    
    return results
