from functools import lru_cache
import os
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.config import PDF_DB_DIR, TOP_K, EMBEDDING_MODEL
from src.models.llm_io import ContextItem
from typing import List


@lru_cache(maxsize=4)
def load_vectorstores(strategy: str = "structure") -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    db_dir = os.path.join(PDF_DB_DIR, strategy)
    
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )

@lru_cache(maxsize=1)
def cached_documents():
    pdf_db = load_vectorstores()
    docs = []
    for db in [pdf_db]:
        data = db.get()
        for i in range(len(data["ids"])):
            docs.append(Document(
                page_content=data["documents"][i],
                metadata=data["metadatas"][i]
            ))

    return docs


def should_merge(query: str) -> bool:
    q = query.lower()

    return any(k in q for k in [
        "summary", "summarize", "overview",
        "key facts", "describe", "explain",
        "list", "what are", "how does"
    ])


def trim_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[Content truncated]"

def make_section_key(section):
    if not section:
        return None
    if isinstance(section, list):
        return tuple(section)
    return section

def deserialize_metadata(meta: dict) -> dict:
    if "section_path" in meta and isinstance(meta["section_path"], str):
        meta["section_path"] = meta["section_path"].split("|||")
        
    if "page_numbers" in meta and isinstance(meta["page_numbers"], str):
        meta["page_numbers"] = [
            int(x) for x in meta["page_numbers"].split("|||") if x
        ]
        
    return meta

def normalize_docs(docs):
    for d in docs:
        d.metadata = deserialize_metadata(d.metadata)
        
    return docs

@lru_cache(maxsize=1)
def get_bm25():
    docs = cached_documents()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = TOP_K
    
    return retriever

@lru_cache(maxsize=4)
def get_vector_retriever(strategy: str = "structure"):
    db = load_vectorstores(strategy)
    
    return db.as_retriever(search_kwargs={"k": TOP_K})

def retrieve_context(query: str, top_k: int = TOP_K, 
                     strategy: str = "structure") -> List[ContextItem]:
    """
    Retrieve relevant contexts for a given query.
    Combines BM25 + vector + PDF retrievers (hybrid retrieval).
    Returns a list of validated ContextItem objects.
    """
    
    # Semantic retrievers
    vector_retriever = get_vector_retriever(strategy)

    # semantic retriever
    vector_docs = vector_retriever.get_relevant_documents(query)
    vector_docs = normalize_docs(vector_docs)

    # BM25 lexical
    bm25_retriever = get_bm25()
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    bm25_docs = normalize_docs(bm25_docs)
        
    # combine results
    combined = vector_docs + bm25_docs

    # flatten and dedupe by doc identity + chunk_index
    seen = set()
    merged_list = []
    for d in combined:
        key = (
            d.metadata.get("source_id"),
            d.metadata.get("chunk_index"),
            d.page_content[:50]  # small fingerprint
        )
        if key not in seen:
            seen.add(key)
            merged_list.append(d)

    # sort: put vector matches first (optional)
    # or keep frequency, or average rank
    # here we just keep current order
    results = merged_list[:top_k * 2]  # optionally extend

    # group for merge
    grouped = {}
    for doc in results:
        sid = doc.metadata.get("source_id")
        section = make_section_key(doc.metadata.get("section_path"))    
        
        key = (sid, section) if should_merge(query) else id(doc)
        grouped.setdefault(key, []).append(doc)

    out: List[ContextItem] = []
    for key, parts in grouped.items():
        parts = sorted(
            parts,
            key=lambda d: d.metadata.get("chunk_index", 0)
        )
        merged_text = "\n\n---\n\n".join(p.page_content for p in parts)
        merged_text = trim_text(merged_text)
        out.append(
            ContextItem(
                source_id=str(parts[0].metadata.get("source_id")),
                content=merged_text,
                section_path=parts[0].metadata.get("section_path"),
                page_numbers=sorted({
                    p for d in parts
                    for p in (d.metadata.get("page_numbers") or [])
                }),
                chunk_indices=[
                    d.metadata.get("chunk_index")
                    for d in parts
                    if d.metadata.get("chunk_index") is not None
                ],
                dataset=parts[0].metadata.get("dataset"),
                score=None
            )
        )

    return out