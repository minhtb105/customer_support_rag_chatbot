from functools import lru_cache
import os
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
from src.shared.config import PDF_DB_DIR, TOP_K, EMBEDDING_MODEL, EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL, BASE_DIR
from src.shared.models.llm_io import ContextItem
from typing import List, Optional
import os as _os
import sqlite3
import time as _time


def _get_embeddings():
    """Factory: OpenAI nếu cấu hình, fallback HuggingFace."""
    if EMBEDDING_PROVIDER == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        except Exception as e:
            # fallback log
            print(f"[retriever] OpenAIEmbeddings fail ({e}), fallback to HuggingFace")
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _resolve_db_dir(strategy: str) -> str:
    # Nếu dùng OpenAI embeddings, tách DB riêng để tránh lệch dimension 384 vs 1536
    if EMBEDDING_PROVIDER == "openai":
        base = BASE_DIR / "embeddings" / "pdf_db_openai"
        return str(base / strategy)
    return os.path.join(PDF_DB_DIR, strategy)


def load_vectorstores(strategy: str = "structure") -> Chroma:
    # No cache — provider may switch between openai/local and dimension mismatch would break cached instance
    embeddings = _get_embeddings()
    db_dir = _resolve_db_dir(strategy)
    _os.makedirs(db_dir, exist_ok=True)
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )

def cached_documents():
    # thử DB theo provider hiện tại, nếu rỗng thì fallback sang legacy pdf_db
    pdf_db = load_vectorstores()
    docs = []
    for db in [pdf_db]:
        try:
            data = db.get()
            for i in range(len(data["ids"])):
                docs.append(Document(
                    page_content=data["documents"][i],
                    metadata=data["metadatas"][i]
                ))
        except Exception:
            pass
    # Fallback: nếu chưa có doc nào và đang dùng openai, thử legacy local DB
    if not docs and EMBEDDING_PROVIDER == "openai":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings as _HF
            from langchain_chroma.vectorstores import Chroma as _Chroma
            legacy_dir = os.path.join(str(PDF_DB_DIR), "structure")
            if os.path.exists(legacy_dir):
                legacy = _Chroma(
                    persist_directory=legacy_dir,
                    embedding_function=_HF(model_name="all-MiniLM-L6-v2"),
                )
                data = legacy.get()
                for i in range(len(data["ids"])):
                    docs.append(Document(
                        page_content=data["documents"][i],
                        metadata=data["metadatas"][i]
                    ))
        except Exception:
            pass
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

def get_bm25():
    docs = cached_documents()
    retriever = LangchainBM25Retriever.from_documents(docs)
    retriever.k = TOP_K
    return retriever

def get_vector_retriever(strategy: str = "structure"):
    db = load_vectorstores(strategy)
    return db.as_retriever(search_kwargs={"k": TOP_K})

def _file_updated_at(source_id: str) -> Optional[str]:
    # lookup metadata_store files.updated_at for file matching source_id
    try:
        from src.shared.metadata_store import META_DB_PATH
        import sqlite3
        conn = sqlite3.connect(str(META_DB_PATH))
        conn.row_factory = sqlite3.Row
        # file_name stored as "filename::strategy", need LIKE
        row = conn.execute("SELECT updated_at FROM files WHERE file_name LIKE ?", (f"%{source_id}%",)).fetchone()
        if row and row["updated_at"]:
            try:
                return __import__("datetime").datetime.fromtimestamp(row["updated_at"]).isoformat()
            except Exception:
                return str(row["updated_at"])
        conn.close()
    except Exception:
        pass
    return None

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
    vector_docs = vector_retriever.invoke(query)
    vector_docs = normalize_docs(vector_docs)

    # BM25 lexical
    bm25_retriever = get_bm25()
    bm25_docs = bm25_retriever.invoke(query)
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
        pages = sorted({p for d in parts for p in (d.metadata.get("page_numbers") or [])})
        first_page = pages[0] if pages else None
        upd = _file_updated_at(str(parts[0].metadata.get("source_id")))
        # file_name from source_id
        file_name = parts[0].metadata.get("source_id")
        # try to find actual pdf name via dataset or file lookup; fallback source_id.pdf
        if file_name and not file_name.endswith(".pdf"):
            # attempt to resolve to real file
            try:
                from pathlib import Path as _P
                from src.shared.config import PDF_DIR as _PDF
                # search rglob for match
                for cand in _PDF.rglob(f"{file_name}.pdf"):
                    file_name = cand.name
                    break
                else:
                    file_name = file_name + ".pdf"
            except Exception:
                file_name = file_name + ".pdf"
        out.append(
            ContextItem(
                source_id=str(parts[0].metadata.get("source_id")),
                content=merged_text,
                section_path=parts[0].metadata.get("section_path"),
                page_numbers=pages,
                chunk_indices=[
                    d.metadata.get("chunk_index")
                    for d in parts
                    if d.metadata.get("chunk_index") is not None
                ],
                dataset=parts[0].metadata.get("dataset"),
                score=None,
                file_name=file_name,
                chunking_strategy=parts[0].metadata.get("chunking_strategy") or strategy,
                embedding_model=EMBEDDING_MODEL,
                chunk_hash=parts[0].metadata.get("chunk_hash"),
                updated_at=upd,
                first_page=first_page
            )
        )

    return out