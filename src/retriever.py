from functools import lru_cache
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from config import PDF_DB_DIR, TOP_K, EMBEDDING_MODEL
from models.llm_io import ContextItem
from typing import List


def load_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    pdf_db = Chroma(persist_directory=PDF_DB_DIR,
                    embedding_function=embeddings)

    return pdf_db


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


def retrieve_context(query: str, top_k: int = TOP_K) -> List[ContextItem]:
    """
    Retrieve relevant contexts for a given query.
    Combines BM25 + vector + PDF retrievers (hybrid retrieval).
    Returns a list of validated ContextItem objects.
    """
    pdf_db = load_vectorstores()

    # Semantic retrievers
    vector_retriever = pdf_db.as_retriever(search_kwargs={"k": top_k})

    # Lexical retriever (BM25)
    docs = cached_documents()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    # Hybrid (ensemble) retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )

    retrieved_docs = hybrid_retriever.invoke(query)

    merge = should_merge(query)
    grouped = {}

    for doc in retrieved_docs:
        sid = doc.metadata.get("source_id")
        section = doc.metadata.get("section_path")

        key = (sid, section) if merge else id(doc)
        grouped.setdefault(key, []).append(doc)

    results: List[ContextItem] = []

    for key, parts in grouped.items():
        parts = sorted(
            parts,
            key=lambda d: d.metadata.get("chunk_index", 0)
        )

        merged_text = "\n\n---\n\n".join(p.page_content for p in parts)
        merged_text = trim_text(merged_text)

        results.append(
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

    return results
