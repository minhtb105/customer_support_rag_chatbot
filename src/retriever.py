from functools import lru_cache
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config import EMBEDDINGS_DIR, TOP_K, HYBRID_ALPHA

 
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=EMBEDDINGS_DIR, embedding_function=embeddings)
    
    return db

@lru_cache(maxsize=1)
def cached_documents():
    db = load_vectorstore()
    docs = []
    all_data = db.get()
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    contents = all_data["documents"]

    for i in range(len(ids)):
        docs.append(Document(page_content=contents[i], metadata=metadatas[i]))
        
    return docs

def retrieve_context(query: str, top_k: int=TOP_K):
    db = load_vectorstore()
    embeddings = db._embedding_function
    
    # Semantic retriever (vector)
    vector_retriever = db.as_retriever(search_kwargs={"k": top_k})
    
    # Lexical retriever (BM25)
    docs = cached_documents()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    
    # Ensemble retriever (Hybrid)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - HYBRID_ALPHA, HYBRID_ALPHA]  # e.g. 0.4 lexical + 0.6 semantic
    )
    
    docs = hybrid_retriever.get_relevant_documents(query)
    grouped = {}
    
    for doc in docs:
        sid = doc.metadata.get("source_id")
        grouped.setdefault(sid, []).append(doc)
        
    results = []
    for sid, parts in grouped.items():
        merged = " <CHUNK_BREAK> ".join(part.page_content for part in parts)
        results.append({"source_id": sid, "content": merged})

    return results
