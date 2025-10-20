from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDINGS_DIR, TOP_K 



def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=EMBEDDINGS_DIR, embedding_function=embeddings)
    
    return db

def retrieve_context(query: str, top_k: int=TOP_K):
    db = load_vectorstore()
    docs = db.similarity_search(query, k=top_k)
    grouped = {}
    
    for doc in docs:
        sid = doc.metadata.get("source_id")
        grouped.setdefault(sid, []).append(doc)
        
    results = []
    for sid, parts in grouped.items():
        merged = " <CHUNK_BREAK> ".join(part.page_content for part in parts)
        results.append({"source_id": sid, "content": merged})

    return results
