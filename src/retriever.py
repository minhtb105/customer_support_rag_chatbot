import logging
from pathlib import Path
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "chroma_index"

def load_vectorstore(persist_dir=EMBEDDINGS_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return db

def retrieve_context(query: str, top_k: int=10):
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

if __name__ == "__main__":
    query = "What are the symptoms of diabetes?"
    contexts = retrieve_context(query, top_k=10)
    logging.info("Contexts found:")
    for i, c in enumerate(contexts):
        print(f"\n[{i+1}] {c['content'][:-1]}...")
