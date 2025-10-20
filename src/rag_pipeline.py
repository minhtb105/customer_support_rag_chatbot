from retriever import retrieve_context
from generator import generate_answer, rerank_contexts


def rag_chat(question, top_k: int = 10, model: str = "llama-3.1-8b-instant"):
    """
    RAG pipeline: Given a question -> retrieve context -> generate answer
    """
    contexts = retrieve_context(question, top_k=top_k)
    reranked = rerank_contexts(question, contexts, top_n=3)
    answer = generate_answer(question, reranked, model=model)

    return {"question": question, "answer": answer, "contexts": reranked}

if __name__ == "__main__":
    user_query = "What are the common causes of migraine headaches?"
    result = rag_chat(user_query)

    print("\nChatbot Answer:")
    print(result["answer"])

    print("\nContext Sources:")
    for i, c in enumerate(result["contexts"]):
        print(f"[{i+1}] {c['content'][:-1]}...")
        