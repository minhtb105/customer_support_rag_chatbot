from retriever import retrieve_context
from config import TOP_K, DEFAULT_MODEL 
from generator import generate_answer, rerank_contexts, format_answer_for_ui


def rag_chat(question, top_k: int = TOP_K, model: str = DEFAULT_MODEL):
    """
    RAG pipeline: Given a question -> retrieve context -> generate answer
    """
    # Retrieve and rerank
    contexts = retrieve_context(question, top_k=top_k)
    reranked = rerank_contexts(question, contexts, top_n=3)
    
    # Generate model answer
    answer = generate_answer(question, reranked, model=model)

    # Simulate citation logic 
    cited_sources = [ctx["source_id"] for ctx in reranked]

    # Build response object
    answer_data = {
        "answer": answer,
        "cited_sources": cited_sources,
        "contexts": reranked
    }

    # Format final HTML for UI
    formatted_output = format_answer_for_ui(answer_data)

    return {
        "raw_answer": answer_data, # JSON-friendly
        "formatted_answer": formatted_output  # HTML-ready for frontend
    }

if __name__ == "__main__":
    user_query = "What are the common causes of migraine headaches?"
    result = rag_chat(user_query)

    print("\nChatbot Answer:")
    print(result["answer"])
        