from retriever import retrieve_context
from config import TOP_K, DEFAULT_MODEL 
from generator import generate_answer, rerank_contexts, format_answer_for_ui
from models.llm_io import LLMInput, LLMOutput, ContextItem


def rag_chat(question, top_k: int = TOP_K, model: str = DEFAULT_MODEL):
    """
    RAG pipeline: Given a question -> retrieve context -> generate answer.
    Returns both raw structured output and formatted HTML for UI.
    """
    # Retrieve and rerank contexts
    contexts = retrieve_context(question, top_k=top_k)
    reranked = rerank_contexts(question, contexts, top_n=3)

    # Convert to ContextItem objects for validation
    context_items = [ContextItem(**ctx) for ctx in reranked]

    # Create LLMInput
    llm_input = LLMInput(query=question, contexts=context_items)

    # Generate model answer (returns LLMOutput)
    llm_output: LLMOutput = generate_answer(llm_input, model=model)

    # Format answer for UI display (HTML-safe)
    formatted_output = format_answer_for_ui(llm_output)

    # Return both raw (JSON-friendly) and formatted output
    return {
        "raw_answer": llm_output.model_dump(),   # validated JSON data
        "formatted_answer": formatted_output     # HTML-ready for frontend
    }
    
    
if __name__ == "__main__":
    user_query = "What are the common causes of migraine headaches?"
    result = rag_chat(user_query)

    print("\nChatbot Answer:")
    print(result["answer"])
        