from typing import Dict, Any
from config import (
    TOP_K, DEFAULT_MODEL,
    CAG_MAX_SIZE, CAG_TTL_SECONDS,
    CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD
)
from retriever import retrieve_context
from generator import generate_answer, rerank_contexts, format_answer_for_ui
from models.llm_io import ContextItem, LLMInput, LLMOutput
from cache import CAGHybridCache


# create a module-level cache instance (tune params in config)
cache = CAGHybridCache(
    max_size=CAG_MAX_SIZE,
    ttl_seconds=CAG_TTL_SECONDS,
    semantic_model_name=CAG_SEMANTIC_MODEL,
    semantic_threshold=CAG_SEMANTIC_THRESHOLD
)


def rag_chat(question: str, top_k: int = TOP_K, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    RAG pipeline with Cache-Augmented Generation (CAG).
    - Check cache first
    - If cache hit -> return cached LLMOutput (and formatted HTML)
    - Else -> retrieve contexts, rerank, call generator, store to cache.
    """
    # 0) Cache check
    cached: LLMOutput | None = cache.get(question)
    if cached:
        formatted = format_answer_for_ui(cached)
        return {
            "raw_answer": cached.model_dump(),
            "formatted_answer": formatted,
            "cache_hit": True,
            "cache_stats": cache.stats()
        }

    # 1) Retrieve contexts (returns List[ContextItem])
    contexts = retrieve_context(question, top_k=top_k)

    # 2) Rerank (expects List[ContextItem] and returns top ContextItem list)
    reranked: list[ContextItem] = rerank_contexts(question, contexts, top_n=3)

    # 3) Build LLMInput
    llm_input = LLMInput(query=question, contexts=reranked)

    # 4) Call generator -> returns LLMOutput
    llm_output: LLMOutput = generate_answer(llm_input, model=model)

    # 5) Store result to cache (so future queries can hit)
    cache.put(question, llm_output)

    # 6) Format for UI
    formatted = format_answer_for_ui(llm_output)

    return {
        "raw_answer": llm_output.model_dump(),
        "formatted_answer": formatted,
        "cache_hit": False,
        "cache_stats": cache.stats()
    }


if __name__ == "__main__":
    q = "What are the common causes of migraine headaches?"
    r = rag_chat(q)
    print("Cache stats:", r["cache_stats"])
    print("Formatted answer:\n", r["formatted_answer"])
