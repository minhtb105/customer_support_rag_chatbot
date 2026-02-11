import time
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
    timings = {}
    
    # 0) Cache 
    t0 = time.perf_counter()
    cached: LLMOutput | None = cache.get(question)
    timings['cache_check'] = time.perf_counter() - t0
    
    if cached:
        t1 = time.perf_counter()
        formatted = format_answer_for_ui(cached)
        timings['format_answer'] = time.perf_counter() - t1
        
        return {
            "raw_answer": cached.model_dump(),
            "formatted_answer": formatted,
            "cache_hit": True,
            "cache_stats": cache.stats(),
            "timings": timings
        }

    # 1) Retrieve contexts (returns List[ContextItem])
    t2 = time.perf_counter()
    contexts = retrieve_context(question, top_k=top_k)
    timings['retrieve_context'] = time.perf_counter() - t2

    # 2) Rerank (expects List[ContextItem] and returns top ContextItem list)
    t3 = time.perf_counter()
    reranked: list[ContextItem] = rerank_contexts(question, contexts, top_n=3)
    timings['rerank_contexts'] = time.perf_counter() - t3

    # 3) Build LLMInput
    t4 = time.perf_counter()
    llm_input = LLMInput(query=question, contexts=reranked)
    timings['build_llm_input'] = time.perf_counter() - t4

    # 4) Call generator -> returns LLMOutput
    t5 = time.perf_counter()
    llm_output: LLMOutput = generate_answer(llm_input, model=model)
    timings['generate_answer'] = time.perf_counter() - t5
    
    # 5) Store result to cache (so future queries can hit)
    t6 = time.perf_counter()
    cache.put(question, llm_output)
    timings['cache_put'] = time.perf_counter() - t6

    # 6) Format for UI
    t7 = time.perf_counter()
    formatted = format_answer_for_ui(llm_output)
    timings['format_answer'] = time.perf_counter() - t7

    return {
        "raw_answer": llm_output.model_dump(),
        "formatted_answer": formatted,
        "cache_hit": False,
        "cache_stats": cache.stats(),
        "timings": timings
    }


if __name__ == "__main__":
    q = "What are the common causes of migraine headaches?"
    r = rag_chat(q)
    print("Cache stats:", r["cache_stats"])
    print("Formatted answer:\n", r["formatted_answer"])
