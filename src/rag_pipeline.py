import time
from typing import Dict, Any, List
from config import (
    TOP_K, DEFAULT_MODEL,
    CAG_MAX_SIZE, CAG_TTL_SECONDS,
    CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD
)
from retriever import retrieve_context
from generator import generate_answer, rerank_contexts, format_answer_for_ui
from models.llm_io import ContextItem, LLMInput, LLMOutput
from cache import CAGHybridCache
from memory import (
    get_short_term_memory, get_episodic_memory, get_long_term_memory
)


# create a module-level cache instance
cache = CAGHybridCache(
    max_size=CAG_MAX_SIZE,
    ttl_seconds=CAG_TTL_SECONDS,
    semantic_model_name=CAG_SEMANTIC_MODEL,
    semantic_threshold=CAG_SEMANTIC_THRESHOLD
)

# Memory instances
short_term_memory = get_short_term_memory()
episodic_memory = get_episodic_memory()
long_term_memory = get_long_term_memory()


def rag_chat(question: str, top_k: int = TOP_K, model: str = DEFAULT_MODEL, 
             user_id: str = "default_user") -> Dict[str, Any]:
    """
    RAG pipeline with Cache-Augmented Generation (CAG) and multi-layer memory.
    - Check cache first
    - If cache hit -> return cached LLMOutput (and formatted HTML)
    - Else -> retrieve contexts, rerank, call generator, store to cache.
    - Update all memory layers with the interaction.
    """
    timings = {}
    
    # Update memory instances for current user
    global short_term_memory, episodic_memory, long_term_memory
    short_term_memory = get_short_term_memory()
    episodic_memory = get_episodic_memory()
    long_term_memory = get_long_term_memory(user_id)
    
    # 0) Cache 
    t0 = time.perf_counter()
    cached: LLMOutput | None = cache.get(question)
    timings['cache_check'] = time.perf_counter() - t0
    
    if cached:
        t1 = time.perf_counter()
        formatted = format_answer_for_ui(cached)
        timings['format_answer'] = time.perf_counter() - t1
        
        # Update memories with cached response
        _update_memories(question, cached.answer, user_id)
        
        return {
            "raw_answer": cached.model_dump(),
            "formatted_answer": formatted,
            "cache_hit": True,
            "cache_stats": cache.stats(),
            "memory_stats": _get_memory_stats(),
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

    # 3) Build LLMInput with memory context
    t4 = time.perf_counter()
    memory_context = _build_memory_context(question, user_id)
    llm_input = LLMInput(query=question, contexts=[c.model_dump() for c in reranked])
    # Add memory context to metadata for generator
    llm_input.contexts.extend(memory_context)
    timings['build_llm_input'] = time.perf_counter() - t4

    # 4) Call generator -> returns LLMOutput
    t5 = time.perf_counter()
    llm_output: LLMOutput = generate_answer(llm_input, model=model)
    timings['generate_answer'] = time.perf_counter() - t5
    
    # 5) Store result to cache (so future queries can hit)
    t6 = time.perf_counter()
    cache.put(question, llm_output)
    timings['cache_put'] = time.perf_counter() - t6

    # 6) Update all memory layers
    t7 = time.perf_counter()
    _update_memories(question, llm_output.answer, user_id)
    timings['update_memories'] = time.perf_counter() - t7

    # 7) Format for UI
    t8 = time.perf_counter()
    formatted = format_answer_for_ui(llm_output)
    timings['format_answer'] = time.perf_counter() - t8

    return {
        "raw_answer": llm_output.model_dump(),
        "formatted_answer": formatted,
        "cache_hit": False,
        "cache_stats": cache.stats(),
        "memory_stats": _get_memory_stats(),
        "timings": timings
    }


def _build_memory_context(question: str, user_id: str) -> List[Dict[str, Any]]:
    """Build memory context from all memory layers."""
    memory_context = []
    
    # Short-term memory context
    short_term_context = short_term_memory.get_context_string(max_tokens=500)
    if short_term_context:
        memory_context.append({
            'source_id': 'short_term_memory',
            'content': f"Recent conversation context: {short_term_context}",
            'score': 0.9,
            'dataset': 'short_term_memory'
        })
    
    # Episodic memory context
    episodic_context = episodic_memory.get_context_from_summaries(question, max_tokens=300)
    if episodic_context:
        memory_context.append({
            'source_id': 'episodic_memory',
            'content': f"Relevant conversation summaries: {episodic_context}",
            'score': 0.8,
            'dataset': 'episodic_memory'
        })
    
    # Long-term memory context
    long_term_facts = long_term_memory.retrieve_facts(question, top_k=3)
    if long_term_facts:
        facts_text = " ".join([f.text for f in long_term_facts])
        memory_context.append({
            'source_id': 'long_term_memory',
            'content': f"User medical history: {facts_text}",
            'score': 0.7,
            'dataset': 'long_term_memory'
        })
    
    return memory_context


def _update_memories(question: str, answer: str, user_id: str):
    """Update all memory layers with the new interaction."""
    # Update short-term memory
    short_term_memory.add_message("user", question)
    short_term_memory.add_message("assistant", answer, metadata={"user_id": user_id})
    
    # Update episodic memory
    episodic_memory.add_message("user", question, metadata={"user_id": user_id})
    episodic_memory.add_message("assistant", answer, metadata={"user_id": user_id})
    
    # Update long-term memory with extracted facts
    # Extract facts from both question and answer
    combined_text = f"User: {question}\nAssistant: {answer}"
    long_term_memory.store_fact(combined_text, source="conversation", 
                               metadata={"user_id": user_id, "interaction_type": "Q&A"})


def _get_memory_stats() -> Dict[str, Any]:
    """Get statistics from all memory layers."""
    return {
        'short_term': short_term_memory.get_stats(),
        'episodic': episodic_memory.get_stats(),
        'long_term': long_term_memory.get_stats()
    }


if __name__ == "__main__":
    q = "What are the common causes of migraine headaches?"
    r = rag_chat(q)
    print("Timings:", r["timings"])
    print("Formatted answer:\n", r["formatted_answer"])
