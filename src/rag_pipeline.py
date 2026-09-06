import time
from typing import Dict, Any, List

try:
    from config import TOP_K, DEFAULT_MODEL, CAG_MAX_SIZE, CAG_TTL_SECONDS, CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD, EMBEDDING_MODEL
except ImportError:
    from src.config import TOP_K, DEFAULT_MODEL, CAG_MAX_SIZE, CAG_TTL_SECONDS, CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD, EMBEDDING_MODEL

try:
    from retriever import retrieve_context
    from generator import generate_answer, rerank_contexts, format_answer_for_ui, detect_tone_and_temp
    from models.llm_io import ContextItem, LLMInput, LLMOutput
    from cache import CAGHybridCache
    from memory import get_short_term_memory, get_episodic_memory, get_long_term_memory
except ImportError:
    from src.retriever import retrieve_context
    from src.generator import generate_answer, rerank_contexts, format_answer_for_ui, detect_tone_and_temp
    from src.models.llm_io import ContextItem, LLMInput, LLMOutput
    from src.cache import CAGHybridCache
    from src.memory import get_short_term_memory, get_episodic_memory, get_long_term_memory

try:
    from observability.local_tracing import start_trace, end_trace, start_span, finish_span, add_span_chunks
    from observability.tracing_db import add_span_chunks as db_add_chunks  # alias
    from prompt_manager import get_prompt_version
except ImportError:
    from src.observability.local_tracing import start_trace, end_trace, start_span, finish_span, add_span_chunks
    from src.observability.tracing_db import add_span_chunks as db_add_chunks
    from src.prompt_manager import get_prompt_version

cache = CAGHybridCache(max_size=CAG_MAX_SIZE, ttl_seconds=CAG_TTL_SECONDS, semantic_model_name=CAG_SEMANTIC_MODEL, semantic_threshold=CAG_SEMANTIC_THRESHOLD)
short_term_memory = get_short_term_memory()
episodic_memory = get_episodic_memory()
long_term_memory = get_long_term_memory()

def rag_chat(question: str, top_k: int = TOP_K, user_id: str = "default_user", username: str | None = None, model: str = DEFAULT_MODEL, **kwargs) -> Dict[str, Any]:
    timings = {}
    # tone detection early for tracing
    try:
        tone, _, _ = detect_tone_and_temp(question)
    except Exception:
        tone = "balanced"
    # start trace
    trace_id = start_trace(user_id=user_id, username=username, query=question, tone=tone, model=model, embedding_model=EMBEDDING_MODEL, chunking_strategy="structure", top_k=top_k)
    # also need prompt version
    try:
        prompt_version = get_prompt_version(tone)
    except Exception:
        prompt_version = None

    global short_term_memory, episodic_memory, long_term_memory
    short_term_memory = get_short_term_memory()
    episodic_memory = get_episodic_memory()
    long_term_memory = get_long_term_memory(user_id)

    # 0) Cache
    s_cache = start_span("cache_check", inputs={"query": question[:200]})
    t0 = time.perf_counter()
    cached: LLMOutput | None = cache.get(question)
    timings['cache_check'] = time.perf_counter() - t0
    finish_span(s_cache, outputs={"hit": bool(cached)}, metadata={"duration_ms": timings['cache_check']*1000})
    if cached:
        t1 = time.perf_counter()
        s_fmt = start_span("format_answer", inputs={"cached": True})
        formatted = format_answer_for_ui(cached)
        timings['format_answer'] = time.perf_counter() - t1
        finish_span(s_fmt, outputs={"len": len(formatted)})
        _update_memories(question, cached.answer, user_id)
        total_ms = sum(timings.values())*1000
        end_trace(trace_id, answer=cached.answer, status="answered", total_latency_ms=total_ms, prompt_version=prompt_version)
        return {
            "raw_answer": cached.model_dump(),
            "formatted_answer": formatted,
            "cache_hit": True,
            "cache_stats": cache.stats(),
            "memory_stats": _get_memory_stats(),
            "timings": timings,
            "contexts": [c.model_dump() for c in cached.contexts],
            "trace_id": trace_id,
            "tone": tone,
            "prompt_version": prompt_version,
        }

    # 1) Retrieve
    s_ret = start_span("retrieve_context", inputs={"query": question, "top_k": top_k})
    t2 = time.perf_counter()
    contexts = retrieve_context(question, top_k=top_k)
    timings['retrieve_context'] = time.perf_counter() - t2
    # add chunks to span
    try:
        chunks_payload = []
        for c in contexts:
            d = c.model_dump()
            # ensure snippet 500
            d["content"] = (d.get("content") or "")[:2000]
            chunks_payload.append(d)
        # use db_add_chunks with s_ret and trace_id
        from src.observability.tracing_db import add_span_chunks as _add
        _add(s_ret, trace_id, chunks_payload)
    except Exception as e:
        pass
    finish_span(s_ret, outputs={"num_contexts": len(contexts)}, metadata={"vector_hits": len(contexts)})

    # 2) Rerank
    s_rer = start_span("rerank_contexts", inputs={"num_candidates": len(contexts)})
    t3 = time.perf_counter()
    reranked: list[ContextItem] = rerank_contexts(question, contexts, top_n=3)
    timings['rerank_contexts'] = time.perf_counter() - t3
    # store rerank scores as metadata and also add top chunks again
    try:
        _add(s_rer, trace_id, [c.model_dump() for c in reranked])
    except Exception:
        pass
    finish_span(s_rer, outputs={"top_n": len(reranked)}, metadata={"top_score": reranked[0].score if reranked else None})

    # 3) Build LLMInput
    s_build = start_span("build_llm_input")
    t4 = time.perf_counter()
    memory_context = _build_memory_context(question, user_id)
    llm_input = LLMInput(query=question, contexts=[c.model_dump() for c in reranked] + [m.model_dump() for m in memory_context])
    timings['build_llm_input'] = time.perf_counter() - t4
    finish_span(s_build, outputs={"num_contexts": len(llm_input.contexts)})

    # 4) Generate
    s_gen = start_span("generate_answer", inputs={"tone": tone, "model": model, "prompt_version": prompt_version})
    t5 = time.perf_counter()
    llm_output: LLMOutput = generate_answer(llm_input, model=model)
    timings['generate_answer'] = time.perf_counter() - t5
    # capture token usage if generator stored
    token_usage = getattr(generate_answer, "last_token_usage", None)
    finish_span(s_gen, outputs={"answer_len": len(llm_output.answer), "cited": llm_output.cited_sources}, metadata={"token_usage": token_usage, "tone": tone, "prompt_version": prompt_version})

    # 5) Cache put
    s_cache_put = start_span("cache_put")
    t6 = time.perf_counter()
    cache.put(question, llm_output)
    timings['cache_put'] = time.perf_counter() - t6
    finish_span(s_cache_put)

    # 6) Memories
    s_mem = start_span("update_memories")
    t7 = time.perf_counter()
    _update_memories(question, llm_output.answer, user_id)
    timings['update_memories'] = time.perf_counter() - t7
    finish_span(s_mem)

    # 7) Format
    s_fmt2 = start_span("format_answer")
    t8 = time.perf_counter()
    formatted = format_answer_for_ui(llm_output)
    timings['format_answer'] = time.perf_counter() - t8
    finish_span(s_fmt2)

    total_ms = sum(timings.values())*1000
    end_trace(trace_id, answer=llm_output.answer, status="answered", total_latency_ms=total_ms, prompt_version=prompt_version)

    return {
        "raw_answer": llm_output.model_dump(),
        "formatted_answer": formatted,
        "cache_hit": False,
        "cache_stats": cache.stats(),
        "memory_stats": _get_memory_stats(),
        "timings": timings,
        "contexts": [c.model_dump() for c in llm_output.contexts],
        "trace_id": trace_id,
        "tone": tone,
        "prompt_version": prompt_version,
    }

def _build_memory_context(question: str, user_id: str) -> List[ContextItem]:
    memory_context: List[ContextItem] = []
    short_term_context = short_term_memory.get_context_string(max_tokens=500)
    if short_term_context:
        memory_context.append(ContextItem(source_id='short_term_memory', content=f"Recent conversation context: {short_term_context}", score=0.9, dataset='short_term_memory'))
    episodic_context = episodic_memory.get_context_from_summaries(question, max_tokens=300)
    if episodic_context:
        memory_context.append(ContextItem(source_id='episodic_memory', content=f"Relevant conversation summaries: {episodic_context}", score=0.8, dataset='episodic_memory'))
    long_term_facts = long_term_memory.retrieve_facts(question, top_k=3)
    if long_term_facts:
        facts_text = " ".join([f.text for f in long_term_facts])
        memory_context.append(ContextItem(source_id='long_term_memory', content=f"User medical history: {facts_text}", score=0.7, dataset='long_term_memory'))
    return memory_context

def _update_memories(question: str, answer: str, user_id: str):
    short_term_memory.add_message("user", question)
    short_term_memory.add_message("assistant", answer, metadata={"user_id": user_id})
    episodic_memory.add_message("user", question, metadata={"user_id": user_id})
    episodic_memory.add_message("assistant", answer, metadata={"user_id": user_id})
    combined_text = f"User: {question}\nAssistant: {answer}"
    long_term_memory.store_fact(combined_text, source="conversation", metadata={"user_id": user_id, "interaction_type": "Q&A"})

def _get_memory_stats() -> Dict[str, Any]:
    return {'short_term': short_term_memory.get_stats(), 'episodic': episodic_memory.get_stats(), 'long_term': long_term_memory.get_stats()}

if __name__ == "__main__":
    q = "What are the common causes of migraine headaches?"
    r = rag_chat(q)
    print("Timings:", r["timings"])
    print("Trace:", r.get("trace_id"))
    print("Formatted answer:\n", r["formatted_answer"])
