import time
from typing import Dict, Any, List

try:
    from shared.config import TOP_K, DEFAULT_MODEL, CAG_MAX_SIZE, CAG_TTL_SECONDS, CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD, EMBEDDING_MODEL
except ImportError:
    from src.shared.config import TOP_K, DEFAULT_MODEL, CAG_MAX_SIZE, CAG_TTL_SECONDS, CAG_SEMANTIC_MODEL, CAG_SEMANTIC_THRESHOLD, EMBEDDING_MODEL

try:
    from chat.retriever import retrieve_context
    from chat.generator import generate_answer, generate_answer_stream, rerank_contexts, format_answer_for_ui, detect_tone_and_temp
    from shared.models.llm_io import ContextItem, LLMInput, LLMOutput
    from shared.cache import CAGHybridCache
    from shared.memory import get_short_term_memory, get_episodic_memory, get_long_term_memory
except ImportError:
    from src.chat.retriever import retrieve_context
    from src.chat.generator import generate_answer, generate_answer_stream, rerank_contexts, format_answer_for_ui, detect_tone_and_temp
    from src.shared.models.llm_io import ContextItem, LLMInput, LLMOutput
    from src.shared.cache import CAGHybridCache
    from src.shared.memory import get_short_term_memory, get_episodic_memory, get_long_term_memory

try:
    from shared.observability.local_tracing import start_trace, end_trace, start_span, finish_span, add_span_chunks
    from shared.observability.tracing_db import add_span_chunks as db_add_chunks  # alias
    from shared.prompt_manager import get_prompt_version
except ImportError:
    from src.shared.observability.local_tracing import start_trace, end_trace, start_span, finish_span, add_span_chunks
    from src.shared.observability.tracing_db import add_span_chunks as db_add_chunks
    from src.shared.prompt_manager import get_prompt_version

cache = CAGHybridCache(max_size=CAG_MAX_SIZE, ttl_seconds=CAG_TTL_SECONDS, semantic_model_name=CAG_SEMANTIC_MODEL, semantic_threshold=CAG_SEMANTIC_THRESHOLD)
short_term_memory = get_short_term_memory()
episodic_memory = get_episodic_memory()
long_term_memory = get_long_term_memory()

try:
    from src.monitors.gho_snapshot import looks_like_stats_question, query_gho_stats
except ImportError:
    try:
        from monitors.gho_snapshot import looks_like_stats_question, query_gho_stats  # type: ignore
    except ImportError:
        looks_like_stats_question = query_gho_stats = None  # type: ignore


def maybe_inject_gho_stats(question: str, contexts: List[ContextItem]) -> List[ContextItem]:
    """Buoc 1b Dual-Storage (dung chung non-stream + stream): cau hoi so lieu
    -> chen context tu SQLite readonly, khong RAG so. Fail-open (khong bao gio crash)."""
    if not (looks_like_stats_question and query_gho_stats):
        return contexts
    try:
        if looks_like_stats_question(question):
            s_gho = start_span("gho_stats_tool", inputs={"query": question[:200]})
            gho_text = query_gho_stats(question)
            finish_span(s_gho, outputs={"hit": bool(gho_text)})
            if gho_text:
                return [ContextItem(source_id="gho_stats", content=gho_text,
                                    section_path=["gho", "VNM"], dataset="gho",
                                    file_name="gho_stats.db")] + contexts
    except Exception:
        pass
    return contexts


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
        from src.shared.observability.tracing_db import add_span_chunks as _add
        _add(s_ret, trace_id, chunks_payload)
    except Exception as e:
        pass
    finish_span(s_ret, outputs={"num_contexts": len(contexts)}, metadata={"vector_hits": len(contexts)})

    # 1b) GHO stats tool (Luong B Dual-Storage)
    contexts = maybe_inject_gho_stats(question, contexts)

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
    try:
        add_span_chunks(s_build, trace_id, [m.model_dump() for m in memory_context])
    except Exception:
        pass
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


def rag_chat_stream(question: str, top_k: int = TOP_K, user_id: str = "default_user", username: str | None = None, model: str = DEFAULT_MODEL, **kwargs):
    """
    SSE streaming variant of rag_chat — yields SSE event dicts:
      {"event": "metadata", "data": {...}}
      {"event": "token", "data": {"delta": str}}
      {"event": "done", "data": {...final payload...}}
      {"event": "error", "data": {"error": str}}
    Caller (FastAPI) formats to wire: f"event: {e}\\ndata: {json}\\n\\n"
    """
    import re
    import json as _json
    timings: Dict[str, float] = {}
    try:
        tone, _, _ = detect_tone_and_temp(question)
    except Exception:
        tone = "balanced"
    trace_id = start_trace(user_id=user_id, username=username, query=question, tone=tone, model=model, embedding_model=EMBEDDING_MODEL, chunking_strategy="structure", top_k=top_k)
    try:
        prompt_version = get_prompt_version(tone)
    except Exception:
        prompt_version = None

    global short_term_memory, episodic_memory, long_term_memory
    short_term_memory = get_short_term_memory()
    episodic_memory = get_episodic_memory()
    long_term_memory = get_long_term_memory(user_id)

    # 0) Cache check
    s_cache = start_span("cache_check", inputs={"query": question[:200]})
    t0 = time.perf_counter()
    cached: LLMOutput | None = cache.get(question)
    timings['cache_check'] = time.perf_counter() - t0
    finish_span(s_cache, outputs={"hit": bool(cached)}, metadata={"duration_ms": timings['cache_check']*1000})

    if cached:
        # metadata early
        contexts_payload = [c.model_dump() for c in cached.contexts]
        yield {"event": "metadata", "data": {"trace_id": trace_id, "prompt_version": prompt_version, "tone": tone, "contexts": contexts_payload, "cache_hit": True, "timings": timings}}
        # stream cached answer in chunks for UX (20 chars per SSE token)
        ans = cached.answer or ""
        chunk_size = 20
        for i in range(0, len(ans), chunk_size):
            yield {"event": "token", "data": {"delta": ans[i:i+chunk_size]}}
        # memories / tracing still needed
        t_mem = time.perf_counter()
        s_mem = start_span("update_memories")
        _update_memories(question, ans, user_id)
        timings['update_memories'] = time.perf_counter() - t_mem
        finish_span(s_mem)
        total_ms = sum(timings.values())*1000
        end_trace(trace_id, answer=ans, status="answered", total_latency_ms=total_ms, prompt_version=prompt_version)
        formatted = format_answer_for_ui(cached)
        # HILT evaluation for cached too (best effort)
        evaluation = None
        is_low = False
        status_str = "answered"
        review_id = None
        try:
            from src.reviews.evaluator import evaluate_rag as _eval
        except ImportError:
            try:
                from reviews.evaluator import evaluate_rag as _eval  # type: ignore
            except Exception:
                _eval = None  # type: ignore
        if _eval is not None:
            try:
                evaluation = _eval(question, ans, contexts_payload)
                is_low = bool(evaluation.get("is_low_confidence"))
                if is_low:
                    status_str = "pending_review"
            except Exception:
                pass
        yield {"event": "done", "data": {
            "answer": ans,
            "formatted_answer": formatted,
            "cited_sources": getattr(cached, "cited_sources", []),
            "contexts": contexts_payload,
            "trace_id": trace_id,
            "prompt_version": prompt_version,
            "tone": tone,
            "cache_hit": True,
            "timings": timings,
            "evaluation": evaluation,
            "is_low_confidence": is_low,
            "status": status_str,
            "review_id": review_id,
        }}
        return

    # 1) Retrieve
    s_ret = start_span("retrieve_context", inputs={"query": question, "top_k": top_k})
    t2 = time.perf_counter()
    contexts = retrieve_context(question, top_k=top_k)
    timings['retrieve_context'] = time.perf_counter() - t2
    try:
        chunks_payload = []
        for c in contexts:
            d = c.model_dump()
            d["content"] = (d.get("content") or "")[:2000]
            chunks_payload.append(d)
        from src.shared.observability.tracing_db import add_span_chunks as _add
        _add(s_ret, trace_id, chunks_payload)
    except Exception:
        pass
    finish_span(s_ret, outputs={"num_contexts": len(contexts)}, metadata={"vector_hits": len(contexts)})

    # 1b) GHO stats tool (Luong B Dual-Storage) — mirror non-stream
    contexts = maybe_inject_gho_stats(question, contexts)

    # 2) Rerank
    s_rer = start_span("rerank_contexts", inputs={"num_candidates": len(contexts)})
    t3 = time.perf_counter()
    reranked: list[ContextItem] = rerank_contexts(question, contexts, top_n=3)
    timings['rerank_contexts'] = time.perf_counter() - t3
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
    try:
        add_span_chunks(s_build, trace_id, [m.model_dump() for m in memory_context])
    except Exception:
        pass
    finish_span(s_build, outputs={"num_contexts": len(llm_input.contexts)})

    # Emit metadata before streaming so frontend can show citations early
    contexts_for_client = [c.model_dump() for c in reranked]
    yield {"event": "metadata", "data": {"trace_id": trace_id, "prompt_version": prompt_version, "tone": tone, "contexts": contexts_for_client, "cache_hit": False, "timings": timings}}

    # 4) Generate streaming
    s_gen = start_span("generate_answer", inputs={"tone": tone, "model": model, "prompt_version": prompt_version})
    t5 = time.perf_counter()
    full_answer = ""
    try:
        for delta in generate_answer_stream(llm_input, model=model):
            full_answer += delta
            yield {"event": "token", "data": {"delta": delta}}
    except Exception as e:
        finish_span(s_gen, outputs={"error": str(e)}, metadata={"tone": tone})
        yield {"event": "error", "data": {"error": f"generate_stream error: {e}"}}
        end_trace(trace_id, answer=full_answer, status="failed", total_latency_ms=sum(timings.values())*1000, prompt_version=prompt_version)
        return
    timings['generate_answer'] = time.perf_counter() - t5
    # retrieve cited sources from generator metadata if available
    cited_sources = getattr(generate_answer_stream, "last_cited_sources", None)
    if cited_sources is None:
        try:
            cited_sources = sorted({int(m) for m in re.findall(r"\[Source\s+(\d+)\]", full_answer)})
        except Exception:
            cited_sources = []
    token_usage = getattr(generate_answer_stream, "last_token_usage", None)
    finish_span(s_gen, outputs={"answer_len": len(full_answer), "cited": cited_sources}, metadata={"token_usage": token_usage, "tone": tone, "prompt_version": prompt_version})

    # Build LLMOutput for downstream steps
    llm_output = LLMOutput(answer=full_answer, cited_sources=cited_sources, contexts=llm_input.contexts)

    # 5) Cache put
    s_cache_put = start_span("cache_put")
    t6 = time.perf_counter()
    try:
        cache.put(question, llm_output)
    except Exception:
        pass
    timings['cache_put'] = time.perf_counter() - t6
    finish_span(s_cache_put)

    # 6) Memories
    s_mem = start_span("update_memories")
    t7 = time.perf_counter()
    _update_memories(question, full_answer, user_id)
    timings['update_memories'] = time.perf_counter() - t7
    finish_span(s_mem)

    # 7) Format (for done payload)
    s_fmt2 = start_span("format_answer")
    t8 = time.perf_counter()
    formatted = format_answer_for_ui(llm_output)
    timings['format_answer'] = time.perf_counter() - t8
    finish_span(s_fmt2)

    # HILT evaluation (same as rag_query)
    evaluation = None
    review_id = None
    status_str = "answered"
    is_low = False
    try:
        from src.reviews.evaluator import evaluate_rag as _eval2
    except ImportError:
        try:
            from reviews.evaluator import evaluate_rag as _eval2  # type: ignore
        except Exception:
            _eval2 = None  # type: ignore
    if _eval2 is not None:
        try:
            evaluation = _eval2(question, full_answer, [c.model_dump() for c in llm_output.contexts])
            is_low = bool(evaluation.get("is_low_confidence"))
            if is_low:
                # create review request
                from src.reviews.service import create_review_request as _create_rr
                try:
                    from src.chat.generator import detect_tone_and_temp as _detect
                    tone2, _, _ = _detect(question)
                    disease = tone2 if tone2 in ("diabetes","hypertension","respiratory","mental") else None
                except Exception:
                    disease = None
                try:
                    rev = _create_rr(query=question, draft_answer=full_answer, contexts=[c.model_dump() for c in llm_output.contexts], evaluation=evaluation, requester_id=user_id, disease=disease, langsmith_run_id=trace_id)
                    review_id = rev["id"]
                    status_str = "pending_review"
                    try:
                        from src.shared.observability.tracing_db import update_trace as _ut
                        _ut(trace_id, review_id=review_id, status="pending_review", is_low_confidence=1, routed_role=evaluation.get("routed_role"))
                    except Exception:
                        pass
                except Exception:
                    status_str = "pending_review"
            else:
                try:
                    from src.auth.db import add_query_history as _aqh
                    from src.shared.observability.tracing_db import update_trace as _ut2
                    _aqh(user_id=user_id, query=question, answer=full_answer, status="answered", confidence=evaluation.get("confidence"))
                    _ut2(trace_id, status="answered", is_low_confidence=0)
                except Exception:
                    pass
        except Exception as e:
            print(f"[HILT stream] evaluator error: {e}")

    total_ms = sum(timings.values())*1000
    # end trace with final status
    end_trace(trace_id, answer=full_answer, status=status_str, total_latency_ms=total_ms, prompt_version=prompt_version)

    yield {"event": "done", "data": {
        "answer": full_answer,
        "formatted_answer": formatted,
        "cited_sources": cited_sources,
        "contexts": [c.model_dump() for c in llm_output.contexts],
        "trace_id": trace_id,
        "prompt_version": prompt_version,
        "tone": tone,
        "cache_hit": False,
        "timings": timings,
        "evaluation": evaluation,
        "is_low_confidence": is_low,
        "status": status_str,
        "review_id": review_id,
    }}

if __name__ == "__main__":
    q = "What are the common causes of migraine headaches?"
    r = rag_chat(q)
    print("Timings:", r["timings"])
    print("Trace:", r.get("trace_id"))
    print("Formatted answer:\n", r["formatted_answer"])
