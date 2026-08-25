import re
from typing import List
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langsmith.run_helpers import traceable
from langsmith.wrappers import wrap_openai

try:
    from config import (
        GROQ_API_KEY, GROQ_BASE_URL,
        OPENAI_API_KEY, OPENAI_BASE_URL,
        LLM_PROVIDER, DEFAULT_MODEL, RERANKER_MODEL,
    )
except ImportError:  # pragma: no cover - allows running as "src." package too
    from src.config import (
        GROQ_API_KEY, GROQ_BASE_URL,
        OPENAI_API_KEY, OPENAI_BASE_URL,
        LLM_PROVIDER, DEFAULT_MODEL, RERANKER_MODEL,
    )

try:
    from prompt_manager import get_prompt_version, get_system_prompt
except ImportError:
    from src.prompt_manager import get_prompt_version, get_system_prompt

try:
    from observability.tracing import (
        add_trace_metadata,
        add_trace_outputs,
        short_hash,
    )
except ImportError:
    from src.observability.tracing import (
        add_trace_metadata,
        add_trace_outputs,
        short_hash,
    )

from models.llm_io import LLMInput, LLMOutput, ContextItem


def _build_llm_client():
    """Create the OpenAI-compatible client for the configured provider."""
    if LLM_PROVIDER == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _normalize_model(model: str) -> str:
    """
    Map model identifiers to the active provider.
    - "openai/gpt-4o-mini" on the openai provider -> "gpt-4o-mini"
    - groq keeps prefixed names as-is (e.g. "openai/gpt-oss-20b")
    """
    m = (model or DEFAULT_MODEL).strip()
    if LLM_PROVIDER != "groq" and m.startswith("openai/"):
        return m.split("/", 1)[1]
    return m


client = wrap_openai(_build_llm_client())
reranker = CrossEncoder(RERANKER_MODEL)

# In-memory conversation history
chat_history = []

def format_context(contexts):
    context_text = ""
    for context in contexts:
        src_id = context.source_id if hasattr(context, 'source_id') else "N/A"
        score = context.score if hasattr(context, 'score') else "N/A"
        try:
            score_str = f"{float(score):.4f}"
        except (ValueError, TypeError):
            score_str = str(score)
        context_text += f"[Source {src_id} | Score={score_str}]\n{context.content}\n\n"
    
    return context_text.strip()

@traceable(name="rerank_contexts")
def rerank_contexts(query: str, contexts: List[ContextItem], top_n=3):
    if not contexts:
        return []
    
    pairs = [(query, ctx.content) for ctx in contexts]
    scores = reranker.predict(pairs)
    
    for i, ctx in enumerate(contexts):
        ctx.score = float(scores[i])
        
    ranked = sorted(contexts, key=lambda x: x.score, reverse=True)
    
    add_trace_metadata(
        reranker_model=RERANKER_MODEL,
        num_candidates=len(contexts),
        top_n=top_n,
        top_score=ranked[0].score if ranked else None,
    )
    
    return ranked[:top_n]

def detect_tone_and_temp(query: str):
    """
    Heuristics: determine tone + temperature based on the content of the query.
    Return tone_key, temperature, max_tokens
    """
    query_lower = query.lower()
    
    strict_keywords = [
        "diagnosis", "treatment", "symptom", "disease", "side effect",
        "risk", "medicine", "disorder", "infection", "pain", "safe for"
    ]
    friendly_keywords = [
        "feel", "stress", "diet", "exercise", "well-being", "advice", "sleep", "healthy"
    ]
    
    # Strict tone
    if any(k in query_lower for k in strict_keywords):
        return "strict", 0.1, 256 # concise factual
    
    # Friendly tone
    if any(k in query_lower for k in friendly_keywords):
        return "friendly", 0.4, 256 # conversational tone

    # Balanced tone
    if query.strip().startswith("why "):
        return "balanced", 0.3, 512  # reasoning-heavy answers

    # Default: balanced
    return "balanced", 0.2, 512

@traceable(name="generate_answer")
def generate_answer(input_data: LLMInput, model=DEFAULT_MODEL) -> LLMOutput:
    """
    Generate an answer that includes inline citations like [Source 1].
    System prompt comes from LangSmith Prompt Hub (fallback: local constants).
    """
    query = input_data.query
    context_text = format_context(input_data.contexts)
    tone, temperature, max_tokens = detect_tone_and_temp(query)
    system_prompt = get_system_prompt(tone)

    user_prompt = (
        f"Context: \n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer clearly and concisely"
    )

    # Combine memory + current question
    messages = [{"role": "system", "content": system_prompt}]
    for past in chat_history[-5:]:  # keep last 5 exchanges
        messages.append({"role": "user", "content": past["user"]})
        messages.append({"role": "assistant", "content": past["assistant"]})
    messages.append({"role": "user", "content": user_prompt})

    resolved_model = _normalize_model(model)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    answer = response.choices[0].message.content.strip()
    # Store in conversation memory
    chat_history.append({"user": query, "assistant": answer})

    usage = getattr(response, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    } if usage else {}
    prompt_version = short_hash(system_prompt)

    add_trace_metadata(
        tone=tone,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_model=resolved_model,
        prompt_version=prompt_version,
        token_usage=token_usage,
    )
    add_trace_outputs(
        prompt_version=prompt_version,
        token_usage=token_usage,
    )

    # Extract which sources were cited
    cited_sources = sorted(
        {int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer)}
    )

    # Return structured output (auto-validates)
    return LLMOutput(
        answer=answer,
        cited_sources=cited_sources,
        contexts=input_data.contexts
    )
        
def format_answer_for_ui(answer_data: LLMOutput) -> str:
    """
    Format chatbot answer for frontend display.
    Converts newlines to <br> and appends citation list.
    """
    formatted_answer = (
        answer_data.answer
        .replace("\n\n", "<br><br>")
        .replace("\n", "<br>")
    )

    citation_entries = []
    for src_id in answer_data.cited_sources:
        dataset = None
        for ctx in answer_data.contexts:
            if ctx.source_id == str(src_id):
                dataset = ctx.dataset
                break

        if dataset:
            citation_entries.append(f"[{src_id}] {dataset}")
        else:
            citation_entries.append(f"[{src_id}]")

    citations_text = (
        " — Sources: " + ", ".join(citation_entries)
        if citation_entries else ""
    )

    return f"{formatted_answer}<br><br><i>{citations_text}</i>"
