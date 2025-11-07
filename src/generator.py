import re
from typing import List
from openai import OpenAI 
from sentence_transformers import CrossEncoder
from config import GROQ_API_KEY, DEFAULT_MODEL, RERANKER_MODEL
from prompt_templates import (
    STRICT_SYSTEM_PROMPT, FRIENDLY_SYSTEM_PROMPT, BALANCED_SYSTEM_PROMPT)
from models.llm_io import LLMInput, LLMOutput, ContextItem


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
reranker = CrossEncoder(RERANKER_MODEL)

# In-memory conversation history
chat_history = []

def format_context(contexts):
    context_text = ""
    for context in contexts:
        src_id = context.get("source_id", "N/A")
        score = context.get("score", "N/A")
        context_text += f"[Source {src_id} | Score={score:.4f}]\n{context['content']}\n\n"
    
    return context_text.strip()

def rerank_contexts(query: str, contexts: List[ContextItem], top_n=3):
    if not contexts:
        return []
    
    pairs = [(query, ctx['content']) for ctx in contexts]
    scores = reranker.predict(pairs)
    
    for i, ctx in enumerate(contexts):
        ctx["score"] = scores[i]
        
    ranked = sorted(contexts, key=lambda x: x["score"], reverse=True)
    
    return ranked[:top_n]

def detect_tone_and_temp(query: str):
    """
    Heuristics: determine tone + temperature based on the content of the query.
    Return system_prompt, temperature, max_tokens
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
        return STRICT_SYSTEM_PROMPT, 0.1, 256 # concise factual
    
    # Friendly tone
    if any(k in query_lower for k in friendly_keywords):
        return FRIENDLY_SYSTEM_PROMPT, 0.4, 256 # conversational tone

    # Balanced tone
    if query.strip().startswith("why "):
        return BALANCED_SYSTEM_PROMPT, 0.3, 512  # reasoning-heavy answers

    # Default: balanced
    return BALANCED_SYSTEM_PROMPT, 0.2, 512

def generate_answer(input_data: LLMInput, model=DEFAULT_MODEL) -> LLMOutput:
    """
    Generate an answer that includes inline citations like [Source 1].
    """
    contexts = [c.model_dump() for c in input_data.contexts]
    query = input_data.query
    
    context_text = format_context(contexts)
    system_prompt, temperature, max_tokens = detect_tone_and_temp(query)
    
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
     
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    answer = response.choices[0].message.content.strip()
    # Store in conversation memory
    chat_history.append({"user": query, "assistant": answer})

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
