from openai import OpenAI 
from sentence_transformers import CrossEncoder
from config import GROQ_API_KEY, DEFAULT_MODEL
from prompt_templates import (
    STRICT_SYSTEM_PROMPT, FRIENDLY_SYSTEM_PROMPT, BALANCED_SYSTEM_PROMPT)


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# In-memory conversation history
chat_history = []

def format_context(contexts):
    context_text = ""
    for context in contexts:
        src_id = context.get("source_id", "N/A")
        score = context.get("score", "N/A")
        context_text += f"[Source {src_id} | Score={score:.4f}]\n{context['content']}\n\n"
    
    return context_text.strip()

def rerank_contexts(query, contexts, top_n=3):
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

def generate_answer(query, contexts, model=DEFAULT_MODEL):
    """
    Generate an answer that includes inline citations like [Source 1].
    """
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
    chat_history.append({
        "user": query,
        "assistant": answer
    })

    # Extract which sources were cited
    cited_sources = set()
    import re
    for match in re.findall(r"\[Source\s+(\d+)\]", answer):
        cited_sources.add(int(match))

    return {
        "answer": answer,
        "cited_sources": sorted(list(cited_sources)),
        "contexts": contexts,
    }
        
def format_answer_for_ui(answer_data: dict) -> str:
    """
    Format chatbot answer for frontend display.
    - Convert newlines to HTML <br> tags
    - Append citation list with source_id and optional dataset name
    """

    answer_text = answer_data.get("answer", "").strip()
    cited_sources = answer_data.get("cited_sources", [])
    contexts = answer_data.get("contexts", [])

    # Format line breaks for web display
    formatted_answer = (
        answer_text
        .replace("\n\n", "<br><br>")
        .replace("\n", "<br>")
    )

    # Build citation text
    citation_entries = []
    for src_id in cited_sources:
        dataset = None
        for ctx in contexts:
            if ctx.get("source_id") == src_id:
                dataset = ctx.get("dataset")
                break
        
        if dataset:
            citation_entries.append(f"[{src_id}] {dataset}")
        else:
            citation_entries.append(f"[{src_id}]")

    # Join citations
    if citation_entries:
        citations_text = " — Sources: " + ", ".join(citation_entries)
    else:
        citations_text = ""

    # Final HTML-safe answer
    formatted_output = f"{formatted_answer}<br><br><i>{citations_text}</i>"

    return formatted_output
