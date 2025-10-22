import re
from openai import OpenAI 
from sentence_transformers import CrossEncoder
from config import GROQ_API_KEY, DEFAULT_MODEL
from prompt_templates import STRICT_PROMPT, FRIENDLY_PROMPT, BALANCED_PROMPT


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_contexts(query, contexts, top_n=3):
    if not contexts:
        return []
    
    pairs = [(query, ctx['content']) for ctx in contexts]
    scores = reranker.predict(pairs)
    
    for i, ctx in enumerate(contexts):
        ctx["score"] = scores[i]
        
    ranked = sorted(contexts, key=lambda x: x["score"], reverse=True)
    
    return ranked[:top_n]

def format_context(contexts):
    context_text = ""
    for i, context in enumerate(contexts):
        context_text += f"[Source {i + 1} | Score={context.get('score', 'N/A'):.4f}]: {context['content']}\n\n"
    
    return context_text.strip()

def detect_tone_and_temp(query: str):
    """
    Heuristics: determine tone + temperature based on the content of the query.
    """
    query_lower = query.lower()
    
    # Tone: strict if the query is a serious medical question
    if any(keyword in query_lower for keyword in ["diagnosis", "treatment", "symptom", "disease", "side effect", "risk"]):
        return STRICT_PROMPT, 0.1
    
    # Tone: friendly if the query is a light, advisory question
    if any(keyword in query_lower for keyword in ["feel", "stress", "diet", "exercise", "well-being", "advice"]):
        return FRIENDLY_PROMPT, 0.5  # friendly tone, slightly higher temperature

    # Default: balanced
    return BALANCED_PROMPT, 0.2

def generate_answer(query, contexts, model=DEFAULT_MODEL):
    context_text = format_context(contexts)
    
    system_prompt, temperature = detect_tone_and_temp(query)
    
    user_prompt = (
        f"Context: \n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer clearly and concisely"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    answer = response.choices[0].message.content.strip()
    
    return answer
        