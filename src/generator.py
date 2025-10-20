from openai import OpenAI 
from sentence_transformers import CrossEncoder
from config import GROQ_API_KEY, DEFAULT_MODEL


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

def generate_answer(query, contexts, model=DEFAULT_MODEL):
    context_text = format_context(contexts)
    system_prompt = (
        "You are a helpful medical assistant."
        "Answer the user's question using only the provided context."
        "If the answer is not in the context, say 'I'm not sure based on the provided information.'"
    )
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
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    
    return answer
        