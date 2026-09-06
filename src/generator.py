import re
from typing import List
from openai import OpenAI
from sentence_transformers import CrossEncoder

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
    from observability.local_tracing import short_hash
except ImportError:
    from src.observability.local_tracing import short_hash  # type: ignore

try:
    from models.llm_io import LLMInput, LLMOutput, ContextItem
except ImportError:
    from src.models.llm_io import LLMInput, LLMOutput, ContextItem


def _build_llm_client():
    if LLM_PROVIDER == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _normalize_model(model: str) -> str:
    m = (model or DEFAULT_MODEL).strip()
    if LLM_PROVIDER != "groq" and m.startswith("openai/"):
        return m.split("/", 1)[1]
    return m


client = _build_llm_client()
reranker = CrossEncoder(RERANKER_MODEL)

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
        # version citation: try to get version_label/publication_date from context
        version = getattr(context, 'version_label', None) or (context.model_extra.get('version_label') if hasattr(context, 'model_extra') and context.model_extra else None) or getattr(context, 'version', None)
        pub_date = getattr(context, 'publication_date', None) or (context.model_extra.get('publication_date') if hasattr(context, 'model_extra') and context.model_extra else None)
        # also check dataset style extra fields
        if not version and hasattr(context, 'model_extra'):
            version = (context.model_extra or {}).get('version_label') or (context.model_extra or {}).get('version')
        if not pub_date and hasattr(context, 'model_extra'):
            pub_date = (context.model_extra or {}).get('publication_date')
        # fallback to metadata dict if stored there
        header = f"[Source {src_id}"
        if version:
            header += f" | {version}"
            if pub_date:
                header += f" | {pub_date}"
        elif pub_date:
            header += f" | {pub_date}"
        header += f" | Score={score_str}]"
        context_text += f"{header}\n{context.content}\n\n"
    return context_text.strip()

def rerank_contexts(query: str, contexts: List[ContextItem], top_n=3):
    if not contexts:
        return []
    pairs = [(query, ctx.content) for ctx in contexts]
    scores = reranker.predict(pairs)
    for i, ctx in enumerate(contexts):
        ctx.score = float(scores[i])
    ranked = sorted(contexts, key=lambda x: x.score, reverse=True)
    return ranked[:top_n]

DIABETES_KEYWORDS = [
    "diabetes", "đái tháo đường", "đai thao duong", "tiểu đường", "tieu duong",
    "hba1c", "insulin", "glucose", "đường huyết", "duong huyet",
    "hypoglycemia", "hyperglycemia", "who", "ada", "pen", "hearts-d",
    "chẩn đoán đái tháo đường", "biến chứng đái tháo đường"
]
HYPERTENSION_KEYWORDS = [
    "hypertension", "tăng huyết áp", "tang huyet ap", "huyết áp", "huyet ap",
    "blood pressure", "120/80", "140/90", "180/120", "aha/acc", "hearts hypertension"
]
RESPIRATORY_KEYWORDS = [
    "asthma", "hen suyễn", "copd", "phổi tắc nghẽn", "peak flow", "gold", "gina",
    "inhaler", "thuốc hít", "cat score", "khó thở"
]
MENTAL_KEYWORDS = [
    "depression", "trầm cảm", "tram cam", "anxiety", "lo âu", "phq-9", "phq9", "gad-7", "gad7",
    "mhgap", "mental health", "sức khỏe tâm thần", "tự tử", "suicide", "tự hại", "self-harm"
]

def detect_tone_and_temp(query: str):
    query_lower = query.lower()
    if any(k in query_lower for k in MENTAL_KEYWORDS):
        return "mental", 0.1, 512
    if any(k in query_lower for k in HYPERTENSION_KEYWORDS):
        return "hypertension", 0.1, 512
    if any(k in query_lower for k in RESPIRATORY_KEYWORDS):
        return "respiratory", 0.1, 512
    if any(k in query_lower for k in DIABETES_KEYWORDS):
        return "diabetes", 0.1, 512
    strict_keywords = ["diagnosis", "treatment", "symptom", "disease", "side effect","risk", "medicine", "disorder", "infection", "pain", "safe for"]
    friendly_keywords = ["feel", "stress", "diet", "exercise", "well-being", "advice", "sleep", "healthy"]
    if any(k in query_lower for k in strict_keywords):
        return "strict", 0.1, 256
    if any(k in query_lower for k in friendly_keywords):
        return "friendly", 0.4, 256
    if query.strip().startswith("why "):
        return "balanced", 0.3, 512
    return "balanced", 0.2, 512

def generate_answer(input_data: LLMInput, model=DEFAULT_MODEL) -> LLMOutput:
    query = input_data.query
    context_text = format_context(input_data.contexts)
    tone, temperature, max_tokens = detect_tone_and_temp(query)
    system_prompt = get_system_prompt(tone)
    user_prompt = (f"Context: \n{context_text}\n\n" f"Question: {query}\n\n" "Answer clearly and concisely")
    messages = [{"role": "system", "content": system_prompt}]
    for past in chat_history[-5:]:
        messages.append({"role": "user", "content": past["user"]})
        messages.append({"role": "assistant", "content": past["assistant"]})
    messages.append({"role": "user", "content": user_prompt})
    resolved_model = _normalize_model(model)
    response = client.chat.completions.create(model=resolved_model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    answer = response.choices[0].message.content.strip()
    chat_history.append({"user": query, "assistant": answer})
    usage = getattr(response, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    } if usage else {}
    prompt_version = short_hash(system_prompt)
    cited_sources = sorted({int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer)})
    # attach metadata to local tracing via context if available
    try:
        from src.observability.local_tracing import get_current_trace_id
        # we don't have span here, but caller rag_pipeline will handle; store in a global for rag_pipeline to pick?
        generate_answer.last_token_usage = token_usage  # type: ignore
        generate_answer.last_prompt_version = prompt_version  # type: ignore
        generate_answer.last_tone = tone  # type: ignore
    except Exception:
        pass
    return LLMOutput(answer=answer, cited_sources=cited_sources, contexts=input_data.contexts)

def format_answer_for_ui(answer_data: LLMOutput) -> str:
    formatted_answer = (answer_data.answer.replace("\n\n", "<br><br>").replace("\n", "<br>"))
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
    citations_text = (" — Sources: " + ", ".join(citation_entries) if citation_entries else "")
    return f"{formatted_answer}<br><br><i>{citations_text}</i>"
