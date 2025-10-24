import json
from openai import OpenAI 
from generator import format_context
from config import GROQ_API_KEY, DEFAULT_MODEL
from prompt_templates import EVALUATION_PROMPT


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def evaluate_rag_triad(question, answer, contexts, model=DEFAULT_MODEL):
    """
    Evaluate RAG chatbot answers using the RAG Triad framework + Fluency.

    Metrics:
    - Faithfulness: Is the answer faithful to the given context (no hallucination)?
    - Contextual Precision: Does the retrieved context precisely support the answer?
    - Contextual Recall: Does the context cover enough relevant information?
    - Fluency: Is the answer written in clear, natural English?
    """
    context_text = format_context(contexts)
    prompt = EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        context_text=context_text
    )
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict and knowledgeable evaluator for RAG systems. "
                    "Evaluate based solely on the context and the answer provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    # Try to extract JSON even if extra text appears
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        evaluation = json.loads(json_str)
    except Exception as e:
        evaluation = {
            "Faithfulness": 0,
            "Faithfulness_comment": "Parsing failed.",
            "Contextual_Precision": 0,
            "Contextual_Precision_comment": "Parsing failed.",
            "Contextual_Recall": 0,
            "Contextual_Recall_comment": "Parsing failed.",
            "Fluency": 0,
            "Fluency_comment": "Parsing failed.",
            "Overall_Comment": f"Invalid JSON or parsing error: {e}",
        }

    return evaluation
