"""HILT Evaluator — faithfulness, context_precision, context_recall, answer_relevance (0-5) via LLM-as-judge"""
from __future__ import annotations
import json
import re
import os
from typing import Dict, Any, List

try:
    from src.config import DEFAULT_MODEL, RAG_EVAL_THRESHOLDS, HILT_ROUTING, PHARMACIST_KEYWORDS
except ImportError:
    from config import DEFAULT_MODEL, RAG_EVAL_THRESHOLDS, HILT_ROUTING, PHARMACIST_KEYWORDS  # type: ignore

EVAL_PROMPT = """
You are an expert evaluator for medical Retrieval-Augmented Generation (RAG) systems.

Evaluate the chatbot's answer based on 4 criteria, 0 (very poor) to 5 (excellent).

1. **Faithfulness** - answer stays true to retrieved context, no hallucination.
2. **Contextual Precision** - retrieved contexts are relevant and specific to question.
3. **Contextual Recall** - answer uses all relevant context needed, complete.
4. **Answer Relevance** - answer directly addresses user's question intent.

Question: {question}
Answer: {answer}
Retrieved Contexts: {context_text}

Return strictly JSON:
{{
  "Faithfulness": 0-5,
  "Faithfulness_comment": "one sentence",
  "Contextual_Precision": 0-5,
  "Contextual_Precision_comment": "one sentence",
  "Contextual_Recall": 0-5,
  "Contextual_Recall_comment": "one sentence",
  "Answer_Relevance": 0-5,
  "Answer_Relevance_comment": "one sentence",
  "Overall_Comment": "brief summary"
}}
"""

def _call_llm_eval(prompt: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": "You are a strict RAG evaluator. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content.strip()
        return json.loads(txt)
    except Exception as e:
        # fallback: heuristically score based on citations etc will be handled outside
        raise e

def evaluate_rag(question: str, answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns: {
      metrics: {faithfulness: float, context_precision: float, context_recall: float, answer_relevance: float},
      comments: {...},
      raw: {...},
      failed_metrics: List[str],
      is_low_confidence: bool,
      confidence: float 0-1 (avg/5),
      routed_role: str
    }
    """
    context_text = "\n\n".join([f"[Source {c.get('source_id','?')}] {c.get('content','')[:800]}" for c in contexts]) if contexts else "NO CONTEXTS"
    prompt = EVAL_PROMPT.format(question=question, answer=answer, context_text=context_text[:6000])

    raw: Dict[str, Any] = {}
    try:
        raw = _call_llm_eval(prompt)
        # normalize keys
        def _pick(*keys):
            for k in keys:
                if k in raw:
                    return raw[k]
            return None
        metrics = {
            "faithfulness": float(_pick("Faithfulness", "faithfulness") or 0),
            "context_precision": float(_pick("Contextual_Precision", "Contextual Precision", "context_precision") or 0),
            "context_recall": float(_pick("Contextual_Recall", "Contextual Recall", "context_recall") or 0),
            "answer_relevance": float(_pick("Answer_Relevance", "Answer Relevance", "answer_relevance") or 0),
        }
        # also support Answer_Relevance vs Answer Relevance naming
        if metrics["answer_relevance"] == 0 and "Answer_Relevance" in raw:
            metrics["answer_relevance"] = float(raw["Answer_Relevance"])
        comments = {
            "faithfulness": str(_pick("Faithfulness_comment", "faithfulness_comment") or ""),
            "context_precision": str(_pick("Contextual_Precision_comment", "context_precision_comment") or ""),
            "context_recall": str(_pick("Contextual_Recall_comment", "context_recall_comment") or ""),
            "answer_relevance": str(_pick("Answer_Relevance_comment", "answer_relevance_comment") or ""),
        }
    except Exception as e:
        # Heuristic fallback when LLM not available
        # faithfulness: if answer has "Tôi chưa tìm thấy" / "I'm not sure" → 0 else 2.5
        # context_recall etc → 2.5 neutral
        lower = answer.lower()
        has_fallback = any(x in lower for x in ["tôi chưa tìm thấy", "i'm not sure", "could you please provide more"])
        has_citation = bool(re.search(r"\[source\s+\d+\]", lower))
        faith = 1.0 if has_fallback else (4.0 if has_citation else 2.5)
        metrics = {
            "faithfulness": faith,
            "context_precision": 3.0 if contexts else 1.0,
            "context_recall": 3.0 if contexts else 1.0,
            "answer_relevance": 3.0,
        }
        comments = {
            "faithfulness": f"heuristic fallback (has_fallback={has_fallback}, has_citation={has_citation}, err={str(e)[:100]})",
            "context_precision": "heuristic fallback",
            "context_recall": "heuristic fallback",
            "answer_relevance": "heuristic fallback",
        }
        raw = {"_fallback": True, "_error": str(e)[:500]}

    # Determine failed metrics vs thresholds
    failed: List[str] = []
    for k, v in metrics.items():
        thresh = RAG_EVAL_THRESHOLDS.get(k, 3.5)
        if v < thresh:
            failed.append(k)

    # Additional hard fails
    lower_ans = answer.lower()
    has_fallback_phrase = any(x in lower_ans for x in ["tôi chưa tìm thấy", "i'm not sure based on the provided information"])
    if has_fallback_phrase and "faithfulness" not in failed:
        failed.append("faithfulness")

    is_low = len(failed) > 0
    # confidence 0-1
    avg = sum(metrics.values()) / len(metrics) if metrics else 0
    confidence = round(avg / 5.0, 3)

    # routing
    routed = route_role(failed, question)

    return {
        "metrics": metrics,
        "comments": comments,
        "raw": raw,
        "failed_metrics": failed,
        "is_low_confidence": is_low,
        "confidence": confidence,
        "routed_role": routed,
        "thresholds": dict(RAG_EVAL_THRESHOLDS),
    }

def route_role(failed_metrics: List[str], question: str) -> str:
    """Chọn role duyệt dựa trên metric thấp nhất / keyword"""
    ql = (question or "").lower()
    # pharmacist priority if drug-related and faithfulness failed
    if any(kw.lower() in ql for kw in PHARMACIST_KEYWORDS):
        # if any failure, pharmacist gets priority for medication queries
        if failed_metrics:
            return "pharmacist"
    if not failed_metrics:
        return HILT_ROUTING.get("default", "doctor")
    # priority order: faithfulness > answer_relevance > context_precision > context_recall
    priority = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    for p in priority:
        if p in failed_metrics:
            return HILT_ROUTING.get(p, HILT_ROUTING.get("default", "doctor"))
    # fallback first failed
    return HILT_ROUTING.get(failed_metrics[0], HILT_ROUTING.get("default", "doctor"))
