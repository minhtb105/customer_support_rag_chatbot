import json
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from retriever import retrieve_context
from models.llm_io import ContextItem

try:
    from observability.tracing import log_evaluation_summary
except ImportError:
    from src.observability.tracing import log_evaluation_summary


# =========================
# 🔹 Retriever Interface
# =========================

class BaseRetriever:
    def retrieve(self, query: str) -> List[ContextItem]:
        raise NotImplementedError


class HybridRetriever(BaseRetriever):
    def __init__(self, strategy: str = "structure", top_k: int = 10):
        self.strategy = strategy
        self.top_k = top_k

    def retrieve(self, query: str) -> List[ContextItem]:
        return retrieve_context(query, top_k=self.top_k, strategy=self.strategy)


# =========================
# 🔹 Document-name normalization
# =========================

def normalize_doc_name(name: str) -> str:
    """
    Canonical form for matching gold document labels against retrieved ids.

    "WHO GUIDELINES ON PHYSICAL ACTIVITY ....pdf" and
    "who_guidelines_on_physical_activity_..." both collapse to the same key.
    """
    n = (name or "").strip().lower()
    if n.endswith(".pdf"):
        n = n[:-4]
    n = n.replace(".pdf", "")  # safety for mid-string artifacts
    n = re.sub(r"[^a-z0-9]+", "_", n)
    return n.strip("_")


# =========================
# 🔹 Dataset loading (multi-schema)
# =========================

def _extract_gold_docs(sample: Dict[str, Any]) -> List[str]:
    """Pull gold document labels out of any known dataset schema."""
    for key in ("relevant_documents", "gold_sources", "supporting_docs",
                "gold_documents"):
        value = sample.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value]
    return []


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a benchmark dataset and normalize it to {question, gold_sources}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = []
    for sample in raw:
        normalized.append({
            "question": sample.get("question", ""),
            "gold_sources": _extract_gold_docs(sample),
            "raw": sample,
        })
    return normalized

# =========================
# 🔹 Helper Functions
# =========================

def extract_doc_id(ctx: ContextItem) -> str:
    raw = ctx.source_id if hasattr(ctx, 'source_id') else "N/A"
    return normalize_doc_name(str(raw))


def _match_set(retrieved: List[ContextItem], relevant: List[str], k: int):
    retrieved_ids = {extract_doc_id(r) for r in retrieved[:k]}
    relevant_set = {normalize_doc_name(r) for r in relevant}
    return retrieved_ids, relevant_set


# =========================
# 🔹 Metrics
# =========================

def recall_at_k(retrieved: List[ContextItem], relevant: List[str], k: int) -> float:
    retrieved_ids, relevant_set = _match_set(retrieved, relevant, k)

    if not relevant_set:
        return 0.0

    return len(retrieved_ids & relevant_set) / len(relevant_set)


def hit_at_k(retrieved: List[ContextItem], relevant: List[str], k: int) -> int:
    retrieved_ids, relevant_set = _match_set(retrieved, relevant, k)

    return int(len(retrieved_ids & relevant_set) > 0)


def mean_reciprocal_rank(retrieved: List[ContextItem], relevant: List[str], k: int) -> float:
    relevant_set = {normalize_doc_name(r) for r in relevant}

    for rank, ctx in enumerate(retrieved[:k], start=1):
        if extract_doc_id(ctx) in relevant_set:
            return 1.0 / rank

    return 0.0


# =========================
# 🔹 Retrieval Evaluator
# =========================

class RetrievalEvaluator:
    def __init__(self, retrievers: Dict[str, BaseRetriever]):
        self.retrievers = retrievers

    def evaluate(self, dataset: List[Dict[str, Any]], k: int = 5) -> Dict[str, Dict[str, float]]:
        results = {
            name: {
                "Recall@K": [],
                "Hit@K": [],
                "MRR": []
            }
            for name in self.retrievers
        }

        # Track which gold labels ever match a retrieved doc, to surface
        # silent labeling mismatches (the classic all-zero-metrics bug).
        matched_gold: Dict[str, bool] = {}

        for sample in dataset:
            query = sample["question"]
            relevant = sample.get("gold_sources", [])

            for gold in relevant:
                matched_gold.setdefault(normalize_doc_name(gold), False)

            for name, retriever in self.retrievers.items():
                contexts = retriever.retrieve(query)

                for ctx in contexts:
                    key = extract_doc_id(ctx)
                    if key in matched_gold:
                        matched_gold[key] = True

                results[name]["Recall@K"].append(
                    recall_at_k(contexts, relevant, k)
                )
                results[name]["Hit@K"].append(
                    hit_at_k(contexts, relevant, k)
                )
                results[name]["MRR"].append(
                    mean_reciprocal_rank(contexts, relevant, k)
                )

        unmatched = [g for g, ok in matched_gold.items() if not ok]
        if unmatched:
            print(f"[eval] WARNING: {len(unmatched)} gold label(s) never matched "
                  f"any retrieved document: {unmatched}")

        return self._aggregate(results)

    def _aggregate(self, results):
        return {
            method: {
                metric: (sum(vals) / len(vals) if vals else 0.0)
                for metric, vals in metrics.items()
            }
            for method, metrics in results.items()
        }


# =========================
# 🔹 Main
# =========================

if __name__ == "__main__":
    import os
    default_ds = os.path.join(os.path.dirname(__file__), "..", "data",
                              "evaluation", "retrieval_evaluation.json")
    dataset = load_dataset(default_ds)

    retrievers = {
        "hybrid": HybridRetriever(),
    }

    evaluator = RetrievalEvaluator(retrievers)
    results = evaluator.evaluate(dataset, k=5)

    print("\n=== Retrieval Evaluation Results ===")
    for method, metrics in results.items():
        print(f"\n[{method}]")
        for m, v in metrics.items():
            print(f"{m}: {v:.4f}")

    # Push the aggregate snapshot to LangSmith so trends show up in dashboards
    logged = log_evaluation_summary("retrieval_evaluation", results, k=5)
    print(f"\nLangSmith logging: {'OK' if logged else 'skipped (tracing unavailable)'}")
            