import json
from typing import List, Dict, Any
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
    def retrieve(self, query: str) -> List[ContextItem]:
        return retrieve_context(query)

# =========================
# 🔹 Helper Functions
# =========================

def extract_doc_id(ctx: ContextItem) -> str:
    return ctx.source_id if hasattr(ctx, 'source_id') else "N/A"


# =========================
# 🔹 Metrics
# =========================

def recall_at_k(retrieved: List[ContextItem], relevant: List[str], k: int) -> float:
    retrieved_ids = [extract_doc_id(r) for r in retrieved[:k]]
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    return len(set(retrieved_ids) & relevant_set) / len(relevant_set)


def hit_at_k(retrieved: List[ContextItem], relevant: List[str], k: int) -> int:
    retrieved_ids = [extract_doc_id(r) for r in retrieved[:k]]
    relevant_set = set(relevant)

    return int(len(set(retrieved_ids) & relevant_set) > 0)


def mean_reciprocal_rank(retrieved: List[ContextItem], relevant: List[str], k: int) -> float:
    retrieved_ids = [extract_doc_id(r) for r in retrieved[:k]]
    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
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

        for sample in dataset:
            query = sample["question"]
            relevant = sample.get("gold_sources", [])

            for name, retriever in self.retrievers.items():
                contexts = retriever.retrieve(query)

                results[name]["Recall@K"].append(
                    recall_at_k(contexts, relevant, k)
                )
                results[name]["Hit@K"].append(
                    hit_at_k(contexts, relevant, k)
                )
                results[name]["MRR"].append(
                    mean_reciprocal_rank(contexts, relevant, k)
                )

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
# 🔹 Dataset Loader
# =========================

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 🔹 Main
# =========================

if __name__ == "__main__":
    dataset = load_dataset("F:/customer_support_rag_chatbot/data/evaluation/benchmark_questions.json")

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
            