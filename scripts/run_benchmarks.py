"""
Full RAG benchmark runner.

Sections:
  B1  Chunking strategies (vector-only): Recall@5 / Hit@5 / MRR
  B2  Retriever types on the best-practice index: bm25 vs vector vs hybrid
  B3  Cross-encoder rerank impact on hybrid results
  B4  Latency P50/P95: retrieval-only, cache-hit E2E, fresh E2E
  B5  LLM-as-judge scoring (faithfulness/precision/recall/fluency, 0-5)

Usage:
  python scripts/run_benchmarks.py [--skip-judge] [--latency-n 10]

Results are written to data/evaluation/results_benchmark_<ts>.json
Run from the project root with the project venv.
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

warnings.filterwarnings("ignore")

DATASET_PATH = os.path.join(ROOT, "data", "evaluation",
                            "retrieval_evaluation.json")
STRATEGIES = ["structure", "sliding", "semantic", "hybrid_section_semantic"]
K = 5


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


T0 = time.time()


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def docs_to_contexts(docs):
    """LangChain Documents -> ContextItem list (minimal fields)."""
    from models.llm_io import ContextItem

    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        out.append(ContextItem(
            source_id=str(meta.get("source_id", "N/A")),
            content=d.page_content[:4000],
            dataset=meta.get("dataset"),
            score=None,
        ))
    return out


def vector_retrieve(query, strategy, k=10):
    from retriever import load_vectorstores, normalize_docs
    retr = load_vectorstores(strategy).as_retriever(
        search_kwargs={"k": k})
    return normalize_docs(retr.invoke(query))


def make_eval_results():
    return {"Recall@K": [], "Hit@K": [], "MRR": []}


def agg(results):
    return {
        metric: round(sum(v) / len(v), 4) if v else 0.0
        for metric, v in results.items()
    }


def run_b123(dataset):
    """B1 strategies x vector-only, B2 retriever types, B3 rerank impact."""
    from evaluation import (HybridRetriever, RetrievalEvaluator,
                            recall_at_k, hit_at_k, mean_reciprocal_rank)

    results = {}

    # ---- B1: per-strategy vector-only -------------------------------------
    for strat in STRATEGIES:
        log(f"B1: evaluating strategy '{strat}' (vector-only)...")
        rows = make_eval_results()
        for sample in dataset:
            q, gold = sample["question"], sample["gold_sources"]
            ctxs = docs_to_contexts(vector_retrieve(q, strat))
            rows["Recall@K"].append(recall_at_k(ctxs, gold, K))
            rows["Hit@K"].append(hit_at_k(ctxs, gold, K))
            rows["MRR"].append(mean_reciprocal_rank(ctxs, gold, K))
        results[f"vector_only/{strat}"] = agg(rows)
        log(f"B1: {strat} -> {results[f'vector_only/{strat}']}")

    # ---- B2: retriever types (structure index) ----------------------------
    log("B2: BM25-only...")
    from retriever import get_bm25
    bm25 = get_bm25()
    rows = make_eval_results()
    for sample in dataset:
        q, gold = sample["question"], sample["gold_sources"]
        ctxs = docs_to_contexts(bm25.invoke(q))
        rows["Recall@K"].append(recall_at_k(ctxs, gold, K))
        rows["Hit@K"].append(hit_at_k(ctxs, gold, K))
        rows["MRR"].append(mean_reciprocal_rank(ctxs, gold, K))
    results["bm25_only/structure"] = agg(rows)
    log(f"B2: bm25 -> {results['bm25_only/structure']}")

    log("B2: hybrid retrieval...")
    ev = RetrievalEvaluator({"hybrid": HybridRetriever()})
    results["hybrid/structure"] = ev.evaluate(dataset, k=K)["hybrid"]
    log(f"B2: hybrid -> {results['hybrid/structure']}")

    # ---- B3: rerank impact -------------------------------------------------
    log("B3: hybrid + cross-encoder rerank (top_n=5)...")
    from generator import rerank_contexts
    rows = make_eval_results()
    latencies = []
    for sample in dataset:
        q, gold = sample["question"], sample["gold_sources"]
        t = time.perf_counter()
        cand = HybridRetriever().retrieve(q)
        reranked = rerank_contexts(q, cand, top_n=5)[:K]
        latencies.append(time.perf_counter() - t)
        rows["Recall@K"].append(recall_at_k(reranked, gold, K))
        rows["Hit@K"].append(hit_at_k(reranked, gold, K))
        rows["MRR"].append(mean_reciprocal_rank(reranked, gold, K))
    res = agg(rows)
    res["avg_latency_s"] = round(statistics.mean(latencies), 3)
    results["hybrid_reranked/structure"] = res
    log(f"B3: hybrid+rerank -> {res}")

    return results


# ---------------------------------------------------------------------------
# B4 latency
# ---------------------------------------------------------------------------

def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1,
                     int(round((p / 100.0) * len(values) + 0.5)) - 1))
    return round(values[idx], 3)


def latency_summary(samples):
    return {
        "n": len(samples),
        "mean_s": round(statistics.mean(samples), 3) if samples else None,
        "p50_s": percentile(samples, 50),
        "p95_s": percentile(samples, 95),
        "min_s": round(min(samples), 3) if samples else None,
        "max_s": round(max(samples), 3) if samples else None,
    }


def run_b4(dataset, n):
    """Latency: retrieval-only, cache-hit E2E, fresh E2E."""
    from evaluation import HybridRetriever

    questions = [s["question"] for s in dataset[:n]]

    # retrieval-only (warm caches)
    HybridRetriever().retrieve(questions[0])  # warm-up
    lat = []
    for q in questions:
        t = time.perf_counter()
        HybridRetriever().retrieve(q)
        lat.append(time.perf_counter() - t)
    retrieval_stats = latency_summary(lat)
    log(f"B4 retrieval-only: {retrieval_stats}")

    # E2E via rag_chat (imports memory stack + CAG cache)
    from rag_pipeline import rag_chat

    # fresh E2E: distinct questions -> expected cache misses
    fresh_lat = []
    for i, q in enumerate(questions):
        variant = f"fresh benchmark question {i}: {q}"
        t = time.perf_counter()
        r = rag_chat(variant)
        dt = time.perf_counter() - t
        if not r.get("cache_hit"):
            fresh_lat.append(dt)
        log(f"B4 fresh E2E [{i + 1}/{len(questions)}] "
            f"cache_hit={r.get('cache_hit')} {dt:.2f}s")
    fresh_stats = latency_summary(fresh_lat)
    log(f"B4 fresh E2E: {fresh_stats}")

    # cache-hit E2E: repeat exact same questions already cached above
    hit_lat = []
    for i, q in enumerate(questions):
        variant = f"fresh benchmark question {i}: {q}"
        t = time.perf_counter()
        r = rag_chat(variant)
        if r.get("cache_hit"):
            hit_lat.append(time.perf_counter() - t)
    hit_stats = latency_summary(hit_lat)
    log(f"B4 cache-hit E2E ({len(hit_lat)}/{len(questions)} hits): {hit_stats}")

    return {
        "retrieval_only": retrieval_stats,
        "e2e_fresh": fresh_stats,
        "e2e_cache_hit": hit_stats,
        "note": "E2E includes Groq/OpenAI LLM call; measured on CPU client.",
    }


# ---------------------------------------------------------------------------
# B5 judge
# ---------------------------------------------------------------------------

def run_b5(dataset, max_questions=None):
    """Generate answers then LLM-as-judge using EVALUATION_PROMPT rubric."""
    import re as _re
    from evaluation import HybridRetriever
    from generator import client as llm_client, format_context, \
        _normalize_model, detect_tone_and_temp
    from prompt_manager import get_system_prompt, get_evaluation_prompt

    samples = dataset[:max_questions] if max_questions else dataset
    scores = []
    for i, s in enumerate(samples):
        q = s["question"]
        ctxs = HybridRetriever().retrieve(q)
        context_text = format_context(ctxs)
        system_prompt = get_system_prompt("balanced")
        user_prompt = (
            f"Context: \n{context_text}\n\n"
            f"Question: {q}\n\n"
            "Answer clearly and concisely"
        )
        resp = llm_client.chat.completions.create(
            model=_normalize_model(None),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content.strip()

        judge_prompt = get_evaluation_prompt().format(
            question=q, answer=answer, context_text=context_text)
        jresp = llm_client.chat.completions.create(
            model=_normalize_model(None),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = jresp.choices[0].message.content.strip()
        cleaned = _re.sub(r"^```(json)?|```$", "", raw, flags=_re.M).strip()
        try:
            verdict = json.loads(cleaned)
        except json.JSONDecodeError:
            m = _re.search(r"\{.*\}", cleaned, _re.S)
            verdict = json.loads(m.group(0)) if m else {}
        scores.append(verdict)
        keys = ["Faithfulness", "Contextual_Precision",
                "Contextual_Recall", "Fluency"]
        got = [verdict.get(k) for k in keys]
        log(f"B5 [{i + 1}/{len(samples)}] judge scores: {got}")

    def avg(key):
        vals = [v.get(key) for v in scores
                if isinstance(v.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {
        "n_scored": sum(1 for v in scores if v),
        "avg_Faithfulness": avg("Faithfulness"),
        "avg_Contextual_Precision": avg("Contextual_Precision"),
        "avg_Contextual_Recall": avg("Contextual_Recall"),
        "avg_Fluency": avg("Fluency"),
    }
    log(f"B5 judge summary: {summary}")
    return {"summary": summary, "per_question": scores}


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-n", type=int, default=None,
                        help="limit judge to first N questions")
    parser.add_argument("--latency-n", type=int, default=10)
    args = parser.parse_args()

    from evaluation import load_dataset
    dataset = load_dataset(DATASET_PATH)
    log(f"dataset: {len(dataset)} questions from {DATASET_PATH}")

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "k": K,
            "dataset": os.path.basename(DATASET_PATH),
            "n_questions": len(dataset),
            "strategies": STRATEGIES,
        }
    }

    output["b123_retrieval"] = run_b123(dataset)
    output["b4_latency"] = run_b4(dataset, n=args.latency_n)
    if not args.skip_judge:
        output["b5_judge"] = run_b5(dataset, max_questions=args.judge_n)

    out_path = os.path.join(
        ROOT, "data", "evaluation",
        f"results_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"results saved -> {out_path}")
    print(json.dumps(output["b123_retrieval"], indent=2))


if __name__ == "__main__":
    main()
