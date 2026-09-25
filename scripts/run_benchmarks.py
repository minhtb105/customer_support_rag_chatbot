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
    from shared.models.llm_io import ContextItem

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
    from chat.retriever import load_vectorstores, normalize_docs
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
    from shared.evaluation import (HybridRetriever, RetrievalEvaluator,
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
    from chat.retriever import get_bm25
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
    from chat.generator import rerank_contexts
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
    from shared.evaluation import HybridRetriever

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
    from chat.rag_pipeline import rag_chat

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
    from shared.evaluation import HybridRetriever
    from chat.generator import client as llm_client, format_context, \
        _normalize_model, detect_tone_and_temp
    from shared.prompt_manager import get_system_prompt, get_evaluation_prompt

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
    # B6-B8 service benchmarks (opt-in; omitted = legacy B1-B5 only, D2).
    parser.add_argument("--services", type=str, default=None,
                        help="comma list triage,labs,solvers or all-services")
    parser.add_argument("--mode", type=str, default="ci_no_key",
                        choices=["ci_no_key", "nightly"])
    parser.add_argument("--llm-n", type=int, default=5,
                        help="cap for nightly LLM judge sampling (reserved; "
                             "B6-B8 currently make 0 LLM calls in both modes)")
    parser.add_argument("--update-baseline", action="store_true",
                        help="manual-only: rewrite baseline_benchmark.json "
                             "from this green run (never in CI)")
    args = parser.parse_args()

    if args.services:
        code = run_services_mode(args)
        sys.exit(code)

    from shared.evaluation import load_dataset
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


# ---------------------------------------------------------------------------
# B6-B8 service benchmarks (triage / labs / solvers).
#
# Pins D1-D8: pure functions only (no TestClient, no DB writes, no LLM/RAG
# imports on the ci path); --services omitted => legacy B1-B5 byte-identical;
# baseline READ-ONLY in CI (--update-baseline is manual-only); overall_status
# + exit code derive SOLELY from safety gates; regression is WARNING (exit 0).
# ---------------------------------------------------------------------------

SERVICES = ("triage", "labs", "solvers")
BENCH_DIR = os.path.join(ROOT, "data", "evaluation")
BASELINE_PATH = os.path.join(BENCH_DIR, "baseline_benchmark.json")
BENCH_FILES = {
    "triage": "bench_triage.json",
    "labs": "bench_labs.json",
    "solvers": "bench_solvers.json",
}
# Fixed display names for the tmp benchmark roster (same shape as tests).
BENCH_NAMES = ["BS. Nguyen Van A", "BS. Tran Thi B", "BS. Le Van C",
               "BS. Pham Thi D", "BS. Hoang Van E"]

REGRESSION_THRESHOLD = 0.05


def load_gold(service):
    path = os.path.join(BENCH_DIR, BENCH_FILES[service])
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _recall(found, total):
    return round(found / total, 4) if total else 1.0


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def run_b6(gold=None):
    """B6 triage: check_red_flag + route_tier (pure)."""
    from src.triage.red_flag import check_red_flag
    from src.labs.handoff import route_tier

    gold = gold if gold is not None else load_gold("triage")
    emer_total = emer_hit = 0
    pan_total = pan_hit = 0
    tier_total = tier_hit = 0
    group_hit = group_total = 0
    lat = []
    for case in gold:
        t = time.perf_counter()
        if case.get("kind") == "tier":
            tier = route_tier(case["text"],
                              panic=bool(case.get("panic", False)))["tier"]
            tier_total += 1
            tier_hit += (tier == case["expect_tier"])
        else:
            res = check_red_flag(case["text"])
            ok = (res["emergency"] == case["expect_emergency"])
            emer_total += 1
            emer_hit += ok
            if case.get("expect_emergency"):
                pan_total += 1
                pan_hit += ok
                group_total += 1
                group_hit += (res.get("red_flag_type")
                              == case.get("expect_group"))
        lat.append(time.perf_counter() - t)
    return {
        "safety_gates": {"panic_recall": _recall(pan_hit, pan_total)},
        "quality": {
            "emergency_accuracy": _recall(emer_hit, emer_total),
            "tier_accuracy": _recall(tier_hit, tier_total),
            "group_accuracy": _recall(group_hit, group_total),
        },
        "performance": latency_summary(lat),
    }


def run_b7(gold=None):
    """B7 labs: is_panic + _parse_question + check_red_flag (pure, no DB)."""
    from src.labs.lab_thresholds import is_panic
    from src.labs.router import _parse_question
    from src.triage.red_flag import check_red_flag

    gold = gold if gold is not None else load_gold("labs")
    pan_total = pan_hit = 0
    parse_scores = []
    carve_total = carve_hit = 0
    red_total = red_hit = 0
    lat = []
    for case in gold:
        t = time.perf_counter()
        kind = case.get("kind")
        if kind == "panic":
            try:
                hit = (is_panic(case["loinc"], float(case["value"]))
                       == case["expect_panic"])
            except Exception:
                hit = (False == case["expect_panic"])
            pan_total += 1
            pan_hit += hit
        elif kind == "parse":
            got = set(_parse_question(case["question"]).get("loincs", []))
            want = set(case.get("expect_loincs", []))
            parse_scores.append(1.0 if want <= got else
                                (len(got & want) / len(want) if want else 1.0))
            if set(want) & {"2160-0", "33914-3", "1920-8", "1742-6"}:
                carve_total += 1
                carve_hit += (want <= got)
        else:  # redflag text
            res = check_red_flag(case["text"])
            red_total += 1
            red_hit += (res["emergency"] == case["expect_emergency"])
        lat.append(time.perf_counter() - t)
    return {
        "safety_gates": {
            "panic_recall": _recall(
                sum(1 for c in gold if c.get("kind") == "panic"
                    and c.get("expect_panic")
                    and _safe_panic(c)),
                sum(1 for c in gold if c.get("kind") == "panic"
                    and c.get("expect_panic"))),
            "carveout_pass": _recall(carve_hit, carve_total),
        },
        "quality": {
            "nopanic_accuracy": _recall(
                sum(1 for c in gold if c.get("kind") == "panic"
                    and not c.get("expect_panic") and _safe_nopanic(c)),
                sum(1 for c in gold if c.get("kind") == "panic"
                    and not c.get("expect_panic"))),
            "parse_loinc_recall": round(_mean(parse_scores), 4),
            "redflag_accuracy": _recall(red_hit, red_total),
        },
        "performance": latency_summary(lat),
    }


def _safe_panic(case):
    try:
        from src.labs.lab_thresholds import is_panic
        return bool(is_panic(case["loinc"], float(case["value"])))
    except Exception:
        return False


def _safe_nopanic(case):
    try:
        from src.labs.lab_thresholds import is_panic
        return not bool(is_panic(case["loinc"], float(case["value"])))
    except Exception:
        return True


def _slot_ok(s):
    """DFS exact-case slot check: not T4 (Wed), outside 10:00-11:00."""
    try:
        from datetime import date as _date
        if (s.get("weekday") == "T4"
                or _date.fromisoformat(s.get("date")).weekday() == 2):
            return False
        return not ("10:00" <= (s.get("time") or "") < "11:00")
    except Exception:
        return False


def _tmp_roster_env():
    """Deterministic tmp roster (exact seed hours) for B8; returns restore fn."""
    import tempfile
    from scripts.seed_synthetic_data import DOCTOR_SPECS
    tmp = tempfile.mkdtemp(prefix="bench_roster_")
    doctors = [
        {"doctor_id": did, "name": name, "specialty": spec,
         "working_hours": hours}
        for (did, spec, hours), name in zip(DOCTOR_SPECS, BENCH_NAMES)
    ]
    with open(os.path.join(tmp, "synthetic_roster.json"), "w",
              encoding="utf-8") as f:
        json.dump({"doctors": doctors}, f, ensure_ascii=False)
    with open(os.path.join(tmp, "doctors_patients.json"), "w",
              encoding="utf-8") as f:
        f.write("{}")
    prev = os.environ.get("SYNTHETIC_SIDECAR_DIR")
    os.environ["SYNTHETIC_SIDECAR_DIR"] = tmp

    def _restore():
        if prev is None:
            os.environ.pop("SYNTHETIC_SIDECAR_DIR", None)
        else:
            os.environ["SYNTHETIC_SIDECAR_DIR"] = prev
    return _restore


def run_b8(gold=None, nightly=False):
    """B8 solvers: parse_triage + SchedulerOrchestrator/Greedy/DFS/GA (pure)."""
    from src.triage.red_flag import check_red_flag
    from src.triage.triage_nlu import parse_triage
    from src.scheduling.solvers import (SchedulerOrchestrator, GreedySolver,
                                        DFSSolver, doctor_attributes,
                                        simulate_time_off)

    restore = _tmp_roster_env()
    try:
        gold = gold if gold is not None else load_gold("solvers")
        route_total = route_hit = 0
        parse_total = parse_hit = 0
        sat_scores = []
        relaxed_lists = []
        ga_summary = None
        ga_elapsed = None
        greedy_lat = []
        lat = []
        for case in gold:
            kind = case.get("kind")
            t = time.perf_counter()
            if kind == "route":
                nlu = parse_triage(case["message"])
                solver, _ = SchedulerOrchestrator.route(nlu)
                route_total += 1
                route_hit += (solver == case["expect_solver"])
            elif kind == "nlu":
                nlu = parse_triage(case["message"])
                exp = case.get("expect", {})
                ok = True
                if "has_bhyt" in exp:
                    ok = ok and (nlu.get("has_bhyt") is exp["has_bhyt"])
                if "tier" in exp:
                    ok = ok and (nlu.get("tier") == exp["tier"])
                if "window" in exp:
                    w = (nlu.get("time_constraints") or {}).get(
                        "excluded_window") or {}
                    ok = ok and (w.get("start") == exp["window"][0]
                                 and w.get("end") == exp["window"][1])
                if "requested_doctors" in exp:
                    ok = ok and (nlu.get("requested_doctors")
                                 == exp["requested_doctors"])
                if "constraint" in exp:
                    ok = ok and (exp["constraint"]
                                 in (nlu.get("constraints") or []))
                parse_total += 1
                parse_hit += ok
            elif kind == "dfs_exact":
                nlu = parse_triage(case["message"])
                res = DFSSolver.solve(nlu)
                relaxed_lists.append(res.get("relaxed", []))
                ok = bool(res.get("recommended_doctors")) and all(
                    _slot_ok(s) for d in res["recommended_doctors"]
                    for s in (d.get("slots") or []))
                sat_scores.append(1.0 if ok else 0.0)
            elif kind == "dfs_relax":
                nlu = parse_triage(case["message"])
                res = DFSSolver.solve(nlu)
                relaxed_lists.append(res.get("relaxed", []))
                ok = (bool(res.get("recommended_doctors"))
                      and "specialty" not in (res.get("relaxed") or []))
                if "bhyt" in (res.get("relaxed") or []):
                    ok = ok and ("không đảm bảo BHYT" in
                                 (res.get("routing_reason") or ""))
                sat_scores.append(1.0 if ok else 0.0)
            elif kind == "greedy_filter":
                nlu = parse_triage(case["message"])
                t0 = time.perf_counter()
                res = GreedySolver.solve(nlu)
                greedy_lat.append(time.perf_counter() - t0)
                ok = (bool(res.get("recommended_doctors")) and all(
                    d.get("accepts_bhyt") for d in
                    res["recommended_doctors"]))
                sat_scores.append(1.0 if ok else 0.0)
            elif kind == "ga":
                nlu = parse_triage(case["message"])
                solver, res = SchedulerOrchestrator.route(nlu)
                route_total += 1
                route_hit += (solver == "ga")
                ga_summary = res.get("simulation_summary")
                ok = (ga_summary is not None and all(
                    k in ga_summary for k in ("reassigned", "unplaced",
                                              "avg_shift_days", "fitness")))
                sat_scores.append(1.0 if ok else 0.0)
            elif kind == "emergency":
                res = check_red_flag(case["message"])
                parse_total += 1
                parse_hit += bool(res["emergency"])
            elif kind == "calibration":
                from scripts.seed_synthetic_data import DOCTOR_SPECS
                ids = [did for did, _, _ in DOCTOR_SPECS]
                attrs = [doctor_attributes(d) for d in ids]
                ok = (sum(1 for a in attrs if a["distance_km"] < 5) >= 2
                      and {a["accepts_bhyt"] for a in attrs} == {True, False}
                      and {a["tier"] for a in attrs} == {"central", "district"})
                parse_total += 1
                parse_hit += ok
            lat.append(time.perf_counter() - t)
        # Standalone GA timing (safety <5s; nightly repeats x3 for stability).
        ga_times = []
        for _ in range(3 if nightly else 1):
            t0 = time.perf_counter()
            ga_summary = simulate_time_off("syn_doctor_01", 3, n=200, seed=42)
            ga_times.append(time.perf_counter() - t0)
        ga_ok = (max(ga_times) < 5.0)
        never_relax = all("specialty" not in (r or []) for r in relaxed_lists)
        return {
            "safety_gates": {
                "never_relax_specialty": 1.0 if never_relax else 0.0,
                "ga_timing_ok": 1.0 if ga_ok else 0.0,
            },
            "quality": {
                "correct_solver_rate": _recall(route_hit, route_total),
                "parse_accuracy": _recall(parse_hit, parse_total),
                "constraint_satisfaction_rate": round(
                    _mean(sat_scores), 4),
            },
            "performance": {
                "greedy_solve": latency_summary(greedy_lat),
                "ga_simulate_s": latency_summary(ga_times),
                "per_case": latency_summary(lat),
            },
            "_ga_summary": ga_summary,
        }
    finally:
        restore()


def services_overall_status(services):
    """PASSED/FAILED solely from safety gates (D6)."""
    for svc, data in (services or {}).items():
        for name, val in (data.get("safety_gates") or {}).items():
            if not isinstance(val, (int, float)) or val < 1.0:
                return "FAILED"
    return "PASSED"


def compare_to_baseline(services, baseline):
    """Per-service mean-quality relative drop >5% -> WARNING (exit 0, D5)."""
    warnings = []
    base_svcs = (baseline or {}).get("services", {})
    if not baseline or not base_svcs:
        return ["WARNING: missing baseline (run with --update-baseline "
                "once from a green tree); regression check skipped."]
    for svc, data in (services or {}).items():
        cur_q = data.get("quality") or {}
        base_q = (base_svcs.get(svc) or {}).get("quality") or {}
        keys = [k for k in cur_q if k in base_q and isinstance(
            cur_q[k], (int, float)) and isinstance(base_q[k], (int, float))]
        if not keys:
            warnings.append(f"WARNING: no comparable quality keys for "
                            f"service '{svc}'; check skipped.")
            continue
        base_mean = sum(base_q[k] for k in keys) / len(keys)
        cur_mean = sum(cur_q[k] for k in keys) / len(keys)
        if base_mean > 0 and (base_mean - cur_mean) / base_mean > \
                REGRESSION_THRESHOLD:
            warnings.append(
                f"WARNING: service '{svc}' mean quality dropped "
                f"{base_mean:.4f} -> {cur_mean:.4f} "
                f"({(base_mean - cur_mean) / base_mean:.1%} > 5%).")
    return warnings


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
            timeout=10).decode().strip()
    except Exception:
        return "unknown"


def run_services_mode(args):
    """Opt-in B6-B8 run. Returns process exit code (D6: safety only)."""
    os.environ["OPENAI_API_KEY"] = ""  # "" survives lazy load_dotenv (override=False) + falsy -> _parse_llm early-None; 0 LLM calls
    selected = [s.strip().lower() for s in (args.services or "").split(",")
                if s.strip().lower() in SERVICES]
    if not selected:
        selected = list(SERVICES)
    nightly = (args.mode == "nightly")
    services = {}
    if "triage" in selected:
        log("B6: triage (pure check_red_flag + route_tier)...")
        services["triage"] = run_b6()
        log(f"B6: {services['triage']['quality']} "
            f"{services['triage']['safety_gates']}")
    if "labs" in selected:
        log("B7: labs (pure is_panic + _parse_question)...")
        services["labs"] = run_b7()
        log(f"B7: {services['labs']['quality']} "
            f"{services['labs']['safety_gates']}")
    if "solvers" in selected:
        log("B8: solvers (pure NLU + Greedy/DFS/GA)...")
        res = run_b8(nightly=nightly)
        services["solvers"] = {k: v for k, v in res.items()
                               if not k.startswith("_")}
        log(f"B8: {services['solvers']['quality']} "
            f"{services['solvers']['safety_gates']}")
    baseline = None
    baseline_ref = {"path": None, "mode": None, "git_sha": None}
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH, encoding="utf-8") as f:
                baseline = json.load(f)
            baseline_ref = {"path": os.path.basename(BASELINE_PATH),
                            "mode": (baseline or {}).get("mode"),
                            "git_sha": (baseline or {}).get("git_sha")}
        except Exception:
            baseline = None
    warnings = compare_to_baseline(services, baseline)
    for w in warnings:
        log(w)
    overall = services_overall_status(services)
    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "overall_status": overall,
        "baseline_ref": baseline_ref,
        "services": services,
        "regression_warnings": warnings,
    }
    out_path = os.path.join(
        ROOT, "data", "evaluation",
        f"results_benchmark_services_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"results saved -> {out_path}")
    if args.update_baseline:
        bl = {"mode": args.mode, "git_sha": _git_sha(),
              "timestamp": output["timestamp"],
              "services": {s: {"quality": d.get("quality", {})}
                           for s, d in services.items()}}
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(bl, f, indent=2, ensure_ascii=False)
        log(f"baseline updated -> {BASELINE_PATH}")
    print(json.dumps({s: d.get("quality") for s, d in services.items()},
                     indent=2))
    return 0 if overall == "PASSED" else 1


if __name__ == "__main__":
    main()
