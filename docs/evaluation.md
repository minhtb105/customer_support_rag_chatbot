# Đánh Giá Hệ Thống — Retrieval & LLM-as-Judge

> Nguồn: `src/evaluation.py:1`, `src/reviews/evaluator.py:1`, `src/observability/tracing_db.py:118`, `data/evaluation/*.json`, `scripts/run_benchmarks.py:1`.

## 1. Tập dữ liệu

| File | Số câu | Schema | Mục đích |
|---|---|---|---|
| `data/evaluation/diabetes_retrieval_evaluation.json` | 15 | `{question_id, question, relevant_documents[], topic, expected_answer_span[]}` | Benchmark chính diabetes (DQ1-DQ15) — WHO Classification 2019, HEARTS-D 2020, PEN 2020, ADA SoC 2024 |
| `data/evaluation/diabetes_benchmark_questions.json` | 5 | `{question, gold_sources, relevant_documents}` | Multi-hop diabetes (so sánh 2 guideline) |
| `data/evaluation/retrieval_evaluation.json` | 20 | `{question, relevant_documents, gold_sources, supporting_docs}` (legacy) | Generic WHO (physical activity, sodium, sleep) — đã gắn `pytest.mark.legacy` skip mặc định |
| `data/evaluation/benchmark_questions.json` | 10 | legacy generic | Alias cũ |
| `data/evaluation/results_benchmark_20260826_2033.json` | n_questions=K=5 | output `run_benchmarks.py` | Lưu vết B1-B5 |

**Chuẩn hóa multi-schema:** `src/evaluation.py:62 _extract_gold_docs` thử lần lượt `relevant_documents → gold_sources → supporting_docs → gold_documents`, trả list string.

## 2. Metrics Retrieval

`src/evaluation.py:90 extract_doc_id` + `43 normalize_doc_name`:

```python
normalize(s) = re.sub("[^a-z0-9]+","_", s.lower().replace(".pdf","").strip("_"))
extract(ctx) = normalize(ctx.source_id)
```

**Công thức (k=5 mặc định):**

- `Recall@K = |retrieved_ids[:K] ∩ relevant| / |relevant|` (`recall_at_k:105`)
- `Hit@K = 1 if |intersection|>0 else 0` (`hit_at_k:114`)
- `MRR = 1 / rank(first_hit) else 0` (`mean_reciprocal_rank:120`)

`RetrievalEvaluator.evaluate(dataset, k=5)` lặp từng sample, gọi `retriever.retrieve(query)` → tính 3 metrics, đồng thời `matched_gold` warning nếu `gold` chưa từng match bất kỳ doc nào (phát hiện lỗi nhãn).

## 3. Retrieval Interface

```python
class HybridRetriever(BaseRetriever):
  def retrieve(self, query: str) -> List[ContextItem]:
    return retrieve_context(query, top_k=self.top_k, strategy=self.strategy)
```

`src/retriever.py: load_vectorstores(strategy)` → `Chroma.as_retriever(k=top_k)` + `BM25Retriever` + hybrid `alpha=0.6` (`config.py:63 HYBRID_ALPHA`), `MERGE_PEERS=True`.

## 4. LLM-as-Judge

### 4.1 RAG HILT Evaluator (online, per query)

`src/reviews/evaluator.py:13 EVAL_PROMPT` 4 tiêu chí 0-5:

1. **Faithfulness** — bám context, không ảo giác
2. **Contextual Precision** — context retrieve có liên quan
3. **Contextual Recall** — dùng đủ context cần thiết
4. **Answer Relevance** — trả lời đúng ý hỏi

**Flow** `evaluate_rag(question, answer, contexts)→`:
- `context_text = "[Source X] content[:800]"` join `[:6000]`
- `_call_llm_eval(prompt)` → `OpenAI(OPENAI_MODEL, temperature 0.1, max_tokens 800, response_format json_object)` → parse `Faithfulness/Contextual_Precision/...`
- Fallback heuristic nếu không có API key: `has_fallback_phrase → faith 1.0`, `has_citation → 4.0 else 2.5`, `contexts empty → 1.0 else 3.0`, ngưỡng `3.5`.

**Kết quả** `src/reviews/evaluator.py:146`:
```python
{
  metrics: {faithfulness, context_precision, context_recall, answer_relevance},
  comments: {...},
  failed_metrics: [k for k,v<threshold],
  is_low_confidence: len(failed)>0,
  confidence: avg/5.0,
  routed_role: route_role(failed, question)  # pharmacist nếu có PHARMACIST_KEYWORDS
}
```

**Routing** `route_role:157`:
- `PHARMACIST_KEYWORDS=[thuốc,medication,drug,dosage,liều,tác dụng phụ,side effect,pharmacist,dược]` → ưu tiên `pharmacist` nếu query chứa.
- Priority `faithfulness > answer_relevance > context_precision > context_recall` → map `HILT_ROUTING` (`config.py:166` `faithfulness:doctor, precision:specialist...`).

`src/api/main.py:210` dùng kết quả để `create_review_request` + `update_trace(is_low_confidence, routed_role)` → `status pending_review`.

### 4.2 RAGAS Tracing (offline, per trace)

`src/observability/tracing_db.py:118 ragas_evaluations` lưu `faithfulness, context_precision, context_recall, answer_relevance, fluency` + 5 comment + `failed_metrics JSON`, `confidence`, `raw_json`, `evaluator_model`.

- `upsert_ragas(trace_id, metrics, raw)` — insert hoặc update.
- Trigger: `POST /v1/admin/traces/{trace_id}/ragas` (admin) → gọi `evaluate_rag` on-demand nếu `RAGAS_ON_DEMAND=true`.

### 4.3 Benchmark B5 (batch)

`scripts/run_benchmarks.py:228 run_b5`:
- Lấy `HybridRetriever().retrieve(q)` → `format_context` → `get_system_prompt("balanced")` → LLM generate `answer` (`temperature 0.2, max 512`).
- Judge: `get_evaluation_prompt().format(question, answer, context_text)` → LLM judge (`temperature 0, max 500`) → regex `\{.*\}` → parse `Faithfulness/Contextual_Precision/Contextual_Recall/Fluency`.
- Output `summary: avg_Faithfulness, avg_Contextual_Precision, avg_Contextual_Recall, avg_Fluency`.

## 5. Prompt Versioning & Tracing

- `src/observability/tracing_db.py:101 prompts` bảng `tone, version=sha256(text)[:8] UNIQUE, status draft|pending_approval|active|archived, is_active`.
- `src/prompt_manager.py:31 PROMPT_REGISTRY` 12 tones: `strict, friendly, balanced, diabetes, hypertension, respiratory, mental, soap, who_rag, evaluation, guideline_diff, safety_summary`.
- `get_system_prompt(tone)` cache `PROMPT_ACTIVE_CACHE_TTL_SECONDS=300`, fallback local nếu DB rỗng.
- `src/observability/local_tracing.py` ContextVar `trace_id`, 7 spans per `rag_chat` (`cache_check → retrieve_context → rerank_contexts → build_llm_input → generate_answer → cache_put → format_answer`), `total_latency_ms`, `prompt_version=short_hash(system_prompt)[:8]`.

**Retention:** `TRACING_RETENTION_DAYS=30`, `list_traces` filter `created_at >= now-30d`, `delete_expired_traces`.

## 6. Feedback

`feedback` table `src/observability/tracing_db.py:133` `trace_id, user_id, key, score, value, comment` — UI 👍/👎 gắn `trace_id`.

## 7. Cách chạy

```bash
# Retrieval only (không cần LLM)
python -m src.evaluation  # mặc định retrieval_evaluation.json
# hoặc
from src.evaluation import load_dataset, RetrievalEvaluator, HybridRetriever
ds = load_dataset("data/evaluation/diabetes_retrieval_evaluation.json")
ev = RetrievalEvaluator({"hybrid": HybridRetriever(strategy="structure")})
print(ev.evaluate(ds, k=5))

# Full pipeline + HILT (cần FastAPI + auth)
curl -X POST http://localhost:8000/v1/query -H "Authorization: Bearer $TOKEN" -d '{"query":"Ngưỡng chẩn đoán đái tháo đường?","top_k":5}'
# → evaluation.failed_metrics + is_low_confidence + routed_role

# Batch benchmark B1-B5 (cần OPENAI_API_KEY)
python scripts/run_benchmarks.py --latency-n 10 --judge-n 5
python scripts/run_benchmarks.py --skip-judge  # chỉ B1-B4
```

## 8. Liên hệ với Benchmark & Data Contract

- `data_contract.md#Metrics` định nghĩa fingerprint/hash; `evaluation.md` dùng `normalize_doc_name` để khớp `source_id` vs `gold` — warning `unmatched` nếu nhãn sai.
- `benchmark.md` sẽ điền số `Recall@5/Hit@5/MRR` từ `run_b123` và `avg_Faithfulness` từ `run_b5`.
