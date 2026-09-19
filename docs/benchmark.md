# Benchmark — Kết Quả Chạy Thực Tế

> Chạy `scripts/run_benchmarks.py` với `OPENAI_API_KEY` thật (2026-09-06). Dataset mặc định `retrieval_evaluation.json` (legacy 20 câu) cho B1-B4, và verify riêng `diabetes_retrieval_evaluation.json` (15 câu DQ1-DQ15) để có số có ý nghĩa.

## 0. Môi trường

- `EMBEDDING_PROVIDER=openai` → `text-embedding-3-small` 1536d, fallback `MiniLM 384d`
- `RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `TOP_K=10`, `HYBRID_ALPHA=0.6`, `K eval=5`
- `CAG: TTL 24h, SEMANTIC_THRESHOLD 0.82, MAX_SIZE 1024`
- CPU client, `HF_HUB_OFFLINE=1`, embeddings qua `api.openai.com` (Batches 1/1, ~4-5s mỗi batch)

Kết quả lưu: `data/evaluation/results_benchmark_20260906_1920.json` (B1-B4, skip-judge) và `results_benchmark_20260826_2033.json` trước đó.

## 1. B1 — Chunking Strategies (vector-only)

Chạy trên **legacy dataset 20 câu** (`retrieval_evaluation.json`) — nhãn `relevant_documents` đa số là WHO physical activity / sodium / water ... **không có trong corpus hiện tại** → tất cả 0, kèm warning `16 gold never matched`. Đây là lỗi nhãn legacy, không phải lỗi retriever.

| Strategy | Recall@5 | Hit@5 | MRR | Ghi chú |
|---|---|---|---|---|
| `structure` | 0.0 | 0.0 | 0.0 | Legacy labels mismatch |
| `sliding` | 0.0 | 0.0 | 0.0 | |
| `semantic` | 0.0 | 0.0 | 0.0 | |
| `hybrid_section_semantic` | 0.0 | 0.0 | 0.0 | |

**Trên diabetes dataset 15 câu (đúng corpus 4 PDFs):**

| Strategy | Recall@5 | Hit@5 | MRR |
|---|---|---|---|
| `structure` | **0.767** | **0.933** | **0.817** |
| `sliding` | 0.0 | 0.0 | 0.0 |
| `semantic` | 0.0 | 0.0 | 0.0 |
| `hybrid_section_semantic` | 0.0 | 0.0 | 0.0 |

- Chỉ `structure` có dữ liệu (33.6 MB Chroma). Các DB `sliding/semantic/hybrid` tồn tại (37-39 MB) nhưng `load_vectorstores` với `OPENAI` provider chỉ resolve `pdf_db_openai/structure` (1.7 MB), không có `sliding`/`semantic` cho OpenAI → trả về rỗng. Đây là gap đã biết, ghi trong `data_contract#Embedding Contract`.
- Kết luận: dùng `structure` cho production; cần re-index `sliding/semantic` bằng OpenAI nếu muốn so sánh công bằng.

## 2. B2 — Retriever Types (structure index)

**Legacy 20 câu:**

| Retriever | Recall@5 | Hit@5 | MRR |
|---|---|---|---|
| `bm25_only/structure` | 0.0 | 0.0 | 0.0 |
| `hybrid/structure` | 0.0 | 0.0 | 0.0 |

**Diabetes 15 câu:**

| Retriever | Recall@5 | Hit@5 | MRR |
|---|---|---|---|
| `bm25_only` | **0.9** | **1.0** | **0.769** |
| `hybrid (BM25 0.4 + vector 0.6)` | **0.767** | **0.933** | **0.817** |

BM25 thắng nhẹ về Recall/Hit nhưng hybrid thắng MRR (đưa đúng doc lên rank cao hơn). Phù hợp với `HYBRID_ALPHA=0.6`.

## 3. B3 — Cross-encoder Rerank

| Pipeline | Recall@5 | Hit@5 | MRR | avg_latency_s |
|---|---|---|---|---|
| `hybrid/structure` (không rerank) | 0.767 (diabetes) / 0.0 (legacy) | — | — | 0.48 (retrieval-only) |
| `hybrid + rerank top_n=5 → top5` | **0.0** (legacy, do labels) | 0.0 | 0.0 | **4.821 s** |

Rerank `CrossEncoder ms-marco-MiniLM-L-6-v2` tốn ~4.8s mỗi query (CPU), nhưng trên diabetes dataset chưa đo lại do script mặc định dùng legacy. Dự kiến cải thiện MRR +5-10% khi chạy đúng dataset (cần verify).

## 4. B4 — Latency (n=5, legacy questions variant)

| Stage | n | mean_s | p50_s | p95_s | min_s | max_s |
|---|---|---|---|---|---|---|
| `retrieval_only` (HybridRetriever warm) | 5 | **0.485** | **0.495** | **0.545** | 0.418 | 0.545 |
| `e2e_fresh` (rag_chat cache miss, gồm LLM) | 5 | **9.792** | **9.079** | **12.185** | 8.244 | 12.185 |
| `e2e_cache_hit` (lặp lại câu vừa cache) | 5 | **1.782** | **1.219** | **4.122** | 1.115 | 4.122 |

- `retrieval_only` <0.6s, đủ nhanh cho pre-fetch.
- `e2e_fresh` ~9-12s do `OpenAI gpt-4o-mini` + 2 lần embeddings (query + CAG).
- `e2e_cache_hit` ~1.2-1.7s (chỉ `cache_check + format_answer`), đạt mục tiêu `<600ms` chưa? Hiện 1.2s do `CrossEncoder` load + `CAGHybridCache semantic FAISS` overhead — cần tối ưu (đã ghi `architecture_diagram#Gaps`).

## 5. B5 — LLM-as-Judge

Chạy với `--skip-judge` nên chưa có số mới. Kết quả cũ trong `results_benchmark_20260826_2033.json`:
- Thường `avg_Faithfulness ~4.2`, `Contextual_Precision ~3.8`, `Contextual_Recall ~4.0`, `Fluency ~4.5` trên 5 câu diabetes (cần re-run với `--judge-n 5` để fill).

Cách chạy đầy đủ:
```bash
python scripts/run_benchmarks.py --latency-n 10 --judge-n 5
# hoặc chỉ judge
python scripts/run_benchmarks.py --skip-judge false --judge-n 15 --latency-n 0
```

## 6. Reproduce

```bash
# 1. B1-B3 retrieval (không cần LLM, chỉ cần Chroma)
python scripts/run_benchmarks.py --skip-judge --latency-n 0
# warnings 16 gold never matched là expected cho legacy

# 2. Diabetes-focused (đúng corpus)
python -c "from src.evaluation import load_dataset, RetrievalEvaluator, HybridRetriever; ds=load_dataset('data/evaluation/diabetes_retrieval_evaluation.json'); print(RetrievalEvaluator({'hybrid':HybridRetriever()}).evaluate(ds,k=5))"
# → {'hybrid': {'Recall@K':0.766, 'Hit@K':0.933, 'MRR':0.816}}

# 3. Full với latency + judge (cần OPENAI_API_KEY, 15-20p)
python scripts/run_benchmarks.py --latency-n 10
ls data/evaluation/results_benchmark_*.json
```

## 7. Khuyến nghị

1. Đổi `DATASET_PATH` mặc định trong `run_benchmarks.py:36` từ `retrieval_evaluation.json` sang `diabetes_retrieval_evaluation.json` để tránh 0 toàn tập.
2. Re-index `sliding/semantic/hybrid` bằng `EMBEDDING_PROVIDER=openai` để B1 có số so sánh công bằng.
3. Tối ưu `CAGHybridCache` và `CrossEncoder` lazy-load để `cache_hit` <600ms.
4. Thêm `B6` monitors benchmark (FDA cache hit rate, BYT scrape success).
