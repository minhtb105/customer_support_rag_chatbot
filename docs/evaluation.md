# Đánh Giá Hệ Thống — Retrieval & LLM-as-Judge

> Code: `src/shared/evaluation.py`, `src/reviews/evaluator.py`, `src/shared/observability/tracing_db.py:118`. Thuật ngữ RAG/CAG xem Glossary ở `docs/architecture_diagram.md`.

> **Mục tiêu đọc-xong-làm-được:** giải thích Recall/Hit/MRR bằng ví dụ đời thường, kể được 4 tiêu chí judge, chạy pytest đúng file.

## 1. Tập dữ liệu

| File | Số câu | Dùng khi nào |
|---|---|---|
| `data/evaluation/diabetes_retrieval_evaluation.json` | 15 | Benchmark chính ĐTĐ (DQ1-DQ15: WHO 2019, HEARTS-D, PEN, ADA 2024) |
| `data/evaluation/diabetes_benchmark_questions.json` | 5 | Câu multi-hop so sánh 2 guideline |
| `data/evaluation/retrieval_evaluation.json` | 20 | Generic WHO cũ, đã gắn `legacy` skip mặc định |
| `results_benchmark_*.json` | K=5 | Output lưu vết chạy batch |

Chuẩn hóa multi-schema: thử lần lượt `relevant_documents → gold_sources → supporting_docs`, trả list string.

## 2. Metrics Retrieval

Ví dụ đời thường: có 10 quả táo chín cần hái (tài liệu đúng), bạn với tay hái 5 quả (top K=5).

- **Recall@K:** hái trúng mấy/10 quả chín. Công thức: số trúng / tổng số đúng.
- **Hit@K:** có hái trúng ít nhất 1 quả chín không (1/0). Đo "đỡ trắng tay".
- **MRR:** quả chín đầu tiên nằm ở vị trí mấy. Vị trí 1 được 1 điểm, vị trí 2 được 1/2, không trúng được 0.

Code (`src/shared/evaluation.py:90`): so tên file chuẩn hóa (bỏ `.pdf`, lower, thay ký tự lạ bằng `_`) rồi tính 3 số trên. Có cảnh báo `matched_gold` khi đáp án vàng chưa từng khớp doc nào (nghi nhãn sai).

## 3. Retrieval Interface

`HybridRetriever.retrieve(query)` gọi `retrieve_context` (BM25 từ khóa + vector ngữ nghĩa, alpha 0.6) trên Chroma theo strategy. Muốn đổi strategy thì truyền `strategy="structure"` khi khởi tạo.

## 4. LLM-as-Judge

### 4.1 RAG HILT Evaluator (online, per query)

`src/reviews/evaluator.py:13` chấm 4 tiêu chí 0-5, nói đơn giản:

1. **Faithfulness (trung thực):** câu trả lời có bịa không — mọi ý phải có trong context.
2. **Contextual Precision (độ trúng):** tài liệu lấy về có liên quan không.
3. **Contextual Recall (độ đủ):** có dùng đủ tài liệu cần thiết không.
4. **Answer Relevance (đúng ý):** có trả lời đúng câu hỏi không.

Flow: ghép context (`[Source X] content[:800]`, tối đa ~6000 ký tự) → gọi LLM (`temperature 0.1`, JSON) → parse điểm. Không có API key thì fallback heuristic (có citation → 4.0, không → 2.5...). Ngưỡng `3.5`: metric nào dưới là fail → `is_low_confidence=true` → chuyển bác sĩ/dược sĩ/chuyên gia theo từ khóa (thuốc → pharmacist; faithfulness fail → doctor).

### 4.2 RAGAS Tracing (offline, per trace)

Bảng `ragas_evaluations` (`tracing_db.py:118`) lưu 5 điểm (4 trên + fluency) + comment + `failed_metrics` + `confidence`. Trigger: `POST /v1/admin/traces/{trace_id}/ragas` (admin, khi `RAGAS_ON_DEMAND=true`). Lưu ý: memory rows bị lọc khỏi evaluator để không thổi điểm faithfulness.

### 4.3 Benchmark B5 (batch)

`scripts/run_benchmarks.py:228`: retrieve → format context → LLM sinh (`temp 0.2`) → LLM judge (`temp 0`) → parse JSON → trung bình 4 điểm. Output vào `results_benchmark_*.json`.

## 5. Prompt Versioning & Tracing

- Bảng `prompts`: mỗi tone 1 version active, version = sha8(text). Registry 12 tones ở `src/shared/prompt_manager.py:31`.
- Mỗi `rag_chat` ghi 7 spans (`cache_check → ... → format_answer`) + `total_latency_ms` + `prompt_version` vào `tracing.db`. Giữ 30 ngày, xem ở trang admin tracing.

## 6. Feedback

Bảng `feedback` (`tracing_db.py:133`): UI 👍/👎 gắn `trace_id`, lưu `score/value/comment` để cải thiện prompt sau.

## 7. Cách chạy

<details>
<summary>Mở rộng: lệnh chạy chi tiết (click để mở)</summary>

```bash
python -m src.evaluation
pytest tests/test_anomaly_detector.py tests/test_soap_summary.py tests/test_api_diabetes.py tests/test_glucose_tracker.py -v
python scripts/run_benchmarks.py --latency-n 10 --judge-n 5
python scripts/run_benchmarks.py --skip-judge
```

</details>

3 bước tối thiểu: chạy retrieval (không cần LLM) → gọi `/v1/query` xem `evaluation.failed_metrics` → chạy batch B1-B5 khi cần số báo cáo.

## 8. Liên hệ với Benchmark & Data Contract

- Fingerprint/hash định nghĩa ở `docs/data_contract.md`; ở đây dùng `normalize_doc_name` để khớp `source_id` với đáp án vàng.
- Số `Recall@5/Hit@5/MRR` và `avg_Faithfulness` điền sang `docs/benchmark.md`.

### Checklist tự kiểm tra

- [ ] Kể ví 10 quả táo cho cả 3 metrics không nhìn tài liệu.
- [ ] Liệt kê 4 tiêu chí judge + ngưỡng 3.5 + ai nhận review khi fail.
- [ ] Chạy được 1 lệnh pytest và chỉ ra file output.
