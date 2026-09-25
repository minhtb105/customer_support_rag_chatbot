# Sơ Đồ Kiến Trúc — WHO-RAG System

> Nguồn chính: `src/shared/config.py`, `src/chat/rag_pipeline.py:36`, `src/shared/observability/tracing_db.py:32`, `src/shared/indexer.py:98`, `src/monitors/*`, `frontend/app/**`.

> **Mục tiêu đọc-xong-làm-được:** vẽ lại luồng Patient → Doctor trên giấy, chạy `uvicorn` + `npm run dev`, giải thích được RAG/CAG/HILT/FQG/SOAP cho người mới.

## 0. Glossary canonical (1 dòng/thuật ngữ — các file khác link về đây)

- **RAG (Retrieval-Augmented Generation):** trả lời dựa trên tài liệu WHO/BYT đã lưu, kèm trích dẫn, để không bịa.
- **CAG (Cache-Augmented Generation):** cache câu trả lời 24h (khớp chính xác + khớp ngữ nghĩa FAISS 0.82), trúng cache thì bỏ qua retrieval.
- **BM25:** tìm kiếm theo từ khóa (đếm từ khớp), hợp với mã bệnh, tên thuốc.
- **Rerank:** chấm lại top kết quả bằng mô hình CrossEncoder rồi chỉ giữ top 3 tốt nhất.
- **Chroma:** kho vector lưu chunks đã embedding, hỏi là tìm đoạn gần nghĩa nhất.
- **SOAP (Subjective/Objective/Assessment/Plan):** mẫu ghi chú bác sĩ chuẩn 4 phần; ở đây P để trống cho bác sĩ quyết.
- **FQG (Follow-up Questions):** 2-3 câu hỏi gợi ý ngữ cảnh (ăn ngọt? quên thuốc?) trước khi escalate; hội thoại nhiều vòng là future-work.
- **HILT (Human-In-The-Loop):** AI chấm điểm câu trả lời, điểm thấp (<3.5/5) thì chuyển cho bác sĩ duyệt, không trả lời liều.

Chi tiết đo lường → xem `docs/evaluation.md`; dữ liệu/corpus → xem `docs/data_contract.md`.

## 1. Tổng quan hệ thống

Luồng 5 hộp (học thuộc lòng luồng này là đủ onboard):

```
[Patient log glucose] -> [API + Anomaly check] -> [RAG + Memory] -> [FQG truoc escalate] -> [Doctor: sparkline 90d + SOAP]
```

- Patient nhập chỉ số → `POST /v1/glucose` kiểm tra spike >250/<70 và trend 3 ngày.
- Hỏi đáp → `POST /v1/query` chạy RAG (cache → hybrid BM25+vector → rerank → LLM `gpt-4o-mini` demo).
- Memory 3 tầng (`src/shared/memory/`): short 500 tokens + episodic 300 tokens + long-term 3 facts; cuối tuần gom thành 1 fact/tuần.
- Doctor xem `previsit`: sparkline 90 ngày chấm đỏ anomaly (30%) + SOAP text (70%), mỗi nhận định có `[Xem log #id]`; bác sĩ còn có hàng đợi `/expert/patients` sort triage critical → trend → watch → safe.
- Kiến trúc module hóa đa chỉ số (BP/HR/SpO2) nhưng MVP tập trung sâu Glucose làm POC.

<details>
<summary>Mermaid chi tiết cũ (click để mở, không cần học thuộc)</summary>

Mermaid gốc 4 sơ đồ đã thu gọn. Muốn xem full thì `git log -- docs/architecture_diagram.md`. Tóm tắt: Client (Next.js) → FastAPI (`src/api/main.py` — composition root, router theo service ở `src/<service>/router.py`) → RAG Pipeline (`src/chat/rag_pipeline.py:36`) → Storage (Chroma + `metadata/*.db`) + Agents giám sát guideline.

</details>

### Tracks A/B/C (mỗi track 1 đoạn, chi tiết code ở file khác)

- **Track A — Hỏi đáp RAG + HILT:** `POST /v1/query` → cache → retrieve → rerank → sinh câu trả lời → `evaluate_rag` chấm 4 tiêu chí, thấp điểm thì `pending_review` cho bác sĩ. Xem contract ở `docs/api_contract.md`.
- **Track B — Theo dõi đường huyết + Anomaly:** `POST /v1/glucose` → lưu log → `anomaly_detector` (spike cứng + trend heuristic demo) → banner + FQG ở tracker. Ngưỡng cứng giữ ở `GLUCOSE_THRESHOLDS_MGDL`, demo LLM là `gpt-4o-mini`.
- **Track C — Pre-visit SOAP cho bác sĩ:** `POST /v1/soap/generate` → S gom notes FQG, O số cứng, A chỉ nói đạt/không đạt HbA1c<7% theo BYT QĐ5481 2020 + ADA 2024 (QĐ3192 là tăng huyết áp, không dùng cho ĐTĐ), **P = ""** để trống hoàn toàn.

### Structure condensed (đọc để biết file nào sửa khi nào)

- `src/api/main.py` — composition root, chỉ wire router + middleware; endpoint thật nằm ở `src/<service>/router.py` + schema ở `src/<service>/schemas.py`.
- `src/chat/` — service RAG: `rag_pipeline.py` luồng chính, `retriever.py` hybrid BM25+vector, `generator.py` LLM, `router.py` `/v1/query` (+ stream); cache CAG ở `src/shared/cache.py`.
- `src/diabetes/` — `glucose_tracker.py`, `anomaly_detector.py`, `scope_guard.py`, `soap_summary.py`, `followup_notes.py`.
- `src/triage/` — `triage_nlu.py`, `red_flag.py`, `triage_events.py`; `src/scheduling/` — `slot_generator.py`, `solvers.py`, `synthetic_roster.py` (gọi in-process từ triage).
- `src/vitals/` — `bp_tracker.py`, `respiratory_tracker.py`, `mood_tracker.py` (base chung `base_tracker.py`).
- `src/shared/` — hạ tầng dùng chung: `config.py`, `memory/`, `observability/tracing_db.py`, `indexer.py`, `embedding/`, `models/`, `evaluation.py`.
- `frontend/app/tracker`, `frontend/app/previsit`, `frontend/app/admin/tracing` — 3 màn hình chính.

## 2. Luồng dữ liệu — Guideline & Safety (5 bước)

1. Crawler HEAD kiểm tra etag/last-modified của nguồn (ADA, WHO, BYT...).
2. Có bản mới → tải về `data/raw/staging/`, kiểm tra `%PDF`, dedup sha256.
3. LLM tóm tắt thay đổi tiếng Việt → ghi `guideline_versions` trạng thái `pending_review`.
4. Chuyên gia duyệt trên UI → file move sang `data/raw/pdfs/`, bản cũ thành `superseded`.
5. Reindex 4 chiến lược chunk + xóa cache CAG để câu trả lời mới dùng ngay. FDA alerts tương tự nhưng chỉ cảnh báo, không index.

## 3. Sequence HILT — RAG Query

1. API nhận `{query, top_k=5, user_id}` → gọi `rag_chat`.
2. Cache trúng → trả ngay; trượt → hybrid retrieve (BM25 + vector, alpha 0.6) → rerank top 3.
3. LLM sinh câu trả lời kèm trích dẫn `[Source X]` + memory ngắn.
4. `evaluate_rag` chấm, metric nào <3.5 thì tạo review, gán bác sĩ/dược sĩ/chuyên gia, trả `pending_review`.

**Spans trace:** `cache_check → retrieve_context → rerank_contexts → build_llm_input → generate_answer → cache_put → update_memories → format_answer` (xem chi tiết ở `docs/evaluation.md`).

## 4. ERD — 6 DBs

| DB | File | Bảng chính để nhớ |
|---|---|---|
| Metadata | `metadata/metadata_store.db` | `files (fname::strategy)`, `chunks` 11428 rows |
| Vector | `embeddings/pdf_db/*` | 4 chiến lược × 33-39MB; `pdf_db_openai` 1.7MB riêng |
| Auth | `metadata/auth.db` | `users`, `review_requests`, `notifications` |
| Tracing | `metadata/tracing.db` | `traces/spans/span_chunks/prompts/ragas/feedback`, giữ 30 ngày |
| Monitoring | `metadata/monitoring.db` | `monitored_sources` 10 rows, `guideline_versions`, `safety_alerts` |
| Vitals | `metadata/vitals.db` | logs theo `disease_type` |

## 5. Gaps đã biết (cần xử lý trước prod)

| # | Gap | Nói đơn giản |
|---|---|---|
| 1 | Scheduler chưa wire vào lifespan | Cron chỉ chạy sidecar, dev không tự chạy |
| 2 | BYT scrape chưa test HTML thật | Có thể parse sai trang luật |
| 3 | FDA chưa backoff 429 | Có thể lỡ recall quan trọng |
| 4 | `frontend/app/monitor` chưa có UI | API có nhưng chưa có màn hình |
| 5 | Reindex chưa ghi version vào Chroma metas | Citation chưa hiện năm guideline |
| 6 | CAG chưa lọc poisoned chunk | Tài liệu độc có thể chui vào câu trả lời (xem file OWASP canonical) |

## 6. Tham chiếu nhanh

- Code: `src/shared/config.py:240` hằng số, `src/chat/rag_pipeline.py:36` luồng, `src/shared/observability/tracing_db.py:32` schema, `src/shared/indexer.py:98` reindex, `src/monitors/scheduler.py` lịch 02:00/03:00.
- **Why business (3-5 dòng, chi tiết ở `docs/benchmark.md` phụ lục):** FreeStyle Libre bán 2-4M VND/bộ chứng minh người dân chịu chi cho monitoring; BHYT chi trả telehealth từ 1/7/2025 + VN chỉ 14 bác sĩ/10k người nên AI triage có giá trị; hướng bán B2B2C cho phòng khám thay vì thu phí lẻ từng bệnh nhân.
- **Run demo 5 phút:**
  - Backend: `pip install -e .` rồi `uvicorn src.api.main:app --reload --port 8000` (cần `OPENAI_API_KEY`, demo chạy `gpt-4o-mini`, production AWQ/vLLM chỉ doc-only).
  - Frontend: `cd frontend && npm install && npm run dev`, mở tracker log 260 → xem banner, mở previsit xem sparkline + SOAP.
- **Lưu ý y khoa:** demo-only, không chẩn đoán mới; A chỉ đối chiếu HbA1c<7% (BYT QĐ5481 + ADA 2024).

### Checklist tự kiểm tra

- [ ] Vẽ lại 5 hộp không nhìn tài liệu.
- [ ] Giải thích được RAG vs CAG vs HILT bằng 1 câu mỗi cái.
- [ ] Chạy được demo và chỉ ra file code của từng bước.
- [ ] Nói được tại sao P để trống và QĐ3192 không dùng cho ĐTĐ.
