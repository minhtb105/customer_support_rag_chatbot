# Sơ Đồ Kiến Trúc — WHO-RAG System

> 4 sơ đồ Mermaid + bảng ràng buộc. Nguồn: `src/config.py`, `src/rag_pipeline.py:36`, `src/observability/tracing_db.py:32`, `src/indexer.py:98`, `src/monitors/*`, `frontend/app/**`.

## 1. Tổng quan hệ thống

```mermaid
graph TB
  subgraph Client
    NextJS[Next.js 14 App Router<br/>frontend/app]
    Playwright[E2E Playwright]
  end
  subgraph API["FastAPI - WHO-RAG Infrastructure API<br/>src/api/main.py:102"]
    Health[GET /health<br/>pdf_count + vector_ready]
    Query[POST /v1/query<br/>HILT + Audit]
    Vitals[POST/GET /v1/glucose|bp|respiratory|mood|vitals]
    SOAP[POST /v1/soap/generate]
    Auth[POST /v1/auth/*]
    Reviews[GET/POST /v1/reviews]
    Monitors[POST/GET /v1/monitors/*]
    Admin[GET/POST /v1/admin/traces|prompts]
  end
  subgraph RAG["RAG Pipeline v2<br/>src/rag_pipeline.py:36"]
    Cache[CAGHybridCache<br/>exact + semantic FAISS 0.82<br/>src/cache.py]
    Retriever[Hybrid Retriever<br/>BM25 + Vector 0.6<br/>src/retriever.py]
    Rerank[CrossEncoder<br/>ms-marco-MiniLM-L-6-v2]
    Generator[LLM Generator<br/>OpenAI/Groq + 12 prompts]
    Memory[Memory<br/>short 500tok + episodic 300tok<br/>+ long-term 3 facts]
    Evaluator[LLM-as-judge<br/>faithfulness 3.5]
  end
  subgraph Storage
    PDFs[data/raw/pdfs<br/>34 PDFs 18+1+4+5+3+3]
    Staging[data/raw/staging<br/>pending_review]
    Chroma[(Chroma<br/>pdf_db 4x33-39MB<br/>pdf_db_openai 1.7MB)]
    Meta[(metadata_store.db<br/>files 75, chunks 11428)]
    AuthDB[(auth.db<br/>users/reviews/notifications)]
    Tracing[(tracing.db<br/>traces/spans/chunks<br/>prompts/ragas)]
    MonitorDB[(monitoring.db<br/>10 sources)]
    VitalsDB[(vitals.db)]
  end
  subgraph Agents["Monitors Scheduler<br/>src/monitors/scheduler.py"]
    FDA[FDA daily 02:00<br/>api.fda.gov 240/min]
    BYT[BYT weekly Mon 03:00<br/>thuvienphapluat.vn]
    Guideline[Guideline monthly HEAD<br/>+ quarterly deep]
  end

  NextJS -->|authFetch 401 refresh| API
  Query --> RAG
  RAG --> Cache
  Cache -->|miss| Retriever
  Retriever --> Chroma
  Retriever --> Meta
  Rerank --> Generator
  Generator --> Evaluator
  Evaluator -->|low confidence| Reviews
  API --> Storage
  Agents --> Staging
  Staging -->|approve| PDFs
  PDFs -->|indexer 4 strategies| Chroma
```

**Điểm nhấn:**
- 4 chiến lược chunk `structure/sliding/semantic/hybrid` mỗi thứ 1 DB riêng, tách `pdf_db` (384d) vs `pdf_db_openai` (1536d) để tránh mismatch.
- HILT: `evaluate_rag` ngưỡng `3.5/5` trên 4 metrics, fail → `routed_role` (doctor/specialist/pharmacist) + `pending_review`.
- CAG 24h TTL, semantic threshold `0.82`, `max_size 1024` — cache hit bypass retrieval.

## 2. Luồng dữ liệu — Guideline & Safety (5 bước)

```mermaid
sequenceDiagram
  participant Crawler as Crawler<br/>crawl_guidelines.py<br/>+ guideline_fetcher.py
  participant Staging as Staging<br/>data/raw/staging
  participant LLM as LLM Summarizer<br/>guideline_diff VI
  participant DB as monitoring.db<br/>guideline_versions
  participant Expert as Specialist/Doctor
  participant Corpus as Corpus<br/>data/raw/pdfs
  participant Indexer as Indexer<br/>hybrid_hash_reindex

  Crawler->>Crawler: HEAD etag/last-modified<br/>so monitored_sources.last_etag
  alt etag đổi
    Crawler->>Staging: download staging/<source>/<version>/file.pdf<br/>sha256 dedup vs approved
    Crawler->>Crawler: extract docling → diff vs approved text
    Crawler->>LLM: old_text + new_text → tom_tat_tieng_viet
    Crawler->>DB: INSERT status=pending_review<br/>change_summary_json
    Crawler->>Expert: notify in-app monitor_pending<br/>specialist+doctor
    Expert->>DB: POST /guidelines/{gid}/decision approved
    Expert->>Staging: move staging→corpus/<source>/file.pdf
    DB->>DB: supersede cũ → superseded
    Staging->>Indexer: reindex_single_pdf ×4 strategies<br/>vector_db.delete + add_texts
    Indexer->>DB: upsert_file_and_chunks + indexed_at
    Indexer->>Crawler: clear CAGHybridCache._exact
  else không đổi
    Crawler->>DB: update last_checked_at
  end

  Note over Crawler,Indexer: Safety tương tự: FDA openFDA JSON<br/>cached 3600s → safety_alerts pending → pharmacist → approved/dismissed<br/>không index vào vector, chỉ alert
```

**Tầng an toàn:**
- SSRF allowlist 10 domains trước mọi `HEAD/GET`.
- `%PDF` magic + `<1KB` reject.
- Dedup `sha256` và `alert_url+title 7d`.

## 3. Sequence HILT — RAG Query

```mermaid
sequenceDiagram
  participant User as User<br/>Next.js
  participant API as POST /v1/query<br/>src/api/main.py:172
  participant RAG as rag_chat<br/>src/rag_pipeline.py:36
  participant Cache as CAGHybridCache
  participant Ret as retrieve_context<br/>Hybrid 0.6
  participant Rer as rerank top3
  participant LLM as generate_answer<br/>12 prompts
  participant Eval as evaluate_rag<br/>faithfulness 3.5
  participant DB as tracing.db + auth.db

  User->>API: {query, top_k=5, user_id}
  API->>RAG: rag_chat(query, top_k, user_id, username)
  RAG->>DB: start_trace(user_id, tone, top_k)
  RAG->>Cache: get(query)
  alt hit
    Cache-->>RAG: LLMOutput cached
    RAG->>DB: end_trace(answered)
    RAG-->>API: {answer, contexts, cache_hit:true}
  else miss
    RAG->>Ret: hybrid BM25+vector
    Ret->>DB: add_span_chunks
    RAG->>Rer: CrossEncoder rerank top3
    RAG->>LLM: system_prompt(tone) + Context[Source X|Score] + Question<br/>+ memory short 500tok
    LLM-->>RAG: {answer, cited_sources}
    RAG->>Cache: put(query, LLMOutput)
    RAG->>DB: end_trace(answered)
    RAG-->>API: {answer, contexts, timings}
  end
  API->>Eval: evaluate_rag(query, answer, contexts)
  alt is_low_confidence (any metric <3.5)
    API->>DB: create_review_request<br/>status pending + notification routed_role
    API->>DB: update_trace(pending_review, is_low=1)
    API-->>User: {status: pending_review, review_id, evaluation}
  else ok
    API->>DB: add_query_history(answered)
    API->>DB: update_trace(answered, is_low=0)
    API-->>User: {status: answered, audit{ citations 300ch, prompt_version, latency_ms}}
  end
```

**Tone auto:**
`mental > hypertension > respiratory > diabetes > strict|friendly|balanced` theo keyword `src/generator.py:detect_tone_and_temp`.

**Spans:** `cache_check → retrieve_context → rerank_contexts → build_llm_input → generate_answer → cache_put → update_memories → format_answer` (7 spans, `duration_ms` tính từ `start_at`).

## 4. ERD — 6 DBs

```mermaid
erDiagram
  files ||--o{ chunks : "file_name FK CASCADE"
  files {
    text file_name PK "fname::strategy"
    text file_hash "size+mtime+head/tail"
    real updated_at
  }
  chunks {
    text chunk_hash PK "text+section+page+index"
    text file_name FK
    int chunk_index
    text vector_id "fname_chunk_hash[:12]"
    text extra_meta "JSON"
  }

  users ||--o{ review_requests : "requester_id"
  users ||--o{ notifications : "user_id"
  users ||--o{ query_history : "user_id"
  review_requests {
    text id PK
    text query
    text draft_answer
    text contexts_json
    real confidence
    text routed_role "doctor|specialist|pharmacist"
    text status "pending|approved|rejected|revised"
    text disease
  }

  traces ||--o{ spans : "trace_id CASCADE"
  spans ||--o{ span_chunks : "span_id CASCADE"
  traces ||--o{ span_chunks : "trace_id"
  traces ||--o{ ragas_evaluations : "trace_id UNIQUE"
  traces ||--o{ feedback : "trace_id"
  traces {
    text id PK
    text user_id
    text query
    text answer
    text status "pending_review|answered|failed"
    text tone "diabetes|hypertension|..."
    text prompt_version "sha8"
    real total_latency_ms
    boolean is_low_confidence
    text routed_role
    text review_id
    text created_at "30d retention"
  }
  prompts {
    text tone
    text version "sha256(text)[:8] UNIQUE"
    text status "draft|pending_approval|active|archived"
    boolean is_active
  }

  monitored_sources ||--o{ guideline_versions : "source"
  monitored_sources ||--o{ monitor_runs : "source_key"
  monitored_sources {
    text source_key PK "ada_soc"
    text check_interval "monthly|weekly|daily"
    text risk_tier "critical|high|medium"
    text last_etag
    text last_hash
  }
  guideline_versions {
    text id PK
    text source
    text version_label
    text status "pending_review|approved|superseded"
    text staging_path
    text corpus_path
    text sha256
    text change_summary_json "VI"
  }
  safety_alerts {
    text id PK
    text source "FDA|DAV|MOH"
    text severity "critical|high|medium|low"
    text alert_type "recall|box_warning"
    text status "pending_review|approved|dismissed"
  }

  vitals {
    text user_id
    text disease_type "diabetes|hypertension|respiratory|mental"
    real value_mgdl
    int systolic_diastolic
    int phq9_gad7
    text crisis_flag
  }
```

**Retention & Index:**
- `tracing.db` `list_traces` enforce `created_at >= now-30d` + `delete_expired_traces`.
- `monitoring.db` `superseded` xóa sau `MONITOR_SUPERSEDED_RETENTION_DAYS=30`.

## 5. Gaps đã biết (cần xử lý trước prod)

| # | Gap | Ảnh hưởng | Mitigation trong doc |
|---|---|---|---|
| 1 | `scheduler` chưa wire vào FastAPI lifespan | Chỉ chạy sidecar, dev không có cron | Ghi trong `monitoring.md#Scheduling` + `dashboard_ui#Roadmap` |
| 2 | BYT scrape chưa test HTML thật | Có thể parse sai `snapshot.html` | Đã mock trong test, cần integration test với `thuvienphapluat.vn` thật |
| 3 | FDA `cached_get` chưa backoff 429 | Có thể miss critical recall | Thêm `time.sleep(0.25)` + TTL 1h, đề xuất exponential backoff |
| 4 | `frontend/app/monitor` chưa tồn tại | API có nhưng không có UI | Tạo stub trong batch cuối (TODO của bạn) |
| 5 | `reindex_single_pdf` chưa ghi `version` vào Chroma `metas` | Citation chưa hiện `GOLD 2025` | Cần `extra_meta.version_label` khi `add_texts` |
| 6 | `CAGHybridCache` chưa filter poisoned chunk | Indirect prompt injection via RAG doc | Ghi vào `owasp_llm_top10_gap_analysis#LLM03` |

## 6. Tham chiếu nhanh

- `src/config.py:240` constants, `src/rag_pipeline.py:36` flow, `src/observability/tracing_db.py:32` schema, `src/indexer.py:98` reindex, `src/monitors/scheduler.py: setup_schedule 02:00/03:00/04:00`.
- Kích thước DB: `metadata_store 6.7 MB 75/11428 chunks`, `pdf_db 33-39 MB ×4`, `monitoring 73 KB`.

```bash
# Verify
python -c "from src.api.main import app; print([r.path for r in app.routes if 'monitor' in r.path])"
ls embeddings/pdf_db/* data/raw/pdfs/* | wc -l
sqlite3 metadata/tracing.db "select count(*) from traces where created_at >= datetime('now','-30 days')"
```
