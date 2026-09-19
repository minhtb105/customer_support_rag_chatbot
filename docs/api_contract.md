# Hợp Đồng API — WHO-RAG Infrastructure API

> Base: `http://localhost:8000` · Prefix: `/v1` · Title: `WHO-RAG Infrastructure API` `1.0.0` (`src/config.py:214` `src/api/main.py:102`). OpenAPI tự sinh tại `/docs` (Swagger) và `/openapi.json`. Mọi endpoint dưới đây trích từ `src/api/main.py`, `src/api/schemas.py`, `src/auth/*`, `src/reviews/*`, `src/monitors/router.py`, `src/admin/*`.

## 1. Quy ước chung

| Mục | Giá trị |
|---|---|
| Auth | `Authorization: Bearer <JWT access_token>` hoặc cookie `access_token` (`src/auth/dependencies.py:22 _extract_token`). Thất bại → `401 {"detail":"Not authenticated"}`. Expert chưa `is_verified` → `403 {"detail":"Expert account not verified"}` |
| JWT | `HS256`, `exp 30m` (`ACCESS_TOKEN_EXPIRE_MINUTES`), `refresh 7d`, `jti`, `type: access|refresh` (`src/auth/security.py:37`) |
| CORS | `allow_origins = env FRONTEND_URL` split `,` (`src/api/main.py:105`), `allow_credentials=True` (không dùng `*` cùng credentials — đã fix `test_a05_cors_misconfiguration`) |
| Content-Type | `application/json` trừ `POST /v1/soap/generate/markdown` → `text/plain` |
| Lỗi chung | `422 Validation Error` (pydantic), `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `500 RAG error` |
| Phân trang | `limit 1..100` (mặc định 10-20), `offset >=0`, `total` trả kèm |
| Trace | `X-Trace-Id` không header riêng, mà trả trong body `trace_id`, `prompt_version`, `langsmith: {trace_id, prompt_version}` |

## 2. Health & Discovery

### `GET /health` và `GET /v1/health` — public
```json
{
  "status": "ok",
  "version": "1.0.0",
  "embedding_provider": "openai|local",
  "embedding_model": "text-embedding-3-small|all-MiniLM-L6-v2",
  "pdf_count": 34,
  "vector_db_ready": true
}
```
Logic `health()` đếm `PDF_DIR.rglob("*.pdf")` và check `embeddings/pdf_db/structure` hoặc `pdf_db_openai/structure` tùy `EMBEDDING_PROVIDER`.

### `GET /` — public
Liệt kê 8 nhóm: `health`, `query`, `glucose`, `bp`, `respiratory`, `mood`, `vitals`, `soap`, `auth`, `reviews`, `notifications`, `monitors`.

### `GET /v1/guidelines/status` — public
```json
{"total_pdfs":34,"by_source":{"gold":1,"gina":1,"diabetes":4},"manifest_path":"data/guideline_manifest.json","last_updated":"2026-09-06T...Z"}
```

## 3. Auth — `src/auth/router.py`

| Method | Path | Body | Resp | Auth |
|---|---|---|---|---|
| POST | `/v1/auth/register` | `{username, email?, password, full_name?, role: user|doctor|pharmacist|specialist}` | `201 {id,username,role}` | public (expert tạo sẽ `is_verified=false` chờ admin) |
| POST | `/v1/auth/login` | `{username, password}` | `{access_token, refresh_token, user}` + Set-Cookie `access_token` httpOnly `lax` | public |
| POST | `/v1/auth/refresh` | `{refresh_token}` | `{access_token}` | public (check `revoked`) |
| POST | `/v1/auth/logout` | — | `200` + clear cookie, revoke refresh | auth |
| GET | `/v1/auth/me` | — | `{id,username,role,is_verified}` | auth |
| GET | `/v1/auth/notifications?limit=50&unread_only=false` | — | `[{id,type,title,body,review_id,is_read,created_at}]` | auth |
| POST | `/v1/auth/notifications/{id}/read` | — | `200` | auth (own) |
| GET | `/v1/admin/users` | `?role=&limit=&offset=` | `[{id,username,role,is_verified}]` | admin |

`type` notification: `review_pending`, `review_pending_user`, `review_decided`, `monitor_pending` (từ `src/monitors/service.py`).

## 4. RAG Query (HILT) — `POST /v1/query` — auth bắt buộc

**Request** `src/api/schemas.py:11 QueryRequest`
```json
{
  "query": "Dấu hiệu đái tháo đường type 2 là gì?",
  "top_k": 5,              // 1..20, default 5
  "tone": "strict|friendly|balanced|null", // null → auto detect
  "user_id": "default_user", // nếu role=user thì phải trùng token sub, expert/admin được override
  "include_audit": true
}
```

**Response** `QueryResponse` + HILT extensions (200):
```json
{
  "answer": "Theo WHO ... [Source 1]",
  "cited_sources": [1],
  "contexts": [{"source_id":"who_hearts","content":"...","score":0.82,"dataset":"who_iris","section_path":["Diabetes"],"page_numbers":[12]}],
  "audit": {
    "cited_sources":[1],
    "citations":[{"source_id":"1","dataset":"who_iris","section_path":["Diabetes"],"page_numbers":[12],"score":0.82,"content_snippet":"...300ch"}],
    "prompt_version":"a1b2c3d4",
    "reranker_model":"cross-encoder/ms-marco-MiniLM-L-6-v2",
    "latency_ms": 842.5
  },
  "langsmith": {"trace_id":"abc123","prompt_version":"a1b2c3d4"},
  "cache_hit": false,
  "timings": {"retrieve_context":0.12,"rerank_contexts":0.08,"generate_answer":0.62},
  "status": "answered|pending_review",
  "review_id": "rev_123|null",
  "evaluation": {
    "metrics":{"faithfulness":4.2,"context_precision":3.8,"context_recall":4.0,"answer_relevance":4.5},
    "comments":{"faithfulness":"..."},
    "failed_metrics":["context_precision"],
    "is_low_confidence":false,
    "confidence":0.82,
    "routed_role":"doctor|pharmacist|specialist",
    "thresholds":{"faithfulness":3.5}
  },
  "is_low_confidence": false,
  "effective_user_id": "user_uuid",
  "trace_id": "abc123"
}
```

**Luồng** `src/api/main.py:172 rag_query`:
1. `effective_user_id = token.sub` (enforce `user_id` mismatch → 403).
2. `rag_chat(query, top_k, user_id, username)` → `retrieve_context` (hybrid 0.6) → `rerank top3` → `generate_answer` (tone auto).
3. `evaluate_rag(query, answer, contexts)` — nếu `is_low_confidence` (bất kỳ metric <3.5) → `create_review_request` → `update_trace(status=pending_review, is_low_confidence=1, routed_role)` → `status=pending_review` + `review_id`.
4. Ngược lại → `add_query_history(status=answered)` + `update_trace(answered)`.
5. `include_audit=true` → build `AuditTrail` 300ch snippet.

**Lỗi:**
- `401` thiếu token, `403` `user_id` mismatch (user role), `500 RAG error: ...`.

## 5. Vitals — 4 bệnh, unified

Tất cả đều `Depends(get_current_user)` + `_resolve_user_id` / `enforce_user_ownership` (user chỉ own, expert/admin override).

### 5.1 Diabetes — Glucose

`POST /v1/glucose` `GlucoseLogCreate`: `{user_id, value_mgdl 20..800, measured_at?, context: fasting|pre_meal|post_meal_2h|bedtime|random, notes?}` → `GlucoseLogOut{id,user_id,value_mgdl,measured_at,context,notes,classification,message}`

`classification` theo `GLUCOSE_THRESHOLDS_MGDL` (`config.py:97`): `low (<70) | normal | elevated | high | critical (>=300)` + `fasting_diabetes 126`, `postprandial_diabetes 200`.

`GET /v1/glucose/{user_id}?limit=50&days=14` → `{user_id, logs[], stats: GlucoseStats{total_logs, avg_mgdl, last_7_days_avg, streak_days, logs_per_week, classification_counts}, should_escalate: bool}`

`should_escalate_to_doctor`: `critical 1 lần` hoặc `>=3 lần high` trong `should_escalate_to_doctor`.

### 5.2 Hypertension — BP

`POST /v1/bp` `BpLogCreate`: `{user_id, systolic 50..300, diastolic 30..200, measured_at?, context: morning|evening|random|post_exercise|clinic, notes?}` → `BpLogOut{classification: normal|elevated|stage1|stage2|crisis}`

`GET /v1/bp/{user_id}` → `{logs, stats: BpStats{avg_sys,avg_dia,last_7_days_avg,streak_days,logs_per_week,classification_counts,at_target_rate}, should_escalate}`

Ngưỡng `BP_THRESHOLDS_MMHG` AHA/ACC 2025: `normal <120/80`, `elevated 120-129/<80`, `stage1 130-139/80-89`, `stage2 ≥140/90`, `crisis ≥180/120`.

### 5.3 Respiratory — Asthma/COPD

`POST /v1/respiratory` `RespiratoryLogCreate`: `{user_id, peak_flow_percent 10..150?, personal_best 50..1000?, cat_score 0..40?, inhaler_correct?, inhaler_steps_correct 0..20, inhaler_steps_total 1..20, measured_at?, context, notes?}` → `RespiratoryLogOut{peak_flow_percent,gold_stage,classification: green|yellow|red (>=80%, 50-79%, <50%), message}`

`GET /v1/respiratory/{user_id}` → `{logs, stats: RespiratoryStats{avg_peak_flow,red_rate,incorrect_inhaler_rate,classification_counts}, should_escalate}` (yellow/red hoặc CAT>20).

### 5.4 Mental — PHQ-9/GAD-7

`POST /v1/mood` `MoodLogCreate`: `{user_id, phq9_score 0..27?, gad7_score 0..21?, mood_notes? (PII-redacted trước lưu), measured_at?, context}` → `MoodLogOut{phq9,gad7,mood_notes:redacted, original_notes_had_pii, classification: minimal|mild|moderate|severe|crisis, message, crisis_flag, crisis_keywords[]}`

PII redact `PII_REDACT_FIELDS={name,phone,email,address,cmnd,cccd}`, crisis keywords 9 từ (`tự tử`, `tu tu`, `suicide`…), hotline `1800-1567 / 1900-1267 / 115`.

`GET /v1/mood/{user_id}` → `{logs, stats: MoodStats{avg_phq9,avg_gad7,crisis_count,classification_counts}, should_escalate}` (crisis hoặc severe).

### 5.5 Unified Vitals

`POST /v1/vitals` `VitalsLogCreate`: `{user_id, disease: diabetes|hypertension|respiratory|mental, ...fields tùy disease}` → `{"disease": "...", "result": <log>}` — tiện cho frontend một endpoint.

## 6. SOAP Pre-visit — `POST /v1/soap/generate` + `/markdown`

**Request** `SoapGenerateRequest`: `{user_id, days 1..90 default14, include_logs=true, language vi|en default vi, disease diabetes|hypertension|respiratory|mental|all default diabetes}`

**Response** `SoapResponse`: `{user_id, generated_at, period: "2026-08-23 to 2026-09-06", soap: {subjective, objective, assessment, plan}, stats: <disease stats>, disease}`

- `language` truyền vào `src/features/soap_summary.py:generate_soap` → prompt `SOAP_PROMPT` (VI) hoặc EN fallback.
- `disease` chọn stats tương ứng (glucose/bp/respiratory/mood).
- `/markdown` trả `text/plain` đã `soap_to_markdown`.

Auth: user chỉ `user_id == token.sub`, expert/admin được `any`.

## 7. Reviews (HILT) — `src/reviews/router.py`

| Method | Path | Query/Body | Resp | Role |
|---|---|---|---|---|
| GET | `/v1/reviews?status=pending&routed_role=doctor&disease=diabetes&limit=50&offset=0` | filter | `{total, items:[{id,query,draft_answer,contexts[],confidence,failed_metrics[],routed_role,status,requester_id,disease,created_at}]}` | expert/admin |
| GET | `/v1/reviews/{id}` | — | `{review + vitals:{glucose_logs,bp_logs,respiratory_logs,mood_logs} + requester}` | expert/admin |
| POST | `/v1/reviews/{id}/decide` | `{decision: approved|rejected|revised, final_answer?, expert_notes?}` | `{review}` | expert/admin (assigned) |
| GET | `/v1/reviews/my?status=` | — | own requests | auth |

## 8. Monitors — `src/monitors/router.py` (13 routes)

| Method | Path | Role | Ghi chú |
|---|---|---|---|
| GET | `/v1/monitors/status` | auth | `{stats:{pending_guidelines,pending_alerts,total_*}, sources[10], recent_runs[10]}` |
| GET | `/v1/monitors/sources` | auth | 10 monitored_sources |
| GET | `/v1/monitors/runs?source_key=&limit=20` | auth | monitor_runs |
| POST | `/v1/monitors/check/guidelines?source_key=&force=false` | specialist/doctor/admin | 1 hoặc all (ada_soc,gold,gina,who_mhgap,who_diabetes,aha_acc_htn,byt_diabetes), tạo `run_id`, `found_new` |
| POST | `/v1/monitors/check/safety?source=fda|byt|all` | pharmacist/admin | FDA daily / BYT weekly |
| GET | `/v1/monitors/guidelines?status=&source=&limit=20&offset=0` | auth | `guideline_versions` |
| GET | `/v1/monitors/guidelines/{gid}` | auth | detail + `change_summary_json_parsed` |
| POST | `/v1/monitors/guidelines/{gid}/decision` | specialist/doctor/admin | `{decision: approved|rejected, notes}` → move staging→corpus + reindex |
| GET | `/v1/monitors/alerts?status=&severity=&source=&limit=20` | auth | `safety_alerts` |
| GET | `/v1/monitors/alerts/{aid}` | auth | detail + `ai_summary` VI |
| POST | `/v1/monitors/alerts/{aid}/decision` | pharmacist/admin | `{decision: approved|dismissed, notes}` |
| POST | `/v1/monitors/maintenance/cleanup-superseded` | admin | xóa superseded >30d |
| GET | `/v1/monitors/stats` | auth | `{pending_guidelines, pending_alerts, ...}` |

## 9. Admin — Tracing & Prompts

### Tracing `src/admin/tracing_router.py: /v1/admin/*`
- `GET /v1/admin/traces?user_id=&tone=&status=&q=&is_low=&limit=10&offset=0` → `30d` cutoff, `total`.
- `GET /v1/admin/traces/{trace_id}` → `{trace, spans[], chunks[], ragas, review, feedback[]}`
- `POST /v1/admin/traces/{trace_id}/ragas` → chạy `evaluate_rag` on-demand, lưu `ragas_evaluations`.
- `DELETE /v1/admin/traces/expired` → `delete_expired_traces(30d)`.
- `GET /v1/admin/traces/stats` → `total, pending_review, low_confidence`.

### Prompts `src/admin/prompt_router.py: /v1/admin/prompts`
- `GET /v1/admin/prompts?tone=&status=&limit=100` → `is_active` + `version` SHA8.
- `POST /v1/admin/prompts` `{tone, text, description}` → `pending_approval`, version `sha256(text)[:8]`, dedup per tone.
- `POST /v1/admin/prompts/{tone}/{version}/approve` → deactivate cũ → active mới.
- `POST /v1/admin/prompts/{tone}/{version}/reject` → archive.
- `POST /v1/admin/prompts/dry-run` `{tone, text, queries:[5]}` → chạy 5 golden queries so sánh `active vs draft`.

## 10. Ví dụ cURL

```bash
# Login
curl -X POST http://localhost:8000/v1/auth/login -H "Content-Type: application/json" -d '{"username":"demo","password":"demo123"}'
# → {"access_token":"eyJ...","refresh_token":"...","user":{"id":"...","role":"user"}}

# RAG query (sẽ pending nếu low confidence)
curl -X POST http://localhost:8000/v1/query -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"query":"Ngưỡng chẩn đoán đái tháo đường theo WHO?","top_k":5}'

# Log vitals
curl -X POST http://localhost:8000/v1/glucose -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"user_id":"<id>","value_mgdl":145,"context":"fasting"}'

# SOAP
curl -X POST http://localhost:8000/v1/soap/generate -H "Authorization: Bearer $TOKEN" -d '{"user_id":"<id>","days":14,"language":"vi","disease":"diabetes"}'

# Monitors — guideline check (specialist)
curl -X POST "http://localhost:8000/v1/monitors/check/guidelines?source_key=gold&force=true" -H "Authorization: Bearer $SPEC_TOKEN"

# Duyệt guideline
curl -X POST http://localhost:8000/v1/monitors/guidelines/<gid>/decision -H "Authorization: Bearer $SPEC_TOKEN" -H "Content-Type: application/json" -d '{"decision":"approved","notes":"Đã kiểm tra, đưa vào corpus"}'

# Duyệt safety (pharmacist)
curl -X POST http://localhost:8000/v1/monitors/alerts/<aid>/decision -H "Authorization: Bearer $PHARM_TOKEN" -d '{"decision":"approved"}'
```

## 11. Mã lỗi & Validation

- `422` — pydantic `Field(ge,le)` ví dụ `value_mgdl 20..800`, `top_k 1..20`, `days 1..90`.
- `401` — thiếu/expired token, `403` — `user can only act on own user_id` hoặc `Requires role (specialist,)` hoặc `Expert not verified`.
- `400` — `Review already approved`, `Invalid decision`, `staging_path not exists`.
- `500` — `RAG error: ...` (wrap `rag_chat` exception).
