# Giao Diện Dashboard — Frontend Next.js

> Stack: `Next.js 14 App Router` + `Tailwind` + `Recharts` + `lib/api.ts` authFetch + `lib/auth.tsx` Context. Nguồn: `frontend/app/**`, `frontend/lib/**`, `src/admin/*`, `src/monitors/router.py`.

## 1. Sơ đồ route

```
/                           # page.tsx — Overview market evidence + corpus status
/(auth)/login/page.tsx      # login form URLSearchParams → /v1/auth/login
/(auth)/register/page.tsx   # register → auto login
/tracker/page.tsx           # Hướng A — Nhật ký đường huyết (diabetes tracker)
/previsit/page.tsx          # Hướng B — SOAP pre-visit
/api-playground/page.tsx    # Hướng C — WHO-RAG API playground
/expert/queue/page.tsx      # Hàng đợi duyệt HILT (specialist/doctor/pharmacist)
/my/reviews/page.tsx        # Lịch sử query của user
/admin/tracing/page.tsx     # Tra cứu traces 30d
/admin/tracing/[traceId]/page.tsx  # Chi tiết trace + spans + chunks + ragas
/admin/prompts/page.tsx     # Prompt versioning
/admin/users/page.tsx       # Quản lý user/verify expert
/monitor/guidelines/page.tsx  # *Stub mới* — Giám sát guideline
/monitor/alerts/page.tsx      # *Stub mới* — Cảnh báo thuốc
components/Header.tsx        # Nav theo role
lib/api.ts (162 dòng)       # authFetch 401→refresh 1 lần, credentials:include
lib/auth.tsx (113 dòng)     # AuthProvider Context
```

*Verify:*
```bash
Get-ChildItem frontend/app -Recurse -File | Select Name,Directory
ls frontend/app/monitor  # sau batch cuối
```

## 2. Auth & RBAC trên UI

`frontend/lib/auth.tsx:30 AuthProvider`:
- `fetchMe()` → `GET /v1/auth/me` credentials:include → `setUser`.
- `login(username,password)` → `POST /v1/auth/login` `application/x-www-form-urlencoded` → `setUser(data.user)`.
- `register` → `POST /v1/auth/register` JSON → auto `login`.
- `logout` → `POST /v1/auth/logout`.
- `isAdmin = role==="admin"`, `isExpert = [doctor,pharmacist,specialist]`, `isUser = role==="user"`.

`frontend/lib/api.ts:4 authFetch`:
```ts
fetch(url, {credentials:"include", headers:{"Content-Type":"application/json",...opts.headers}})
if 401 → POST /v1/auth/refresh (1 lần) → retry
```

`components/Header.tsx` render nav theo `isAdmin/isExpert`:
- `admin` → `Tracing | Prompts | Users | Monitor`
- `expert` → `Queue`
- `user` → `Tracker | Previsit | My Reviews`

Chưa login → `tracker/page.tsx:52` hiện `Cần đăng nhập → Link /login`.

## 3. Các trang hiện có (chi tiết)

### 3.1 `/` — Overview (`app/page.tsx:7`)

- Hero `BHYT chi trả telehealth từ 1/7/2025` gradient blue-indigo.
- Stat 4 ô: `7,3% ~7M`, `>60% chưa chẩn đoán`, `20,2% tự theo dõi`, `14/10k bác sĩ`.
- Khung 5 tiêu chí `Criteria` (ok/warn).
- Bảng `Nghiên cứu Thanh Nhàn 20,2%` + `by_source` badges từ `getGuidelinesStatus()` (`/v1/guidelines/status`).
- 3 TrackCard: Hướng A `/tracker`, B `/previsit`, C `/api-playground`.
- Verify: `getHealth()` → `embedding_provider · embedding_model` + `API_BASE`.

### 3.2 `/tracker` — Hướng A (`app/tracker/page.tsx:17`)

- Form: `value_mgdl` number + `context` select (fasting/post_meal_2h/random/pre_meal/bedtime) + `notes`.
- `logGlucose({user_id, value_mgdl, context, notes})` → `POST /v1/glucose` → `rec.classification` màu `classColor` (normal emerald, high red, critical red-600) + `rec.message`.
- KPIs: `total_logs`, `avg_mgdl`, `last_7_days_avg`, `logs_per_week` alert nếu `<3` (amber).
- Chart: `Recharts LineChart` 20 điểm gần nhất, `ReferenceLine y=126,200,70` (WHO thresholds), `Tooltip`.
- Escalate banner `should_escalate` nếu `critical 1 lần` hoặc `3×high`.
- Trend RAG: `explainTrend()` → ghép `last 5 logs` + stats → `queryRag(q,5,effectiveId)` → nếu `pending_review` hiện `⏳ Đang chờ duyệt (routed_role)`, else `answer`.

### 3.3 `/previsit` — Hướng B

`generateSoap(user_id, days=14, disease)` → `POST /v1/soap/generate` → `SoapResponse{subjective, objective, assessment, plan, stats}` + `generateSoapMarkdown` download `text/plain`.

### 3.4 `/api-playground` — Hướng C

`queryRag(query, top_k, user_id)` → `POST /v1/query include_audit:true` → render `answer`, `audit.citations` (source_id, page_numbers, snippet 300ch), `prompt_version`, `latency_ms`, `timings`, `evaluation` (nếu low).

### 3.5 `/expert/queue` — HILT Queue

`getReviews(status=pending)` → list `review_requests` theo `routed_role`. `getReviewDetail(id)` → `query, draft_answer, contexts[], confidence, vitals`. `decideReview(id, {decision, final_answer, expert_notes})` → `approved|rejected|revised`.

### 3.6 `/my/reviews` — User history

`getMyReviews()` → `query_history` của chính user, hiện `status pending_review/answered`.

### 3.7 `/admin/tracing` & `[traceId]`

- `adminListTraces({user_id,tone,status,q,is_low,limit=10,offset})` → `30d` cutoff, `total`.
- `adminGetTrace(id)` → `{trace, spans[7], chunks, ragas, review, feedback}`.
- `adminTriggerRagas(id)` → `POST /v1/admin/traces/{id}/ragas` on-demand.
- UI: filter, pagination `10`, table `created_at, status, is_low_confidence, routed_role`, detail tabs.

### 3.8 `/admin/prompts`

- `adminListPrompts(tone)` → `prompts` với `is_active, version sha8, status draft|pending_approval|active|archived`.
- `adminCreatePromptDraft(tone,text)` → `pending_approval`.
- `adminApprovePrompt(tone,version)` → active.
- `adminDryRun(tone,text,queries[5])` → so sánh active vs draft trên 5 golden queries.

### 3.9 `/admin/users`

`adminListUsers(role)` + `adminUpdateUser`, `adminVerifyExpert` → toggle `is_verified`.

## 4. Trang mới — Monitor (stub tạo ở batch cuối)

### `/monitor/guidelines` (cho specialist/doctor/admin)

- Table `GET /v1/monitors/guidelines?status=pending_review` → `id, source, title, version_label, sha256, staging_path, change_summary_json.tom_tat_tieng_viet, fetched_at`.
- Filter `source` (ada/gold/gina/who_mhgap/byt).
- Detail drawer: `GET /guidelines/{gid}` → diff `old vs new` (nếu có `corpus_path`), AI summary VI, nút `Approve / Reject` → `POST /guidelines/{gid}/decision`.
- Trigger: `POST /monitors/check/guidelines?source_key=gold&force=true` (nút Kiểm tra ngay).

### `/monitor/alerts` (cho pharmacist/admin)

- Table `GET /v1/monitors/alerts?severity=critical` → `drug_name, alert_title, severity badge critical red/high amber, source FDA|DAV|MOH, ai_summary.tom_tat_vi, published_date`.
- Detail: `ai_summary`, `raw_json`, `alert_url` link FDA.
- Decision: `POST /alerts/{aid}/decision {approved|dismissed}`.

**UI stub** sẽ là `use client` page đơn giản, gọi `authFetch` với `NEXT_PUBLIC_API_URL`, xử lý `401→login`, hiện `pending` count badge như `expert/queue`.

## 5. API binding matrix

| Page | API | Method | Role |
|---|---|---|---|
| Overview | `getHealth`, `getGuidelinesStatus` | GET /v1/health, /guidelines/status | public |
| Tracker | `logGlucose`, `getGlucose`, `queryRag` | POST/GET /glucose, POST /query | user (own), expert/admin override |
| Previsit | `generateSoap`, `generateSoapMarkdown` | POST /soap/generate | user own, expert any |
| Playground | `queryRag` | POST /query | auth |
| Expert Queue | `getReviews`, `getReviewDetail`, `decideReview` | GET/POST /reviews | expert/admin |
| My Reviews | `getMyReviews` | GET /reviews/me | auth |
| Admin Tracing | `adminListTraces`, `adminGetTrace`, `adminTriggerRagas` | GET/POST /admin/traces | admin |
| Admin Prompts | `adminListPrompts`, `adminCreatePromptDraft`, `adminApprovePrompt`, `adminDryRun` | GET/POST /admin/prompts | admin |
| Monitor Guidelines | `GET /monitors/guidelines`, `POST /check/guidelines`, `POST /guidelines/{gid}/decision` | GET/POST | specialist/doctor/admin |
| Monitor Alerts | `GET /monitors/alerts`, `POST /alerts/{aid}/decision` | GET/POST | pharmacist/admin |

## 6. Error & Empty states

- `401` → redirect `/login` (authFetch đã refresh 1 lần).
- `403` → toast `Requires role ...`.
- `422` → field validation (value_mgdl 20..800, top_k 1..20).
- Empty: `tracker` → `Chưa có dữ liệu`, `queue/my` → `Không có review nào`, `admin` → `Không có trace`.

## 7. E2E

`frontend/e2e/*.spec.ts` + `playwright.config.ts`:
- `overview.spec.ts` → hero stats, 5 criteria, 3 cards.
- `tracker.spec.ts` → form submit, chart, KPI.
- `previsit.spec.ts` → SOAP generate, markdown download.
- `api-playground.spec.ts` → query, audit trail, `top_k` slider.

Chạy: `cd frontend && npm run test:e2e` (cần `npm run dev` 3000 + FastAPI 8000).

## 8. Roadmap gap (ghi để không quên)

- `frontend/app/monitor` chưa tồn tại trước batch cuối — API đã sẵn, chỉ thiếu UI.
- Chưa có `WebSocket` realtime cho `monitor_pending` — hiện polling `getNotifications` + badge.
- Chưa có `dark mode` thống nhất.
