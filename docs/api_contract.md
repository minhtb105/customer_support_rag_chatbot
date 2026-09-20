# Hợp Đồng API — WHO-RAG Infrastructure API

> Base `http://localhost:8000`, prefix `/v1`. Tài liệu sống ở `/docs` (Swagger). Code: `src/api/main.py`, `src/api/schemas.py`.

> **Mục tiêu đọc-xong-làm-được:** login lấy token, log đường huyết, hỏi RAG, tạo SOAP, đọc được lỗi 401/422/403.

## 1. Quy ước chung

| Mục | Nói đơn giản |
|---|---|
| Auth | Gửi `Authorization: Bearer <token>` hoặc cookie `access_token`. Không có/sai → `401`, chưa verify expert → `403` |
| JWT | Token `HS256`, hết hạn 30 phút, refresh 7 ngày (`src/auth/security.py:37`) |
| CORS | Chỉ origin trong `FRONTEND_URL` mới gọi được kèm cookie |
| Lỗi chung | `422` sai kiểu dữ liệu, `401` chưa login, `403` không có quyền, `404` không thấy, `500` lỗi RAG |
| Phân trang | `limit 1..100`, `offset >=0`, kèm `total` |
| Trace | Không header riêng; `trace_id`, `prompt_version` nằm trong body trả về |

RAG (tìm tài liệu rồi sinh câu trả lời có trích dẫn) xem Glossary ở `docs/architecture_diagram.md`.

## 2. Health & Discovery

- `GET /health` (public): trả `{status, version, pdf_count, vector_db_ready}`. Dùng để kiểm tra backend sống chưa.
- `GET /` (public): liệt kê nhóm endpoint (health, query, glucose, soap, auth, reviews, monitors).
- `GET /v1/guidelines/status` (public): số PDF theo nguồn.

## 3. Auth — `src/auth/router.py`

Chỉ cần nhớ 3 cặp: `register` tạo user → `login` lấy token → `me` kiểm tra. Expert tạo ra bị `is_verified=false` chờ admin duyệt. Token refresh ở `/v1/auth/refresh`, logout ở `/v1/auth/logout`. Admin xem users ở `GET /v1/admin/users`. Chi tiết field xem Swagger `/docs`.

## 4. RAG Query (HILT) — `POST /v1/query` — auth bắt buộc

Hỏi guideline, AI trả lời kèm trích dẫn; điểm thấp thì chuyển bác sĩ duyệt (`pending_review`).

- Request tối thiểu: `{"query": "Dấu hiệu đái tháo đường type 2 là gì?", "top_k": 5}`. Thêm `tone` (strict/friendly/balanced) nếu muốn, `user_id` phải trùng token nếu role=user.
- Response tối thiểu: `{"answer": "...[Source 1]", "status": "answered|pending_review", "trace_id": "abc", "cache_hit": false}`. Đầy đủ còn có `contexts[]`, `audit` (prompt_version, latency), `evaluation` (4 điểm, `is_low_confidence`, `routed_role`).
- Luồng (`src/api/main.py:172`): check quyền → `rag_chat` (cache → retrieve hybrid 0.6 → rerank top3 → sinh) → `evaluate_rag`, thấp điểm (<3.5) thì tạo review.
- Ngoài-scope ĐTĐ (ví dụ "uống kháng sinh nào?") bị chặn bằng mẫu cứng TRƯỚC khi gọi RAG, không tốn lookup.

## 5. Vitals — 4 bệnh, unified

Tất cả cần login; user thường chỉ được thao tác trên `user_id` của mình.

### 5.1 Diabetes — Glucose

- `POST /v1/glucose`: `{"user_id": "...", "value_mgdl": 145, "context": "fasting"}` (`value 20..800`, context: fasting/pre_meal/post_meal_2h/bedtime/random). Trả thêm `classification` (low/normal/elevated/high/critical), `message`, `anomaly {spike|trend|none}`, `follow_up_questions[]`.
- Ngưỡng (`config.py:97`): low <70, critical ≥300, fasting cao ≥126, sau ăn ≥200. Spike >250/<70 luôn cảnh báo an toàn trước.
- `GET /v1/glucose/{user_id}?limit=50&days=14`: trả `logs[]` + `stats` (trung bình, streak) + `should_escalate` (critical 1 lần hoặc ≥3 high).

### 5.2 Hypertension — BP

`POST /v1/bp`: `{"user_id": "...", "systolic": 130, "diastolic": 85}` → `classification` normal/elevated/stage1/stage2/crisis (chuẩn AHA/ACC: crisis ≥180/120). GET tương tự glucose.

### 5.3 Respiratory — Asthma/COPD

`POST /v1/respiratory`: gửi `peak_flow_percent` → `classification` green (≥80%) / yellow (50-79%) / red (<50%). Kèm điểm CAT và kỹ thuật hít.

### 5.4 Mental — PHQ-9/GAD-7

`POST /v1/mood`: gửi `phq9_score 0..27`, `gad7_score 0..21` → `classification` minimal/mild/moderate/severe/crisis. Ghi chú tự động xóa PII (tên, phone, email), phát hiện từ khóa khủng hoảng thì gắn hotline 1800-1567/1900-1267/115.

### 5.5 Unified Vitals

`POST /v1/vitals`: một endpoint cho frontend, thêm field `disease` rồi gửi kèm field của bệnh đó.

## 6. SOAP Pre-visit — `POST /v1/soap/generate` + `/markdown`

- Request tối thiểu: `{"user_id": "...", "days": 14, "language": "vi", "disease": "diabetes"}` (`days 1..90`).
- Response tối thiểu: `{"soap": {"subjective": "...[Xem log #3]", "objective": "...", "assessment": "đạt/không đạt HbA1c<7%...", "plan": ""}, "period": "..."}`. **P luôn `""`** + placeholder mờ cho bác sĩ; S/O/A đều có link `[Xem log #id]`. A chỉ đối chiếu HbA1c<7% (BYT QĐ5481 2020 + ADA 2024), không chẩn đoán mới.
- `/markdown` trả `text/plain` đã format sẵn.

## 7. Reviews (HILT) — `src/reviews/router.py`

Dành cho bác sĩ/dược sĩ duyệt câu trả lời điểm thấp: `GET /v1/reviews?status=pending` lọc, `GET /v1/reviews/{id}` xem kèm vitals, `POST /v1/reviews/{id}/decide` với `{decision: approved|rejected|revised}`. User thường xem yêu cầu của mình ở `/v1/reviews/my`.

## 8. Monitors — `src/monitors/router.py` (13 routes)

Chỉ cần nhớ: `GET /v1/monitors/status` xem tổng quan; `POST /v1/monitors/check/guidelines?source_key=gold` (specialist) và `check/safety` (pharmacist) để quét bản mới; `GET .../guidelines` + `POST .../guidelines/{gid}/decision` để duyệt; alerts tương tự cho dược sĩ; `cleanup-superseded` (admin) dọn bản cũ >30 ngày.

## 9. Admin — Tracing & Prompts

Tracing (`src/admin/tracing_router.py`): `GET /v1/admin/traces` lọc + `GET /v1/admin/traces/{trace_id}` xem spans/chunks/ragas + `POST .../ragas` chấm lại + `DELETE .../expired` dọn >30 ngày. Prompts (`src/admin/prompt_router.py`): xem/tạo/duyệt version prompt theo tone, `dry-run` chạy 5 câu golden để so sánh.

## 10. Ví dụ cURL

Thử 3 bước sau khi có token (`TOKEN` từ login):

```bash
curl -X POST http://localhost:8000/v1/auth/login -H "Content-Type: application/json" -d '{"username":"demo","password":"demo123"}'
curl -X POST http://localhost:8000/v1/query -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"query":"Nguong chan doan dai thao duong theo WHO?","top_k":5}'
curl -X POST http://localhost:8000/v1/glucose -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"user_id":"<id>","value_mgdl":145,"context":"fasting"}'
```

Thêm SOAP và monitors xem Swagger `/docs` (không paste dài ở đây để file gọn).

## 11. Mã lỗi & Validation

- `422`: sai kiểu/miền — ví dụ `value_mgdl` ngoài 20..800, `top_k` ngoài 1..20, `days` ngoài 1..90. Cách sửa: đọc `detail` trả về, sửa field đó.
- `401`: thiếu/hết hạn token → login lại, gắn `Authorization: Bearer`.
- `403`: user thao tác nhầm `user_id` người khác, hoặc thiếu role (cần specialist/pharmacist/admin), hoặc expert chưa verify.
- `400`: thao tác không hợp lệ (review đã duyệt rồi...). `500`: lỗi RAG — đọc message, kiểm tra OpenAI key/Chroma.

### Checklist tự kiểm tra

- [ ] Login + gọi 3 API query/glucose/soap bằng Swagger thành công.
- [ ] Kể được khác biệt 401 vs 403 vs 422 bằng 1 câu mỗi loại.
- [ ] Chỉ ra file schema và router của 1 endpoint bất kỳ.
