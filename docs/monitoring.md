# Giám sát Guideline & An toàn thuốc — Kiến trúc Agent

> Triết lý: `agent tìm → tóm tắt thay đổi (VI) → gắn cờ "cần review" → chuyên gia duyệt → mới đưa vào corpus (kèm version/ngày)`.

## Tổng quan

Hai agent chung nền tảng `src/monitors/`:

- **Guideline Agent** — theo dõi ADA/AHA (hypertension), GOLD, GINA, mhGAP/WHO IRIS, Bộ Y tế (BYT). Kiểm tra **monthly HEAD** + **quarterly deep download** (vì guideline tự ghi "cập nhật hàng năm", monthly deep là lãng phí).
- **Safety Agent** — theo dõi FDA (openFDA drug/enforcement + label boxed warnings, anonymous + cache 1h, tuân thủ 240 req/min) và BYT/DAV (thuvienphapluat.vn + moh.gov.vn + dav.gov.vn). Phân tầng rủi ro: **daily 02:00** cho Class I / boxed warning (critical), **weekly Thứ 2 03:00** cho Class II / tương tác mới.

Tất cả phát hiện mới đều vào `data/raw/staging/<source>/<version>/` trước, **không tự động vào** `data/raw/pdfs/<source>/`. Chỉ sau khi duyệt mới promote và reindex.

## Luồng 5 bước (đúng yêu cầu)

```
1. Fetch (HEAD/etag hoặc FDA JSON)
   ↓ etag/last-modified/sha256 đổi ?
2. Download → staging
   ↓
3. Parse (docling / pypdf fallback) → Diff (so với bản approved gần nhất) → LLM tóm tắt VI
   ↓
4. Ghi DB status=pending_review → Notification in-app tới role phù hợp
   ↓
5. Expert decision (approved/rejected) → nếu approved: move staging→corpus, supersede bản cũ, reindex, version ghi nhận
```

- Guideline → `specialist` + `doctor` (cả hai đều được duyệt, specialist ưu tiên).
- Safety → `pharmacist` (theo `HILT_ROUTING` hiện có).

---

## Data Model — `metadata/monitoring.db`

### monitored_sources (10 dòng seed)

| source_key | display_name | interval | risk_tier |
|---|---|---|---|
| ada_soc | ADA SoC Diabetes | monthly | high |
| aha_acc_htn | AHA/ACC HTN | monthly | high |
| gold | GOLD COPD | monthly | high |
| gina | GINA Asthma | monthly | high |
| who_mhgap | WHO mhGAP | monthly | medium |
| who_diabetes | WHO Diabetes | monthly | high |
| byt_diabetes | BYT QĐ Đái tháo đường | weekly | high |
| fda_recall | FDA enforcement | daily | critical |
| dav_thuhoi | DAV thu hồi | weekly | critical |
| moh_canhbao | MOH cảnh báo | weekly | high |

Trường: `last_etag`, `last_modified`, `last_hash`, `last_checked_at`.

### guideline_versions

`id, source, title, url, version_label, publication_date, fetched_at, sha256, etag, staging_path, corpus_path, status ENUM(pending_review/approved/rejected/superseded/archived), change_summary_json, diff_text, reviewer_id, reviewed_at, review_notes, supersedes_id, indexed_at, created_at`

- `status=superseded` → sau 30 ngày (`MONITOR_SUPERSEDED_RETENTION_DAYS`) sẽ **xoá hẳn** file + DB row (theo yêu cầu của bạn), qua `delete_superseded_expired()` được scheduler gọi Chủ nhật 06:00 và API `POST /v1/monitors/maintenance/cleanup-superseded` (admin).

### safety_alerts

`id, source, alert_type (recall/interaction/box_warning/shortage/label_change), severity (critical/high/medium/low), drug_name, alert_title, alert_url, published_date, fetched_at, raw_json, ai_summary (JSON VI), ai_risk_score, status (pending_review/approved/dismissed/archived), assigned_role='pharmacist', reviewer_id, reviewed_at, review_notes, is_notified`

Dedup: trùng `alert_url` hoặc `drug_name+title` trong 7 ngày → trả về bản cũ, không tạo mới.

### monitor_runs

`id, source_key, started_at, finished_at, found_new, status (running/success/failed), error, trace_id` — cho `/v1/monitors/runs`.

---

## Module chi tiết `src/monitors/`

| File | Vai trò |
|---|---|
| `db.py` | SQLite `monitoring.db`, init, seed, CRUD cho 3 bảng + stats |
| `utils.py` | SSRF allowlist (`ALLOWED_MONITOR_DOMAINS`), `head_with_etag`, `download_file_safe` (PDF magic), `cached_get` (FDA cache 1h + 0.25s delay cho 240/min), `extract_text_from_pdf` |
| `summarizer.py` | LLM VI: `summarize_guideline_diff()` (prompt `guideline_diff`) + `summarize_safety_alert()` (prompt `safety_summary`), fallback heuristic nếu không có API key |
| `guideline_fetcher.py` | `GUIDELINE_SOURCE_CONFIGS` (7 nguồn), `check_guideline_update()` HEAD/etag, `_check_byt_scrape()` scrape thuvienphapluat.vn weekly, `_download_and_stage()` sha256 dedup, diff + summarize → `create_guideline_version` + notify |
| `safety_fetcher.py` | `fetch_fda_recalls()` (openFDA enforcement, anonymous), `fetch_fda_label_warnings()`, `check_fda_alerts()`, `fetch_byt_dav_alerts()` scrape DAV/MOH/thuvienphapluat, `check_all_safety()` |
| `service.py` | `notify_guideline_pending` (specialist/doctor), `notify_safety_pending` (pharmacist), `promote_guideline_to_corpus()` (move staging→corpus, supersede cũ, `reindex_single_pdf`, clear cache), `decide_safety_alert()`, `cleanup_superseded()` |
| `indexer_helper.py` | `reindex_single_pdf()` tái dùng `src/shared/indexer.py:hybrid_hash_reindex` cho 4 strategies, không quét toàn bộ corpus |
| `router.py` | FastAPI `/v1/monitors/*` (chi tiết dưới) |
| `scheduler.py` | `schedule` loop: FDA daily 02:00, BYT weekly Mon 03:00, guideline monthly HEAD (day=1 04:00), quarterly deep (Jan/Apr/Jul/Oct), cleanup Sun 06:00 |

### Prompts — `src/shared/prompt_templates.py` + `src/shared/prompt_manager.py`

Thêm 2 tone:

- `guideline_diff` (`GUIDELINE_DIFF_PROMPT`) → JSON VI `{tom_tat_tieng_viet, changed_sections[], dosage_changes[], new_recommendations[], removed_recommendations[], version_phat_hien, ngay_xuat_ban, muc_do_quan_trong}`
- `safety_summary` (`SAFETY_SUMMARY_PROMPT`) → JSON VI `{tieu_de_vi, tom_tat_vi, thuoc_lien_quan[], loai_canh_bao, muc_do, ly_do, khuyen_cao, nguon, diem_rui_ro}`

Seed tự động qua `PROMPT_REGISTRY` trong `prompt_manager.py:31`.

### Config — `src/shared/config.py`

Thêm:

```
MONITORING_DB_PATH, STAGING_DIR, FDA_API_BASE, FDA_CACHE_TTL_SECONDS=3600, FDA_RATE_LIMIT_PER_MIN=240,
MONITOR_SUPERSEDED_RETENTION_DAYS=30, ALLOWED_MONITOR_DOMAINS (10 domains)
```

---

## API — `src/monitors/router.py` (mount tại `src/api/main.py:105`)

| Method | Path | Role | Mô tả |
|---|---|---|---|
| GET | `/v1/monitors/status` | auth | Tổng quan stats + sources + recent runs |
| GET | `/v1/monitors/sources` | auth | List monitored_sources |
| GET | `/v1/monitors/runs?source_key=&limit=` | auth | Lịch sử runs |
| POST | `/v1/monitors/check/guidelines?source_key=&force=` | specialist/doctor/admin | Trigger 1 hoặc all guideline checks (force bỏ qua etag) |
| POST | `/v1/monitors/check/safety?source=fda\|byt\|all` | pharmacist/admin | Trigger safety checks |
| GET | `/v1/monitors/guidelines?status=&source=&limit=&offset=` | auth | List guideline_versions |
| GET | `/v1/monitors/guidelines/{gid}` | auth | Detail + diff + AI summary |
| POST | `/v1/monitors/guidelines/{gid}/decision` | specialist/doctor/admin | `{decision: approved\|rejected, notes}` → promote hoặc reject |
| GET | `/v1/monitors/alerts?status=&severity=&source=&limit=&offset=` | auth | List safety_alerts |
| GET | `/v1/monitors/alerts/{aid}` | auth | Detail |
| POST | `/v1/monitors/alerts/{aid}/decision` | pharmacist/admin | `{decision: approved\|dismissed, notes}` |
| POST | `/v1/monitors/maintenance/cleanup-superseded` | admin | Xoá superseded >30 ngày |
| GET | `/v1/monitors/stats` | auth | `{pending_guidelines, pending_alerts, total_*}` |

Tất cả endpoint đều yêu cầu `Authorization: Bearer <JWT>` hoặc cookie `access_token` (theo `src/auth/dependencies.py`).

---

## Scheduling

**Cách chạy (2 lựa chọn):**

1. **Sidecar process (khuyến nghị production):**
   ```bash
   python -m src.monitors.scheduler
   # loop 60s, jobs: FDA 02:00, BYT Mon 03:00, guideline monthly day1 04:00, quarterly deep, cleanup Sun 06:00
   ```

2. **Background thread trong FastAPI (dev):**
   ```python
   from src.monitors.scheduler import start_background_thread
   start_background_thread()  # daemon thread, gọi trong lifespan
   ```

**BYT scrape:** weekly (theo yêu cầu của bạn), vì không có API ổn định. Chỉ HEAD/etag không đủ nên luôn fetch HTML và hash để phát hiện thay đổi.

**FDA cache:** `cached_get` lưu memory 1h, delay 0.25s mỗi request để không vượt 240/min.

---

## Promote & Corpus Versioning

1. `promote_guideline_to_corpus(gid, reviewer_id)`:
   - Move `staging_path` → `PDF_DIR/<source>/filename.pdf`
   - Đánh `superseded` cho tất cả `approved` cũ cùng `source`
   - Gọi `indexer_helper.reindex_single_pdf()` cho 4 strategies (tái dùng `hybrid_hash_reindex`)
   - Update `indexed_at`, ghi `corpus_path`, clear `CAGHybridCache._exact` (nếu có)
   - Sau 30 ngày, `cleanup_superseded()` xoá file + row (đúng yêu cầu "xoá hẳn").

2. Citation: `src/chat/generator.py:format_context` sẽ hiển thị `[Source X | GOLD 2025 v1.2 | 2024-02-11]` nhờ metadata `version_label` + `publication_date` lưu trong `change_summary_json` (tương lai có thể embed vào Chroma `metas`).

---

## Bảo mật

- **SSRF allowlist** (`src/monitors/utils.py:is_url_allowed`) chỉ cho phép 10 domains trong `ALLOWED_MONITOR_DOMAINS`; kiểm tra trước mọi `HEAD`/`GET`/`download`. Đã có test `test_ssrf_allowlist`.
- **PDF magic** (`%PDF` 4 byte) từ chối HTML giả PDF.
- **Dedup** bằng `sha256` và `alert_url` tránh spam.
- **RBAC** chặt: guideline chỉ specialist/doctor/admin, safety chỉ pharmacist/admin.

---

## Vận hành & Troubleshooting

```bash
# Kiểm tra thủ công 1 nguồn (cần login)
curl -X POST http://localhost:8000/v1/monitors/check/guidelines?source_key=gold&force=true \
  -H "Authorization: Bearer $TOKEN"

# Duyệt guideline
curl -X POST http://localhost:8000/v1/monitors/guidelines/<gid>/decision \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"decision":"approved","notes":"Đã kiểm tra, đưa vào corpus"}'

# Duyệt safety (pharmacist)
curl -X POST http://localhost:8000/v1/monitors/alerts/<aid>/decision \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"decision":"approved","notes":"Đã xác nhận thu hồi"}'

# Xem staging
ls data/raw/staging/*/*

# Reindex thủ công nếu cần
python -m src.indexer

# Cleanup superseded >30d
curl -X POST http://localhost:8000/v1/monitors/maintenance/cleanup-superseded \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Logs: scheduler ghi `[scheduler]` prefix; monitor_runs lưu DB để audit.

---

## Testing

`tests/test_monitors.py` (10 tests, đã pass):

- SSRF allowlist, summarizer fallback VI, DB CRUD + dedup, API status/guideline trigger/decision, safety role check, promote staging→corpus, superseded cleanup.

Chạy:

```bash
pytest tests/test_monitors.py -v
pytest tests/test_monitors.py::test_api_guideline_trigger_and_decision -v
```
