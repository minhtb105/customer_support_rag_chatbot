# Hợp Đồng Dữ Liệu — Data Contract

> Mọi số liệu trích từ code + DB thật, không ước lượng.

> **Mục tiêu đọc-xong-làm-được:** kể được 4 nguồn để làm gì, liệt kê corpus ĐTĐ, chạy index 1 file, phân biệt embedding OpenAI vs local.

## 1. Tổng quan lưu trữ

| Thành phần | Đường dẫn | Số lượng (2026-09-06) |
|---|---|---|
| Corpus gốc | `data/raw/pdfs/` | 18 file gốc + subfolders byt/diabetes/hypertension/mental/respiratory |
| GHO snapshot | `data/raw/gho/<INDICATOR>/<stamp>.json` | 4 chỉ số VN, fetch quarterly |
| Vector local 384d | `embeddings/pdf_db/{structure,sliding,semantic,hybrid}/` | mỗi chiến lược ~33-39MB |
| Vector OpenAI 1536d | `embeddings/pdf_db_openai/structure/` | 1.7MB, chỉ structure |
| Metadata | `metadata/metadata_store.db` | files 75 rows, chunks 11428 rows |
| Vitals/Auth/Tracing/Monitoring | `metadata/{vitals,auth,tracing,monitoring}.db` | nhỏ, mỗi DB một nhiệm vụ |

## 2. Nguồn để làm gì (purpose-4-rows, plain-VI)

| Nguồn | Để làm gì (1 câu) |
|---|---|
| WHO IRIS (who_iris) | Guideline quốc tế (HEARTS-D, PEN, phân loại ĐTĐ) — đáp án nền trung lập |
| ADA Standards of Care 2024 (ada) | Chuẩn thực hành ĐTĐ Mỹ mới nhất — đối chiếu mục tiêu HbA1c<7% |
| BYT Việt Nam (byt) | QĐ3319 + QĐ5481 (ĐTĐ type 2 VN) để trả lời đúng tuyến VN; **QĐ3192 là tăng huyết áp, không dùng cho ĐTĐ** |
| GHO snapshot (WHO data) | Số liệu thống kê (tỷ lệ, điều trị, VN) — trả lời câu hỏi "bao nhiêu %" |

## 3. Corpus chi tiết (condensed)

- **WHO:** 3 PDFs lõi (Classification 2019, HEARTS-D 2020, PEN 2020) + các file `who_iris` khác.
- **ADA:** Standards of Care 2024 trong `data/raw/pdfs/ada/`.
- **BYT:** `byt/` có QĐ3319 + QĐ5481 (ĐTĐ) và QĐ3192 (**THA — note đỏ: không dùng cho ĐTĐ**, A chỉ dùng 5481+ADA).
- **Bệnh khác:** `hypertension 5`, `respiratory 3`, `mental 3` file để demo đa bệnh.
- Mapping thư mục ở `src/config.py:70 GUIDELINE_SOURCES` (11 keys); crawler tôn trọng mapping, indexer quét đệ quy nên thêm subfolder là tự index.

## 4. Chunking Contract

`src/chunk_strategies.py:37`: 6 strategy, hay dùng 4 (`structure/sliding/semantic/hybrid_section_semantic`). Tham số nhớ: `MAX_TOKENS 480`, overlap 200, sliding 350/64, semantic gộp khi cosine ≥0.9128, tokenizer `bert-base-uncased`. Structure giữ headings/số trang (tốt cho trích dẫn), hybrid giữ section + semantic con khi section dài.

## 5. Embedding & Vector Store

1 đoạn duy nhất: có 2 nhà cung cấp — **OpenAI** (`text-embedding-3-small`, 1536 chiều, cần API key, chính xác hơn) vs **local** (`all-MiniLM-L6-v2`, 384 chiều, chạy offline). `src/indexer.py:72` thử OpenAI trước, rớt thì fallback local. **Tuyệt đối không trộn 2 DB** (384 vs 1536 sẽ lỗi query Chroma) — vì vậy tách `pdf_db/` và `pdf_db_openai/`.

## 6. Fingerprinting & Hashing

Mỗi chunk có hash = sha256(nội dung chuẩn hóa + section + trang + index); mỗi file có fingerprint = sha256(size + mtime + head/tail 256KB). Dùng để skip file không đổi và xóa chunk cũ khi reindex.

## 7. Incremental Reindex

Luồng `hybrid_hash_reindex` (`src/indexer.py:98`): tính fingerprint mới → so cũ → trùng thì skip; khác thì parse docling → chunk → so hash cũ/mới → xóa removed, thêm added vào Chroma → upsert metadata. Promote guideline từ staging tái dùng đúng hàm này (`reindex_single_pdf`).

<details>
<summary>Chi tiết schema SQL + ingestion pipeline (click để mở)</summary>

- `files(file_name PK dạng fname::strategy, file_hash, updated_at)`, `chunks(chunk_hash PK, file_name FK, chunk_index, vector_id, extra_meta JSON)`, WAL on.
- Pipeline: `crawl_guidelines.py` (download + manifest) → `indexer.main` (×4 strategies) → `guideline_fetcher` (staging + pending_review) → `safety_fetcher` (FDA alerts) → `service.promote` (move + reindex + clear cache) → `fetch_gho_snapshot.py` (quarterly, bypass review, audit monitor_runs).

</details>

## 8. Versioning & Promotion

Staging `data/raw/staging/<source>/<version>/` chỉ agent ghi; corpus `data/raw/pdfs/<source>/` chỉ move sang sau khi specialist/doctor approve. Trạng thái `pending_review → approved → superseded` (xóa sau 30 ngày). Mỗi lần check đều audit vào `monitor_runs`.

## 9. Ingestion Pipeline (đầu-cuối, 3 bước để nhớ)

1. **Crawl:** tải PDF về staging, kiểm `%PDF`, dedup sha.
2. **Index:** `hybrid_hash_reindex` ×4 strategies vào Chroma + metadata.
3. **Promote/audit:** duyệt → move → reindex → clear cache; GHO đi đường riêng quarterly.

## 10. Kiểm chứng nhanh

```bash
python -c "from src.config import GUIDELINE_SOURCE_DIRS; print(list(GUIDELINE_SOURCE_DIRS.keys()))"
python -c "import sqlite3; c=sqlite3.connect('metadata/metadata_store.db'); print(c.execute('select count(*) from files').fetchone(), c.execute('select count(*) from chunks').fetchone())"
ls embeddings/pdf_db/structure
```

Kết quả mong đợi: 11 keys, files 75 / chunks 11428, thư mục Chroma ~33MB.

## 11. Ràng buộc & Lỗi thường gặp

- Không trộn `pdf_db` và `pdf_db_openai` (lệch chiều vector).
- `file_name` trong DB là `fname::strategy`, không phải tên file đơn.
- List trong Chroma metas nối bằng `|||` — đọc phải split lại.
- Copy file giữ mtime có thể khiến fingerprint trùng giả — head/tail 256KB bù lại nhưng vẫn nên reindex force khi nghi.

### Checklist tự kiểm tra

- [ ] Kể được 4 nguồn để làm gì + QĐ nào không dùng cho ĐTĐ.
- [ ] Phân biệt OpenAI vs local embedding bằng 1 câu.
- [ ] Chạy được lệnh kiểm chứng và đọc đúng số files/chunks.
