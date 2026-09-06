# Hợp Đồng Dữ Liệu — Data Contract

> Nguồn chân lý cho corpus, chunking, embedding, versioning và ingestion pipeline. Mọi số liệu đều trích từ code và DB thực tế (`python -c` + `sqlite3`), không ước lượng.

## 1. Tổng quan lưu trữ

| Thành phần | Đường dẫn | Kích thước / Số lượng (2026-09-06) | Ghi chú |
|---|---|---|---|
| Corpus gốc | `data/raw/pdfs/` | 18 file gốc + subfolders: `byt 1`, `diabetes 4`, `hypertension 5`, `mental 3`, `respiratory 3` = **34 PDFs** | Quét đệ quy `src/indexer.py:177 _iter_pdf_files`** |
| Staging (chờ duyệt) | `data/raw/staging/<source>/<version>/` | 0 bản chờ (đã promote xong) | Chỉ agent giám sát ghi, expert duyệt mới move |
| Đã xử lý | `data/processed/docling_chunks/` | rỗng (output trung gian docling) | Không dùng làm source RAG |
| Vector DB — local 384d | `embeddings/pdf_db/{structure,sliding,semantic,hybrid_section_semantic}/` | `structure 33.6 MB`, `sliding 37.7 MB`, `semantic 39.4 MB`, `hybrid 33.7 MB` | Mỗi chiến lược 6 files Chroma (`chroma.sqlite3`, `data_level0.bin`, `header.bin`, `link_lists.bin`…) |
| Vector DB — OpenAI 1536d | `embeddings/pdf_db_openai/structure/` | 1.7 MB, 5 files | Chỉ `structure` được index bằng `text-embedding-3-small`; tách để tránh dimension mismatch |
| Memory | `embeddings/memory_db*`, `memory_db_openai*` | 0.2–0.6 MB | Episodic memory |
| Metadata | `metadata/metadata_store.db` | **6.7 MB**, `files 75 rows`, `chunks 11428 rows` | WAL enabled, FK `chunks.file_name → files.file_name` |
| Vitals | `metadata/vitals.db` 32 KB, `glucose_logs.db` 24 KB (legacy) | Unified `disease_type` | |
| Auth | `metadata/auth.db` 110 KB | `users`, `review_requests`, `notifications`, `query_history` | |
| Tracing | `metadata/tracing.db` 114 KB | `traces`, `spans`, `span_chunks`, `prompts`, `ragas_evaluations`, `feedback` | |
| Monitoring | `metadata/monitoring.db` 73 KB | `monitored_sources 10 rows`, `guideline_versions`, `safety_alerts`, `monitor_runs` | |

*Kiểm chứng:*
```bash
Get-ChildItem -Recurse data\raw\pdfs -Filter *.pdf | Group-Object DirectoryName
# 18 + 1 + 4 + 5 + 3 + 3 = 34
python -c "import sqlite3; c=sqlite3.connect('metadata/metadata_store.db'); print(c.execute('select count(*) from files').fetchone()); print(c.execute('select count(*) from chunks').fetchone())"
# files 75 (= fname::strategy), chunks 11428
```

## 2. Hợp đồng thư mục guideline

`src/config.py:70 GUIDELINE_SOURCES = ["who_iris","ada","aha_acc","gold","gina","mhgap","byt"]`

`src/config.py:80 GUIDELINE_SOURCE_DIRS` (11 keys, giá trị là `Path`):

```python
who_iris    → data/raw/pdfs/who_iris
ada         → data/raw/pdfs/ada
aha_acc     → data/raw/pdfs/aha_acc
gold        → data/raw/pdfs/gold
gina        → data/raw/pdfs/gina
mhgap       → data/raw/pdfs/mhgap
byt         → data/raw/pdfs/byt
diabetes    → data/raw/pdfs/diabetes
hypertension→ data/raw/pdfs/hypertension
respiratory → data/raw/pdfs/respiratory
mental      → data/raw/pdfs/mental
```

Crawler `scripts/crawl_guidelines.py` tôn trọng mapping này; `src/indexer.py:177` quét `PDF_DIR.rglob("*.pdf")` nên subfolder mới tự động được index không cần đổi code.

## 3. Schema Metadata Store

`src/metadata_store.py:10 init_db`

```sql
-- WAL
PRAGMA journal_mode=WAL;

CREATE TABLE files (
  file_name TEXT PRIMARY KEY,  -- format: "{pdf_filename}::{strategy}" e.g. "GOLD_COPD_2024.pdf::structure"
  file_hash TEXT NOT NULL,     -- fingerprint sha256(size+mtime+head/tail 512KB)
  updated_at REAL NOT NULL     -- time.time()
);
CREATE TABLE chunks (
  chunk_hash TEXT PRIMARY KEY, -- sha256(normalized_text + section_path + page_numbers + chunk_index)
  file_name TEXT NOT NULL,     -- FK files.file_name, CASCADE
  chunk_index INTEGER,
  vector_id TEXT,              -- "{fname}_{chunk_id}_{hash[:12]}" dùng cho Chroma delete
  extra_meta TEXT              -- JSON: {source_id, chunking_strategy, section_path, page_numbers, ...}
);
CREATE INDEX idx_chunks_file ON chunks(file_name);
```

Hàm chính:
- `upsert_file_and_chunks(file_name, file_hash, chunks)` — `INSERT ... ON CONFLICT DO UPDATE` transaction
- `get_chunk_hashes_for_file(file_name)` — ordered `chunk_index`
- `find_vector_ids_for_chunk_hashes([hash])` — để `vector_db.delete(ids)`
- `delete_chunks_by_hashes([hash])`
- `get_file_hash(file_name)` — check skip

## 4. Chunking Contract

`src/chunk_strategies.py:37 ChunkingStrategy = structure | sentence | sliding | semantic | hybrid_section_semantic | auto`

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `MAX_TOKENS` | 480 | `config.py:27` |
| `CHUNK_OVERLAP` | 200 | `config.py:28` |
| `SLIDING_WINDOW_TOKENS` | 350 | `config.py:31` |
| `SLIDING_OVERLAP` | 64 | `config.py:32` |
| `SEMANTIC_SIM_THRESHOLD` | 0.9128 | `config.py:33` (cosine) |
| `ATOMIC_TOKEN_SIZE` | 120 | `config.py:34` |
| `SENTENCE_GROUP` | 4 | `config.py:35` |
| `TOKENIZER_MODEL` | `bert-base-uncased` | `config.py:50` |
| `HybridChunker` | `docling.chunking.HybridChunker` + `HuggingFaceTokenizer(max_tokens=480, merge_peers=True)` | `chunk_strategies.py:54` |

**Dispatcher** `chunk_document(doc, pdf_path, strategy, raw_text, embed_fn)`:
- `STRUCTURE` → `structure_chunk` (giữ `headings`/`page_numbers` từ `c.meta`)
- `SLIDING` → tokenize rồi cửa sổ 350 overlap 64
- `SEMANTIC` → atomic 120 tokens, embed, gộp nếu `cosine >= 0.9128`
- `HYBRID_SECTION_SEMANTIC` → nếu section `>480` tokens thì semantic con, giữ `section_path/page_numbers`

**Serialize**: `serialize_metatdata` nối list bằng `|||` để lưu Chroma `metadatas` (string only).

## 5. Embedding & Vector Store

`src/config.py:39 EMBEDDING_PROVIDER = openai | local`
- `openai` → `OpenAIEmbeddings(model="text-embedding-3-small")` 1536d
- `local` → `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` 384d
- `EMBEDDING_DIMENSIONS_MAP` 3-small 1536, 3-large 3072, ada-002 1536, MiniLM 384

`src/indexer.py:72 _make_embeddings()` thử OpenAI trước, fallback HF, log warning.

`src/indexer.py:81 _resolve_index_db_dir(strategy)`:
```python
if EMBEDDING_PROVIDER == "openai":  return BASE_DIR/"embeddings/pdf_db_openai"/strategy
else:                               return PDF_DB_DIR/strategy   # embeddings/pdf_db/<strategy>
```

`get_or_create_vectorstore(db_dir)` → `Chroma(persist_directory=db_dir, embedding_function=embeddings)`.

## 6. Fingerprinting & Hashing

`src/indexer.py:46 compute_chunk_hash(chunk)`:
```python
sha256( " ".join(text.split()) + section_path + page_numbers + chunk_index )
```

`src/indexer.py:57 compute_file_fingerprint(path, sample=262144)`:
```python
sha256( size + mtime_ns + head 256KB + tail 256KB )
```

Dùng cho `file_key = "{fname}::{strategy}"` — nếu `new_fp == old_fp` → skip toàn bộ file (`[SKIP]` log).

## 7. Incremental Reindex

`src/indexer.py:98 hybrid_hash_reindex(pdf_path, vector_db, strategy, embedding_adapter)`:

```
new_fp = fingerprint(pdf)
old_fp = get_file_hash(file_key)
if equal → return
doc = DocumentConverter(do_ocr=False).convert(pdf).document
chunks = chunk_document(...)
new_hashes = [hash(c) for c in chunks]
old_hashes = set(get_chunk_hashes_for_file(fname))
removed = old-old ∩ new, added = new - old
if removed: vector_db.delete(ids=find_vector_ids(removed)); delete_chunks_by_hashes(removed)
for added: vid="{fname}_{chunk_id}_{hash[:12]}", meta=serialize({source_id, chunking_strategy, section_path, page_numbers})
vector_db.add_texts(texts, metadatas, ids)
upsert_file_and_chunks(file_key, new_fp, rows)
```

`src/monitors/indexer_helper.py: reindex_single_pdf(pdf_path)` tái dùng logic này cho 4 strategies khi promote guideline từ staging.

## 8. Versioning & Promotion

- **Staging** `data/raw/staging/<source>/<version>/file.pdf` (ví dụ `staging/gold/20250211/GOLD_2024.pdf`) — chỉ `guideline_fetcher` ghi.
- **Corpus** `data/raw/pdfs/<source>/file.pdf` — chỉ `service.promote_guideline_to_corpus` move sau khi `specialist/doctor` approve.
- DB `guideline_versions.status`: `pending_review → approved → superseded → (deleted sau 30 ngày)`; `rejected/archived` giữ lại để audit.
- `monitor_runs` audit mỗi lần `check_all_guidelines` hoặc `check_all_safety`.
- **Retention:** `MONITOR_SUPERSEDED_RETENTION_DAYS=30` (`config.py:192`), job Chủ nhật 06:00 + `POST /v1/monitors/maintenance/cleanup-superseded` (admin).

## 9. Ingestion Pipeline (đầu-cuối)

```
scripts/crawl_guidelines.py
  CURATED_SEEDS[{source, title, url, direct_pdf}] 
  → resolve_iris_handle_to_content_url (DSpace 7: discover/search → bundles/ORIGINAL → bitstreams)
  → download_file (HEAD etag, %PDF magic, <1KB reject, tmp→replace, sha256[:16])
  → save_manifest(data/guideline_manifest.json) + sha256

src/indexer.py main()
  _iter_pdf_files(PDF_DIR.rglob) → hybrid_hash_reindex ×4 strategies

src/monitors/guideline_fetcher.py (agent)
  HEAD etag/last-modified → so monitored_sources.last_etag → if đổi → download staging → sha256 dedup vs approved → extract docling → summarize LLM VI → create_guideline_version pending → notify specialist/doctor

src/monitors/safety_fetcher.py
  FDA https://api.fda.gov/drug/enforcement.json?search=recall_initiation_date:[since TO 99991231] + label boxed_warning, cache 3600s, dedup alert_url+title 7d → safety_alerts pending → notify pharmacist

src/monitors/service.py promote
  move staging→corpus → supersede cũ → reindex_single_pdf → update indexed_at → clear CAGHybridCache._exact
```

## 10. Kiểm chứng nhanh

```bash
# Corpus
python -c "from src.config import GUIDELINE_SOURCE_DIRS; print(list(GUIDELINE_SOURCE_DIRS.keys()))"
#  ['who_iris','ada','aha_acc','gold','gina','mhgap','byt','diabetes','hypertension','respiratory','mental']

# Metadata
python -c "import sqlite3; c=sqlite3.connect('metadata/metadata_store.db'); print('files', c.execute('select count(*) from files').fetchone()[0]); print('chunks', c.execute('select count(*) from chunks').fetchone()[0])"
# files 75, chunks 11428

# Vector DB
ls embeddings/pdf_db/structure  # chroma.sqlite3 ~33 MB
ls embeddings/pdf_db_openai/structure  # 1.7 MB

# Monitoring
python -c "import sqlite3; c=sqlite3.connect('metadata/monitoring.db'); print(list(c.execute('select source_key,check_interval,risk_tier from monitored_sources')))"
# 10 rows: ada_soc monthly high ... fda_recall daily critical
```

## 11. Ràng buộc & Lỗi thường gặp

- Không trộn `pdf_db` và `pdf_db_openai` — dimension mismatch 384 vs 1536 sẽ lỗi Chroma query.
- `file_name` trong `files` là `fname::strategy`, không phải `fname` đơn — `get_chunk_hashes_for_file` dùng `fname` gốc để tương thích cũ, nhưng `upsert` dùng `file_key`.
- `serialize_metatdata` nối list bằng `|||` — khi đọc phải split lại, không dùng JSON array trực tiếp trong Chroma.
- `compute_file_fingerprint` phụ thuộc `mtime_ns` — nếu copy file giữ mtime, fingerprint có thể trùng dù nội dung đổi → sha head/tail 256KB bù lại.
