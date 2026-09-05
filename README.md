# Customer Support RAG Chatbot — Diabetes Assistant (WHO-RAG)

**LLM-powered Customer Support Agent with Advanced RAG & Agentic System — Nâng cấp cho Đái tháo đường tại Việt Nam**

> **Song ngữ / Bilingual:** Phần 0–4 trình bày song ngữ Việt–Anh (thị trường & giải pháp). Phần 5+ là tài liệu kỹ thuật (Technical Documentation) bằng tiếng Anh để giữ tính quốc tế cho CV/GitHub.
> **Bilingual note:** Sections 0–4 are bilingual VI–EN (market & solution). Sections 5+ remain in English for international readability.

This project implements a **task-oriented customer support chatbot** using an **advanced Retrieval-Augmented Generation (RAG) pipeline** designed to be production-oriented, observable, and extensible. The system focuses on **high answer precision, low latency, and reduced hallucination** through multi-stage retrieval, reranking, caching, and structured generation.

**Nâng cấp 2025–2026:** Hệ thống đã được mở rộng thành **trợ lý đái tháo đường** với 3 hướng giải pháp (A/B/C) dựa trên khảo sát thị trường Việt Nam, hỗ trợ **OpenAI embeddings (`text-embedding-3-small`)**, **FastAPI WHO-RAG Infrastructure API**, và **Next.js frontend**.

---

## 0. TL;DR — Vì sao Đái tháo đường tại Việt Nam là “Pain Point chín muồi” / Why Diabetes in Vietnam is a “Ripe Pain Point”

**🇻🇳 Tiếng Việt:**
- **7,3% người trưởng thành (~7 triệu người)** mắc đái tháo đường (ĐTĐ), tăng từ 2,7% (2002) → 5,4% (2012) → 7,3% (2020); tiền ĐTĐ 17,8%. Hơn **60% chưa được chẩn đoán**, >50% người trưởng thành chưa từng xét nghiệm đường huyết. Hơn **55% đã có biến chứng** (tim mạch 34%, mắt/thần kinh 39,5%, thận 24%). Dự báo **+78% đến 2045**.
- **Khoảng trống giữa 2 lần tái khám:** tuân thủ dùng thuốc 76,9–83,5% nhưng **tự theo dõi đường huyết tại nhà chỉ 20,2%** (BV Thanh Nhàn, n=104) và tái khám định kỳ 46,2–63,1% — bệnh nhân “mù thông tin” hàng tháng trời (chuẩn HbA1c mỗi 3 tháng).
- **Đã có hành vi chi trả:** cảm biến FreeStyle Libre 2–4 triệu/14 ngày bán rộng rãi; Jio Health gọi vốn 20M USD, Doctor Anywhere 27M USD; telemedicine 242M USD (2023) → 374M USD (2029, CAGR 7,5%).
- **Lực đẩy:** từ **1/7/2025 BHYT chi trả khám chữa bệnh từ xa**; chỉ **14 bác sĩ/10.000 dân** (mục tiêu 15, so với Pháp 34) — AI lọc case là nhu cầu thực.
- **Khoảng trống cạnh tranh:** DiaB, Medihome, eDoctor đông nhưng thiếu lớp RAG có audit trail trên guideline WHO/BYT đáng tin.

**🇬🇧 English:**
- **7.3% adults (~7M people)** have diabetes in Vietnam, rising from 2.7% (2002) → 5.4% (2012) → 7.3% (2020); prediabetes 17.8%. >**60% undiagnosed**, >50% adults never tested glucose. >**55% already have complications** (CV 34%, eye/nerve 39.5%, kidney 24%). Forecast **+78% by 2045**.
- **Gap between follow-ups:** medication adherence 76.9–83.5% but **self-monitoring only 20.2%** (Thanh Nhan Hosp., n=104) and follow-up 46.2–63.1% — patients are “blind” for months (HbA1c every 3 months).
- **Willingness to pay exists:** FreeStyle Libre 2–4M VND/14 days sold widely; Jio Health $20M, Doctor Anywhere $27M; telemedicine $242M (2023) → $374M (2029, CAGR 7.5%).
- **Tailwind:** **National health insurance reimburses telehealth from 1/7/2025**; only **14 doctors/10k** (target 15, vs France 34) — AI triage is real need.
- **Competition gap:** DiaB/Medihome/eDoctor exist but lack auditable RAG over WHO/MOH guidelines.

---

## 1. Khung đánh giá “Pain Point chín muồi” / 5-Criteria Framework for “Ripe Pain Point”

| # | Tiêu chí / Criterion | Câu hỏi kiểm tra / Key Question |
|---|----------------------|----------------------------------|
| 1 | **Quy mô & mức độ nghiêm trọng / Scale & Severity** | Bao nhiêu người ảnh hưởng? Hậu quả nếu không giải quyết (tiền, sức khỏe, thời gian)? / How many affected? Cost if unsolved? |
| 2 | **Tần suất & vị trí trong hành trình / Frequency & Journey** | Vấn đề xảy ra thường xuyên? Nằm ở bước nào mà người dùng chủ động tìm giải pháp? / How frequent? Where in user journey? |
| 3 | **Hành vi chi trả đã tồn tại / Existing Payment Behavior** | Người dùng đã trả tiền cho giải pháp nào (dù chưa hoàn hảo)? Tín hiệu mạnh hơn “khảo sát nói sẽ trả”. / Have users already paid for workaround? |
| 4 | **Lực đẩy bên ngoài / Tailwind** | Chính sách, bảo hiểm, công nghệ, xu hướng nào làm thị trường “chín” nhanh hơn? / Policy/tech trends accelerating? |
| 5 | **Khoảng trống cạnh tranh / Competitive Gap** | Ai đã làm? Họ bỏ sót phần nào mà bạn có lợi thế kỹ thuật? / Who built what? What gap can you fill technically? |

> **Cách dùng / How to use:** Chấm điểm mỗi tiêu chí 0–5, ưu tiên tiêu chí 3 (hành vi chi trả thật) và 4 (tailwind). Điểm 5/5 ở tiêu chí 1 không đủ nếu thiếu 3.

---

## 2. Bằng chứng thị trường — Sức khỏe: Đái tháo đường / Market Evidence — Health: Diabetes

### 2.1 Quy mô & mức độ nghiêm trọng / Scale & Severity

**🇻🇳 Tiếng Việt:**
- Điều tra toàn quốc 2020: **7,3%** người trưởng thành mắc ĐTĐ (~7 triệu), tăng mạnh từ 2,7% (2002) và 5,4% (2012). Tiền ĐTĐ **17,8%** [^vtv] [^tuoitre] [^sytbn].
- Bệnh tiến triển âm thầm: **>60% chưa được chẩn đoán**, >50% người trưởng thành chưa từng xét nghiệm đường huyết (VTV 11/2025) [^vtv-undiagnosed]. Hội Nội tiết – ĐTĐ VN: chỉ **31% được chẩn đoán**, trong đó **29% điều trị đạt yêu cầu** (GS.TS Trần Hữu Dàng) [^baodautu].
- Hậu quả: **>55% đã có biến chứng** — tim mạch 34%, mắt/thần kinh 39,5%, thận 24% [^baodautu]. Dự báo **+78% đến 2045**, nhanh hơn hầu hết nước khu vực [^sytbn].

**🇬🇧 English:**
- National survey 2020: **7.3%** adults have diabetes (~7M), up from 2.7% (2002) and 5.4% (2012). Prediabetes **17.8%** [^vtv] [^tuoitre] [^sytbn].
- Silent progression: **>60% undiagnosed**, >50% never tested glucose (VTV 11/2025) [^vtv-undiagnosed]. Vietnam Endocrine-Diabetes Association: only **31% diagnosed**, **29% of those controlled** (Prof. Tran Huu Dang) [^baodautu].
- Consequences: **>55% have complications** — CV 34%, eye/nerve 39.5%, kidney 24% [^baodautu]. Forecast **+78% by 2045**, faster than regional peers [^sytbn].

→ **Tiêu chí 1: rất mạnh / Criterion 1: very strong.** Không phải bệnh hiếm; giá của không kiểm soát là biến chứng vĩnh viễn.

### 2.2 Khoảng trống giữa các lần tái khám — có dẫn chứng định lượng / Gap Between Follow-ups — Quantified

| Nghiên cứu / Study | Tuân thủ dùng thuốc / Medication | Tự theo dõi tại nhà / Self-monitoring | Tái khám định kỳ / Follow-up |
|-------------------|----------------------------------|----------------------------------------|------------------------------|
| **BV Thanh Nhàn 2025** (ĐTĐ type 2 + biến chứng thận, n=104) | 76,9% | **20,2%** | 46,2% |
| **BV Thống Nhất 2023** (n=255) | 83,5% | 63,1% *(gộp “kiểm soát đường huyết + khám định kỳ”)* | 63,1% (gộp) |
| **PK ĐH Y tế Công cộng 2022** (n=240 hồ sơ) | — | — | 53,75% |

*Nguồn / Sources:* Tạp chí Y học Việt Nam 2022 [^tapchi2022], 2025 [^tapchi2025]; sti.vista.gov.vn 2023 [^vista2023]; chuẩn tái khám HbA1c mỗi 3 tháng [^daithaoduong].

**🇻🇳 Phân tích:** Uống thuốc đều (76,9–83,5%) nhưng **gần như “mù” đường huyết thực tế** giữa 2 lần khám — đúng khoảng trống dự án nhắm. Lịch HbA1c 3 tháng/lần = khoảng trống kéo dài hàng tháng.

**🇬🇧 Insight:** Patients take meds regularly (76.9–83.5%) but are **blind to glucose trends** between visits — the exact gap this project targets. HbA1c every 3 months = months-long blind spot.

→ **Tiêu chí 2: mạnh, có số định lượng / Criterion 2: strong, quantified.** Trích trực tiếp bảng này vào phần “Problem” của đồ án.

### 2.3 Bằng chứng thị trường đã sẵn sàng chi trả / Willingness to Pay — Real Evidence

**🇻🇳 Tiếng Việt:**
- **Tự trả tiền túi / Out-of-pocket:** FreeStyle Libre (14 ngày, không chích máu) **2–4 triệu/bộ**, bán tại Pharmacity, Tiki — nhóm bệnh nhân đã tự mua công nghệ theo dõi [^pharmacity] [^nhathuocHN].
- **Vốn mạo hiểm / VC:** Jio Health **$5M (2019 Series A) → $20M (2022 Series B, Heritas Capital)** [^vjst]; eDoctor từng nhận tài trợ Facebook/Google; Doctor Anywhere **$27M** khi vào VN [^baodautu-yte].
- **Quy mô thị trường / Market size:** telemedicine **$242,12M (2023) → $374,10M (2029, CAGR ~7,5%, TechSci)** [^medpro]; health IT **$4,4B vào 2033 (CAGR ~13%)**; VJST/VnEconomy ghi nhận tiềm năng chăm sóc sức khỏe trực tuyến [^vneconomy].

**🇬🇧 English:**
- **Out-of-pocket:** FreeStyle Libre (14-day, no finger prick) **2–4M VND/kit**, sold via Pharmacity, Tiki — patients already buy monitoring tech [^pharmacity] [^nhathuocHN].
- **VC:** Jio Health **$5M (2019) → $20M (2022, Heritas)** [^vjst]; eDoctor backed by Facebook/Google; Doctor Anywhere **$27M** entering Vietnam [^baodautu-yte].
- **Market size:** telemedicine **$242.12M (2023) → $374.10M (2029, CAGR ~7.5%, TechSci)** [^medpro]; health IT **$4.4B by 2033 (CAGR ~13%)** [^vneconomy].

> ⚠️ **Lưu ý quan trọng / Honest caveat:** Chi trả hiện gắn với **phần cứng/khám** (Jio, eDoctor), **chưa có bằng chứng người dùng cá nhân trả subscription cho app thuần phần mềm**. Đây là rủi ro mô hình B2C — lý do Hướng B chọn **B2B2C** (bán cho phòng khám).

→ **Tiêu chí 3: có bằng chứng thật, nhưng cần B2B2C / Criterion 3: real but hardware-tied — B2B2C mitigates.**

### 2.4 Lực đẩy chính sách & nhân lực / Tailwind — Policy & Workforce

- **BHYT chi trả telehealth từ 1/7/2025** — cú hích chuyển đổi số, giảm rào cản “ai trả tiền” cho theo dõi từ xa [^bhyt2025].
- **Thiếu bác sĩ:** **14 bác sĩ/10.000 dân** (mục tiêu 2025 là 15; Pháp 34, Úc 36) [^tuoitre-bs] [^nld-bs] — thời gian bác sĩ khan hiếm → **AI lọc case** (chỉ đẩy khi bất thường) có giá trị thật với cả **bác sĩ**, không chỉ bệnh nhân.
- **AI đã được chấp nhận:** Bạch Mai, Vinmec, 108, Tâm Anh, ĐK Đức Giang đã dùng AI chẩn đoán hình ảnh & bệnh án điện tử [^skds-ai].

→ **Tiêu chí 4: thuận lợi rõ rệt từ giữa 2025 / Criterion 4: clear tailwind since mid-2025.**

### 2.5 Khoảng trống cạnh tranh / Competitive Gap

| Đối thủ / Player | Đã làm / Done | Bỏ sót / Gap | Lợi thế của dự án / Our edge |
|------------------|---------------|--------------|------------------------------|
| **DiaB** (Docosan), **FPT MediCare**, eDoctor, Medihome | App theo dõi, đặt khám, tư vấn [^docosan] [^fptmedicare] | Thiếu **RAG có audit trail** trên WHO/BYT guideline đáng tin; thiếu đo faithfulness | **WHO-RAG Infrastructure API** có citation + faithfulness đo được |
| **Thiết bị** (FreeStyle Libre) | Phần cứng chính xác | Không diễn giải xu hướng bằng ngôn ngữ dễ hiểu | **Hướng A:** RAG diễn giải trend + chỉ escalate khi vượt ngưỡng |
| **Phòng khám** | Hồ sơ giấy, tóm tắt thủ công | Mất thời gian chuẩn bị trước tái khám | **Hướng B:** tóm tắt **SOAP/ADA** chuẩn hóa, bán B2B2C |

→ **Tiêu chí 5: khoảng trống là lớp “hạ tầng kiến thức y tế đáng tin” / Criterion 5: gap is trustworthy knowledge infrastructure.**

---

## 3. Ba hướng giải pháp / Three Solution Tracks (A/B/C)

### Hướng A — Trợ lý tuân thủ tự theo dõi đường huyết / Track A — Self-Monitoring Adherence Assistant

**Nhắm thẳng vào con số 20,2% / Targets the 20.2% directly.**
- Bệnh nhân nhập/kết nối chỉ số đường huyết tại nhà; chatbot dùng **RAG trên WHO/BYT** để diễn giải xu hướng bằng ngôn ngữ dễ hiểu, và **chỉ đẩy lên bác sĩ khi vượt ngưỡng rõ ràng** (giảm tải cho bác sĩ khan hiếm — §2.4).
- **Thước đo thành công / Success metric:** *“% người dùng thử nghiệm duy trì tự theo dõi ≥3 lần/tuần sau 4 tuần”* (thay vì chỉ đo chất lượng retrieval).
- **Kỹ thuật / Tech:** `src/features/glucose_tracker.py` (SQLite `glucose_logs.db` + ngưỡng WHO/ADA) + `POST /v1/glucose` + RAG `diabetes` tone.

### Hướng B — Công cụ “chuẩn bị hồ sơ trước tái khám” / Track B — Pre-Visit Summary Tool

**Nhắm vào 46–53% tuân thủ tái khám / Targets 46–53% follow-up adherence.**
- Tóm tắt dữ liệu tự theo dõi thành **bản chuẩn hóa SOAP/khuyến cáo ADA** gửi bác sĩ trước hẹn — **bán vào phòng khám/bệnh viện (B2B2C)** thay vì trực tiếp cho bệnh nhân, tận dụng BHYT chi trả telehealth.
- **Kỹ thuật / Tech:** `src/features/soap_summary.py` (LLM tạo SOAP + fallback rule-based) + `POST /v1/soap/generate` → JSON/PDF/markdown.

### Hướng C — Lớp “hạ tầng WHO-RAG” làm API / Track C — WHO-RAG Infrastructure API

**Vì thị trường consumer app đã đông / Consumer app market is crowded.**
- Định vị như **lớp truy vấn kiến thức y tế đáng tin cậy (có audit trail, có chỉ số faithfulness đo được)** mà các app hiện có (eDoctor, Medihome…) có thể tích hợp, thay vì tự xây thêm một app cạnh tranh.
- **Kỹ thuật / Tech:** `src/api/main.py` FastAPI — `POST /v1/query` trả về `{answer, citations, prompt_version, faithfulness, trace_url}`. Dùng cho mọi app bên thứ 3.

**Kiến trúc tổng thể / Architecture:**

```
[Next.js Frontend] ──┐
[Streamlit (legacy)] ├──► [FastAPI /v1/*  — Track C] ─► [RAG Pipeline v2]
                     │         │                       ├─ retrieve_context (WHO ĐTĐ corpus mới)
                     │         └─ audit logs           ├─ rerank + faithfulness scorer
[Track B — SOAP] ────┘                                 └─ generator (tone=diabetes strict)
[Track A — Glucose] ──► SQLite `glucose_logs` + thresholds engine
```

---

## 4. Nguồn dữ liệu & Data Pipeline / Data Sources & Pipeline

### 4.1 Corpus hiện có / Existing Corpus

`data/raw/pdfs/` (18 PDFs WHO về vận động/dinh dưỡng/ngủ) + `data/raw/pdfs/diabetes/` *(mới — 3 PDFs tải tự động)*:
- `WHO_Classification_Diabetes_2019.pdf` (990 KB) — IRIS 10665/325182
- `WHO_HEARTS_D_Diabetes_2020.pdf` (1.2 MB) — IRIS 10665/331710
- `WHO_PEN_2020_NCD.pdf` (2.5 MB) — IRIS 10665/334186 (có chương ĐTĐ)
- `ADA_Standards_of_Care_2024_Abridged.pdf` — *optional, paywall 403; hướng dẫn tải thủ công trong `scripts/crawl_guidelines.py`*.

### 4.2 Pipeline crawl guideline / Guideline Crawler Pipeline

**Crawler đa nguồn / Multi-source crawler:** `scripts/crawl_guidelines.py`

| Nguồn / Source | Thư mục / Folder | Phương pháp / Method |
|----------------|------------------|----------------------|
| **WHO IRIS** | `data/raw/pdfs/who_iris/` | DSpace 7 REST: `discover/search/objects` → `items/{id}/bundles` → `bitstreams/{id}/content` (resolve handle `10665/xxx`) |
| **ADA** | `data/raw/pdfs/ada/` | Curated seeds + alt PDF (paywall → placeholder hướng dẫn) |
| **AHA/ACC, GOLD, GINA** | `data/raw/pdfs/aha_acc|gold|gina/` | Curated direct PDF URLs (GOLD 2024, GINA 2024) |
| **BYT — Quyết định/Bộ Y tế** | `data/raw/pdfs/byt/` | Crawl `moh.gov.vn` + `thuvienphapluat.vn` (QĐ 3319/QĐ-BYT và cập nhật) — curated, bổ sung thủ công nếu cần |
| **mhGAP, Diabetes** | `data/raw/pdfs/mhgap|diabetes/` | WHO mhGAP v2.0 + Diabetes curated |

**Sử dụng / Usage:**

```bash
# 1. Tạo manifest (curated seeds)
python scripts/crawl_guidelines.py crawl --all --limit 5
python scripts/crawl_guidelines.py crawl --source who_iris --query "diabetes" --limit 5

# 2. Tải 3 PDF diabetes public (WHO Classification + HEARTS-D + PEN)
python scripts/crawl_guidelines.py diabetes3
# hoặc
python scripts/download_diabetes_pdfs.py --force

# 3. Tải toàn bộ theo manifest
python scripts/crawl_guidelines.py download

# 4. Kiểm tra trạng thái
python scripts/crawl_guidelines.py status
```

**Indexer đệ quy / Recursive indexer:** `src/indexer.py` quét **đệ quy** `data/raw/pdfs/**/*.pdf` (hỗ trợ subfolder), hỗ trợ **OpenAI embeddings** (`text-embedding-3-small`, 1536d) với fallback HuggingFace (`all-MiniLM-L6-v2`, 384d). DB tách riêng: `embeddings/pdf_db/` (local) vs `embeddings/pdf_db_openai/` (OpenAI) để tránh lệch dimension.

```bash
# Index tất cả PDFs (bao gồm diabetes subfolder)
python -m src.indexer
# hoặc với OpenAI embeddings (cần OPENAI_API_KEY trong .env)
EMBEDDING_PROVIDER=openai python -m src.indexer
```

### 4.3 Cấu hình embedding / Embedding Configuration

`.env`:
```bash
PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # 1536 dims
EMBEDDING_PROVIDER=openai   # "openai" | "local"
```

`src/config.py` tự động chọn model và tách DB theo provider. Nếu thiếu key hoặc `langchain-openai` chưa cài, hệ thống **fallback về local** mà không crash.

---

## 5. Technical Documentation (English)

### 5.1 Project Overview

* Built as a **research-to-production style project** aligned with real-world customer support use cases (now specialized for diabetes).
* Combines **hybrid retrieval (dense + sparse)**, **cache-augmented generation (CAG)**, and **agentic routing**.
* Designed to demonstrate best practices in **RAG architecture, context engineering, evaluation, and observability**.
* New: **3-track diabetes assistant** + **FastAPI** + **Next.js**.

### 5.2 Key Features

#### Cache-Augmented Generation (CAG)
* Hybrid cache: **Exact-match** + **Semantic (FAISS/Chroma)** for near-duplicates.
* **Sub-600ms latency on cache hits**.

#### Adaptive Document Chunking
* Section-based, semantic, sliding-window, hybrid — reduces retrieval noise.
* Located in `src/chunk_strategies.py`.

#### Multi-stage Retrieval Pipeline
1. **Hybrid Search:** Dense (OpenAI `text-embedding-3-small` or `all-MiniLM-L6-v2`) + Sparse (BM25)
2. **Reranking:** Cross-encoder `ms-marco-MiniLM-L-6-v2` + diversity filtering
3. **Context Compression:** Remove overlap, optimize prompt length

#### Agentic Query Routing
* LLM-based query rewriting & intent classification → RAG / SQL / tool-calling.

#### Reliable Generation
* Structured outputs via Pydantic, inline citations `[Source X]`, audit trail.

### 5.3 Prompt Engineering Layer

`src/prompt_templates.py` + `src/prompt_manager.py` (LangSmith Hub).

| Prompt | Use |
|--------|-----|
| `STRICT_SYSTEM_PROMPT` | Factual, concise (regulated domains) |
| `FRIENDLY_SYSTEM_PROMPT` | Empathetic, simple language |
| `BALANCED_SYSTEM_PROMPT` | Reasoning + structured |
| `DIABETES_STRICT_PROMPT` | **New:** WHO/ADA/BYT RAG with citations, VN language, thresholds |
| `SOAP_PROMPT` | **New:** Pre-visit SOAP generator |
| `WHO_RAG_AUDIT_PROMPT` | **New:** Infrastructure audit layer |
| `EVALUATION_PROMPT` | LLM-as-judge (faithfulness/precision/recall/fluency 0–5) |

Hub-first with local fallback: `prompt_manager.get_system_prompt(tone)` pulls latest Hub version (TTL `PROMPT_HUB_CACHE_TTL_SECONDS`), falls back to local constants offline. New tones: `diabetes`, `soap`, `who_rag`.

### 5.4 RAG Evaluation & Observability

* **Retrieval metrics:** `src/evaluation.py` — Recall@K, Hit@K, MRR
* **LLM-as-judge:** `scripts/run_benchmarks.py` B5 — Faithfulness, Precision, Recall, Fluency
* **Tracing:** LangSmith `rag_chat` root trace with `retrieve_context`, `rerank_contexts`, `generate_answer` spans; records `prompt_version`, token usage, latency.
* **Feedback:** 👍/👎 + trace URL in UI.

### 5.5 Context Engineering (LangChain-aligned)

| Step | Goal | Implementation |
|------|------|----------------|
| Define Context Types | Separate static/dynamic/persistent | System prompts, chunks, FAISS cache |
| Write Context | Persist reasoning & state | Logs, agent states |
| Select Context | Avoid irrelevant info | Hybrid retrieval + metadata filtering |
| Compress Context | Handle long docs | Summarization & pruning |
| Isolate Context | Prevent leakage | Scoped agent prompts |
| Manage Lifecycle | Control evolution | Session-level memory |
| Evaluate & Tune | Improve quality | Evaluation prompts |

### 5.6 Project Structure

```
src/
├── api/
│   ├── main.py              # FastAPI WHO-RAG API (Track C)
│   └── schemas.py           # Pydantic schemas
├── features/
│   ├── glucose_tracker.py   # Track A: glucose logs + thresholds
│   └── soap_summary.py      # Track B: SOAP generator
├── app.py                   # Streamlit demo (legacy)
├── cache.py                 # Hybrid CAG
├── config.py                # Models, thresholds, guideline sources
├── generator.py             # LLM generation + diabetes tone detection
├── retriever.py             # Hybrid retrieval (OpenAI/local embeddings)
├── rag_pipeline.py          # Orchestration + memory
├── indexer.py               # Recursive PDF indexing
├── prompt_templates.py      # Prompts incl. diabetes/SOAP
├── prompt_manager.py        # LangSmith Hub sync
├── observability/           # Tracing, feedback
├── models/                  # Pydantic schemas
├── memory/                  # Short/episodic/long-term
scripts/
├── crawl_guidelines.py      # Multi-source guideline crawler
└── download_diabetes_pdfs.py# Wrapper for 3 PDFs
frontend/                    # Next.js 14 (App Router)
├── app/
│   ├── page.tsx             # Overview (market evidence)
│   ├── tracker/page.tsx     # Track A
│   ├── previsit/page.tsx    # Track B
│   └── api-playground/page.tsx # Track C
data/
├── raw/pdfs/                # WHO IRIS / ADA / BYT / diabetes subfolders
├── guideline_manifest.json  # Crawler manifest
└── evaluation/              # Benchmark datasets
embeddings/
├── pdf_db/                  # Chroma (local 384d)
└── pdf_db_openai/           # Chroma (OpenAI 1536d)
metadata/
├── metadata_store.db        # File/chunk hashes
└── glucose_logs.db          # Glucose logs (Track A)
```

---

## 6. Quickstart

### Backend (FastAPI + RAG)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# hoặc
pip install -e .

# .env
PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=customer-support-rag
```

```bash
# 1. Tải diabetes PDFs
python scripts/crawl_guidelines.py diabetes3

# 2. Index (tự động dùng OpenAI embeddings nếu cấu hình)
python -m src.indexer

# 3. Chạy FastAPI (Track C)
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/v1/health

# 4. Test RAG
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Dấu hiệu đái tháo đường type 2?","top_k":5,"user_id":"demo"}'

# 5. Track A — log glucose
curl -X POST http://localhost:8000/v1/glucose \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo","value_mgdl":145,"context":"fasting"}'

# 6. Track B — SOAP
curl -X POST http://localhost:8000/v1/soap/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo","days":14}'
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev  # http://localhost:3000
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Streamlit (legacy)

```bash
streamlit run src/app.py
python -m src.rag_pipeline  # CLI demo
```

### LangSmith Prompt Sync

```bash
cd src && python prompt_manager.py
# Creates medical-support-strict|friendly|balanced|diabetes|soap|evaluation repos
```

---

## 7. LangSmith Tracing, Prompt Versioning & Observability

*(unchanged — see previous README for full details; added `diabetes`, `soap`, `who_rag` prompt repos)*

Every `rag_chat()` call creates one root trace with `retrieve_context`, `rerank_contexts`, `generate_answer` spans. New diabetes queries use `tone=diabetes` and `DIABETES_STRICT_PROMPT` (versioned in Hub as `medical-support-diabetes-strict`).

---

## 8. Purpose

This project demonstrates:
* Production-style **RAG system design** (hybrid retrieval + reranking + CAG)
* Practical **agentic workflows** + **3-track diabetes assistant**
* Context engineering & **auditable RAG** evaluation
* **WHO-RAG Infrastructure API** for B2B integration
* It aligns with **LLM-powered Customer Support Agent (Advanced RAG & Agentic System)** and serves as a foundation for real-world diabetes support.

---

## 9. Contributing

* Open issues/PRs
* Add unit tests under `src/` or `frontend/`
* Follow clean architecture & reproducibility

---

## 10. References / Nguồn tham khảo

[^vtv]: VTV — Tỷ lệ mắc ĐTĐ: https://suckhoe.vtv.vn/suc-khoe/ty-le-nguoi-mac-dai-thao-duong-chua-duoc-chan-doan-tai-viet-nam-hien-tai-la-hon-60-20241109143256325.htm
[^tuoitre]: Tuổi Trẻ — Khoảng 7 triệu người Việt mắc ĐTĐ: https://tuoitre.vn/khoang-7-trieu-nguoi-viet-dang-mac-dai-thao-duong-20240701080303435.htm
[^sytbn]: Sở Y tế Bắc Ninh — Hơn 3,5 triệu → 6,3 triệu vào 2045: https://syt.bacninh.gov.vn/news/-/details/22511/hon-3-5-trieu-nguoi-viet-mac-ai-thao-uong-va-se-tang-len-6-3-trieu-vao-nam-2045
[^vtv-undiagnosed]: VTV — Ngày Thế giới phòng chống ĐTĐ 2025: https://vtv.vn/ngay-the-gioi-phong-chong-dai-thao-duong-2025-chung-tay-hanh-dong-vi-suc-khoe-cong-dong-100251113233503187.htm
[^baodautu]: Báo Đầu Tư — Báo động tỷ lệ mắc ĐTĐ (GS Trần Hữu Dàng): https://baodautu.vn/bao-dong-ty-le-mac-benh-dai-thao-duong-o-nguoi-viet-d230000.html
[^tapchi2022]: Tạp chí Y học Việt Nam — Tuân thủ tái khám, PK ĐH YTCC 2022: https://tapchiyhocvietnam.vn/index.php/vmj/article/view/6421
[^tapchi2025]: Tạp chí Y học Việt Nam — Tuân thủ ĐTĐ type 2 + thận, BV Thanh Nhàn 2025: https://tapchiyhocvietnam.vn/index.php/vmj/article/view/16281
[^vista2023]: STI/VISTA — Tuân thủ ĐTĐ type 2, BV Thống Nhất 2023: https://sti.vista.gov.vn/publication/view/nghien-cuu-su-tuan-thu-dieu-tri-cua-nguoi-benh-dai-thao-duong-type-2-dieu-tri-ngoai-tru-tai-benh-vien-thong-nhat-nam-2023-dddd5a1294a2e26a9f9c922233e890a0-381685.html
[^daithaoduong]: Đái tháo đường .com — Lịch tái khám: https://daithaoduong.com/lich-trinh-theo-doi-benh-tieu-duong/
[^pharmacity]: Pharmacity — FreeStyle Libre: https://www.pharmacity.vn/bo-cam-bien-may-do-duong-huyet-nhanh-freestyle-libre.html
[^nhathuocHN]: Nhà thuốc Dược Hà Nội — Giá FreeStyle Libre 2–4 triệu: https://nhathuocduochanoi.com.vn/san-pham/may-do-duong-huyet-freestyle-libre-cham-soc-suc-khoe-benh-tieu-duong.html
[^vjst]: VJST — Jio Health $20M: https://vjst.vn/20-trieu-usd-duoc-rot-vao-start-up-viet-jio-health-21832.html
[^baodautu-yte]: Báo Đầu Tư — Startup y tế $27M (Doctor Anywhere): https://baodautu.vn/startup-y-te-hung-dong-von-khung-d120819.html
[^medpro]: Medpro — Telemedicine Việt Nam (TechSci 2023): https://medpro.vn/tin-tuc/telemedicine-kham-benh-tu-xa-la-gi-va-no-hoat-dong-the-nao-
[^vneconomy]: VnEconomy — Tiềm năng chăm sóc sức khỏe trực tuyến: https://vneconomy.vn/tiem-nang-tu-thi-truong-cham-soc-suc-khoe-truc-tuyen
[^bhyt2025]: BHYT chi trả telehealth từ 1/7/2025 (tổng hợp từ báo chí; kiểm chứng qua Cổng TTĐT BYT)
[^tuoitre-bs]: Tuổi Trẻ — Mục tiêu 15 bác sĩ/vạn dân: https://tuoitre.vn/nld/viet-nam-dat-muc-tieu-15-bac-si-van-dan-196240228205038154.htm
[^nld-bs]: Người Lao Động — Đạt 14 bác sĩ/10.000 dân: https://nld.com.vn/viet-nam-dat-14-bac-si-tren-10000-dan-196241224105553454.htm
[^skds-ai]: Sức khỏe & Đời sống — AI trợ thủ bác sĩ: https://suckhoedoisong.vn/tri-tue-nhan-tao-ai-tro-thu-dac-luc-cua-bac-si-trong-kham-chua-benh-quan-ly-benh-an-dien-tu-169250720153923241.htm
[^docosan]: Docosan — Ứng dụng DiaB: https://www.docosan.com/blog/noi-tiet/ung-dung-diab/
[^fptmedicare]: FPT MediCare — Ứng dụng theo dõi đường huyết: https://fptmedicare.vn/ung-dung-theo-doi-duong-huyet-thong-minh/
[^who-class]: WHO Classification of Diabetes Mellitus 2019 (IRIS 10665/325182): https://iris.who.int/handle/10665/325182
[^who-hearts]: WHO HEARTS-D Diagnosis & Management of Type 2 Diabetes 2020 (IRIS 10665/331710): https://iris.who.int/handle/10665/331710
[^who-pen]: WHO PEN 2020 (IRIS 10665/334186): https://iris.who.int/handle/10665/334186
