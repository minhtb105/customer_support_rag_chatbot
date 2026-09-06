# Diabetes Assistant — WHO-RAG (Customer Support RAG Chatbot)

**LLM-powered Customer Support Agent with Advanced RAG & Agentic System — Specialized for Diabetes Care in Vietnam**

An advanced Retrieval-Augmented Generation (RAG) pipeline designed for production use — focused on high answer precision, low latency, and reduced hallucination via multi-stage retrieval, reranking, caching, and structured generation. Since 2025–2026 the system has been extended into a **diabetes assistant with 3 tracks (A/B/C)**, grounded in Vietnam market research, with **OpenAI embeddings (`text-embedding-3-small`)**, **FastAPI WHO-RAG Infrastructure API**, and **Next.js frontend**.

---

## 0. TL;DR — Why Diabetes in Vietnam Is a “Ripe Pain Point”

- **7.3% of adults (~7M people)** have diabetes, rising from 2.7% (2002) → 5.4% (2012) → 7.3% (2020); prediabetes 17.8%. Over **60% undiagnosed**, >50% of adults never had a glucose test. Over **55% already have complications** (cardiovascular 34%, eye/neuropathy 39.5%, kidney 24%). Forecast **+78% by 2045**.
- **Gap between follow-ups:** medication adherence 76.9–83.5% but **self-monitoring at home only 20.2%** (Thanh Nhan Hospital, type 2 + kidney complication, n=104) and periodic follow-up 46.2–63.1% — patients are “blind” for months (standard HbA1c every 3 months).
- **Willingness to pay proven:** FreeStyle Libre sensor (14 days, no finger prick) **2–4M VND/kit** sold at Pharmacity/Tiki; Jio Health raised **$5M (Series A, 2019) → $20M (Series B, 2022, Heritas Capital)**; Doctor Anywhere **$27M** entering Vietnam; telemedicine market **$242M (2023) → $374M (2029, CAGR ~7.5%, TechSci)**.
- **Tailwind:** National health insurance **reimburses telehealth from 1 July 2025**; only **14 doctors per 10,000 people** (target 15 by 2025; France 34, Australia 36) — AI triage is a real need. Major hospitals (Bach Mai, Vinmec, 108, Tam Anh, Duc Giang) already use AI for imaging and EMR.
- **Competitive gap:** DiaB, Medihome, eDoctor, FPT MediCare are crowded but lack an **auditable RAG layer over WHO/MOH guidelines with faithfulness scoring**.

---

## 1. Five-Criteria Framework for a “Ripe Pain Point”

| # | Criterion | Key Question |
|---|-----------|--------------|
| 1 | **Scale & Severity** | How many are affected? What is the cost if unsolved (money, health, time)? |
| 2 | **Frequency & Journey Position** | How often does it occur? Where in the user journey does the user actively seek a solution? |
| 3 | **Existing Payment Behavior** | Have users already paid for a workaround (even imperfect)? Stronger signal than “survey says will pay.” |
| 4 | **Tailwind** | What policy, insurance, technology, or trend accelerates the market? |
| 5 | **Competitive Gap** | Who built what? What gap can you fill with a technical advantage? |

> **How to use:** Score each criterion 0–5; prioritize #3 (real payment) and #4 (tailwind). 5/5 on #1 alone is insufficient without #3.

---

## 2. Market Evidence — Health: Diabetes in Vietnam

### 2.1 Scale & Severity

- National survey 2020: **7.3%** of adults have diabetes (~7M), up from 2.7% (2002) and 5.4% (2012). Prediabetes **17.8%** [^vtv] [^tuoitre] [^sytbn].
- Silent progression: **>60% undiagnosed**, >50% never tested (VTV, Nov 2025) [^vtv-undiagnosed]. Vietnam Endocrine-Diabetes Association: only **31% diagnosed**, and among them **29% controlled** (Prof. Tran Huu Dang) [^baodautu].
- Consequences: **>55% have complications** — cardiovascular 34%, eye/nerve 39.5%, kidney 24% [^baodautu]. Forecast **+78% by 2045**, faster than most regional peers [^sytbn].

→ **Criterion 1: very strong.** Not a rare disease; the price of uncontrolled disease is permanent complications.

### 2.2 Gap Between Follow-ups — Quantified

| Study | Medication Adherence | Self-Monitoring at Home | Periodic Follow-up |
|-------|----------------------|-------------------------|-------------------|
| **Thanh Nhan Hospital 2025** (type 2 + kidney complication, n=104) | 76.9% | **20.2%** | 46.2% |
| **Thong Nhat Hospital 2023** (n=255) | 83.5% | 63.1% *(combined “glucose control + follow-up”)* | 63.1% (combined) |
| **University of Public Health Clinic 2022** (n=240 records) | — | — | 53.75% |

*Sources:* Vietnam Journal of Medicine 2022 [^tapchi2022], 2025 [^tapchi2025]; VISTA 2023 [^vista2023]; standard HbA1c every 3 months [^daithaoduong].

**Insight:** Patients take meds regularly (76.9–83.5%) but are **blind to actual glucose trends** between visits — exactly the gap this project targets. A 3-month HbA1c cycle = months-long blind spot.

→ **Criterion 2: strong, quantified.** Cite this table directly in the “Problem” section.

### 2.3 Willingness to Pay — Real Evidence

- **Out-of-pocket:** FreeStyle Libre (14-day, no finger prick) **2–4M VND/kit** at Pharmacity, Tiki — a segment already buys monitoring tech out-of-pocket [^pharmacity] [^nhathuocHN].
- **VC investment:** Jio Health **$5M (2019 Series A) → $20M (2022 Series B, Heritas Capital)** [^vjst]; eDoctor backed by Facebook/Google; Doctor Anywhere **$27M** entering Vietnam [^baodautu-yte].
- **Market size:** telemedicine **$242.12M (2023) → $374.10M (2029, CAGR ~7.5%, TechSci)** [^medpro]; health IT **$4.4B by 2033 (CAGR ~13%)** [^vneconomy].

> ⚠️ **Honest caveat:** Current payments are tied to **hardware or consultations** (Jio Health, eDoctor); **no evidence yet of personal subscription payments for pure software**. This is a B2C risk — hence Track B is positioned as **B2B2C** (sell to clinics).

→ **Criterion 3: real but hardware-tied — B2B2C mitigates.**

### 2.4 Tailwind — Policy & Workforce

- **Telehealth reimbursed by national insurance from 1 July 2025** — a digital-health inflection that removes the “who pays” barrier for remote monitoring [^bhyt2025].
- **Doctor shortage:** **14 doctors per 10,000** (target 15 by 2025; France 34, Australia 36) [^tuoitre-bs] [^nld-bs] — doctor time is scarce → **AI triage** (only escalate when truly abnormal) has real value for doctors, not just patients.
- **AI accepted:** Major hospitals (Bach Mai, Vinmec, 108, Tam Anh) already use AI for imaging and EMR [^skds-ai].

→ **Criterion 4: clear tailwind since mid-2025.**

### 2.5 Competitive Gap

| Player | What They Built | Gap | Our Edge |
|--------|-----------------|-----|----------|
| **DiaB** (Docosan), **FPT MediCare**, eDoctor, Medihome | Tracking apps, appointment booking, consultations [^docosan] [^fptmedicare] | Lack **auditable RAG** over trustworthy WHO/MOH guidelines; no faithfulness metric | **WHO-RAG Infrastructure API** with citations + measurable faithfulness |
| **Devices** (FreeStyle Libre) | Accurate hardware | No plain-language trend explanation | **Track A:** RAG explains trends, only escalates when thresholds exceeded |
| **Clinics** | Paper records, manual summaries | Time-consuming pre-visit prep | **Track B:** standardized **SOAP/ADA** summaries, B2B2C |

→ **Criterion 5: gap is a trustworthy medical knowledge infrastructure layer.**

---

## 3. Three Solution Tracks (A/B/C)

### Track A — Self-Monitoring Adherence Assistant

**Directly targets the 20.2% figure.**
- Patients log or connect home glucose readings; chatbot uses **RAG over WHO/MOH** guidelines to explain trends in plain language and **only escalates to a doctor when thresholds are clearly exceeded** (reduces load on scarce doctors — §2.4).
- **Success metric:** *“% of trial users who maintain self-monitoring ≥3 times/week after 4 weeks”* (not just retrieval quality).
- **Tech:** `src/features/glucose_tracker.py` (SQLite `glucose_logs.db` + WHO/ADA thresholds) + `POST /v1/glucose` + RAG `diabetes` tone.

### Track B — Pre-Visit Summary Tool

**Targets the 46–53% follow-up adherence.**
- Summarizes self-monitoring data into a **standardized SOAP / ADA summary** sent to the doctor before the appointment — **sold to clinics/hospitals (B2B2C)** instead of direct-to-patient, leveraging telehealth reimbursement.
- **Tech:** `src/features/soap_summary.py` (LLM SOAP + rule-based fallback) + `POST /v1/soap/generate` → JSON/PDF/markdown.

### Track C — WHO-RAG Infrastructure API

**Because the consumer app market is crowded.**
- Positioned as a **trustworthy medical knowledge query layer (with audit trail and measurable faithfulness)** that existing apps (eDoctor, Medihome, etc.) can integrate, instead of building yet another competing app.
- **Tech:** `src/api/main.py` FastAPI — `POST /v1/query` returns `{answer, citations, prompt_version, faithfulness, trace_url}`.

**Overall Architecture:**

```
[Next.js Frontend] ──┐
[Streamlit (legacy)] ├──► [FastAPI /v1/*  — Track C] ─► [RAG Pipeline v2]
                     │         │                       ├─ retrieve_context (WHO diabetes corpus)
                     │         └─ audit logs           ├─ rerank + faithfulness scorer
[Track B — SOAP] ────┘                                 └─ generator (tone=diabetes strict)
[Track A — Glucose] ──► SQLite `glucose_logs` + thresholds engine
```

---

## 4. Data Sources & Pipeline

### 4.1 Existing Corpus

`data/raw/pdfs/` (18 WHO PDFs on physical activity/nutrition/sleep) + `data/raw/pdfs/diabetes/` *(new — auto-downloaded 3 PDFs)*:
- `WHO_Classification_Diabetes_2019.pdf` (990 KB) — IRIS 10665/325182
- `WHO_HEARTS_D_Diabetes_2020.pdf` (1.2 MB) — IRIS 10665/331710
- `WHO_PEN_2020_NCD.pdf` (2.5 MB) — IRIS 10665/334186 (contains diabetes chapter)
- `ADA_Standards_of_Care_2024_Abridged.pdf` — *optional, paywall 403; manual download instructions in `scripts/crawl_guidelines.py`*.

### 4.2 Guideline Crawler Pipeline

**Multi-source crawler:** `scripts/crawl_guidelines.py`

| Source | Folder | Method |
|--------|--------|--------|
| **WHO IRIS** | `data/raw/pdfs/who_iris/` | DSpace 7 REST: `discover/search/objects` → `items/{id}/bundles` → `bitstreams/{id}/content` (resolve handle `10665/xxx`) |
| **ADA** | `data/raw/pdfs/ada/` | Curated seeds + alt PDF (paywall → placeholder guide) |
| **AHA/ACC, GOLD, GINA** | `data/raw/pdfs/aha_acc|gold|gina/` | Curated direct PDF URLs (GOLD 2024, GINA 2024) |
| **MOH — Decisions/Ministry of Health** | `data/raw/pdfs/byt/` | Scrape `moh.gov.vn` + `thuvienphapluat.vn` (Decision 3319/QD-BYT and updates) — curated, manual supplement if needed |
| **mhGAP, Diabetes** | `data/raw/pdfs/mhgap|diabetes/` | WHO mhGAP v2.0 + Diabetes curated |

**Usage:**

```bash
# 1. Create manifest (curated seeds)
python scripts/crawl_guidelines.py crawl --all --limit 5
python scripts/crawl_guidelines.py crawl --source who_iris --query "diabetes" --limit 5

# 2. Download 3 public diabetes PDFs (WHO Classification + HEARTS-D + PEN)
python scripts/crawl_guidelines.py diabetes3
# or
python scripts/download_diabetes_pdfs.py --force

# 3. Download everything in manifest
python scripts/crawl_guidelines.py download

# 4. Check status
python scripts/crawl_guidelines.py status
```

**Recursive indexer:** `src/indexer.py` scans **recursively** `data/raw/pdfs/**/*.pdf` (supports subfolders), supports **OpenAI embeddings** (`text-embedding-3-small`, 1536d) with HuggingFace fallback (`all-MiniLM-L6-v2`, 384d). Separate DBs: `embeddings/pdf_db/` (local) vs `embeddings/pdf_db_openai/` (OpenAI) to avoid dimension mismatch.

```bash
# Index all PDFs (including diabetes subfolder)
python -m src.indexer
# or with OpenAI embeddings (requires OPENAI_API_KEY in .env)
EMBEDDING_PROVIDER=openai python -m src.indexer
```

### 4.3 Embedding Configuration

`.env`:
```bash
PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # 1536 dims
EMBEDDING_PROVIDER=openai   # "openai" | "local"
```

`src/config.py` auto-selects model and splits DB by provider. If the key or `langchain-openai` is missing, the system **falls back to local** without crashing.

---

## 5. Technical Documentation

### 5.1 Project Overview

* Built as a **research-to-production** project aligned with real-world customer support use cases (now specialized for diabetes).
* Combines **hybrid retrieval (dense + sparse)**, **cache-augmented generation (CAG)**, and **agentic routing**.
* Demonstrates best practices in **RAG architecture, context engineering, evaluation, and observability**.
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
| `DIABETES_STRICT_PROMPT` | **New:** WHO/ADA/MOH RAG with citations, thresholds |
| `SOAP_PROMPT` | **New:** Pre-visit SOAP generator |
| `WHO_RAG_AUDIT_PROMPT` | **New:** Infrastructure audit layer |
| `EVALUATION_PROMPT` | LLM-as-judge (faithfulness/precision/recall/fluency 0–5) |

Hub-first with local fallback: `prompt_manager.get_system_prompt(tone)` pulls the latest Hub version (TTL `PROMPT_HUB_CACHE_TTL_SECONDS`), falling back to local constants offline. New tones: `diabetes`, `soap`, `who_rag`.

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
├── e2e/                     # Playwright E2E (chromium)
│   ├── fixtures.ts
│   ├── overview.spec.ts
│   ├── tracker.spec.ts
│   ├── previsit.spec.ts
│   └── api-playground.spec.ts
tests/                       # Pytest suite (diabetes-focused)
├── conftest.py
├── test_glucose_tracker.py
├── test_soap_summary.py
├── test_api_diabetes.py
├── test_diabetes_retrieval.py
└── security/test_owasp_top10.py  # OWASP Top 10 2021
data/
├── raw/pdfs/                # WHO IRIS / ADA / MOH / diabetes subfolders
├── guideline_manifest.json  # Crawler manifest
└── evaluation/              # Benchmark datasets (diabetes_retrieval_*, legacy)
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
# or
pip install -e .[test]

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
# 1. Download diabetes PDFs
python scripts/crawl_guidelines.py diabetes3

# 2. Index (auto uses OpenAI embeddings if configured)
python -m src.indexer

# 3. Run FastAPI (Track C)
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/v1/health

# 4. Test RAG
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are WHO diagnostic thresholds for diabetes?","top_k":5,"user_id":"demo"}'

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

## 7. Testing

### Backend — Pytest (diabetes-focused)

```bash
pip install -e .[test]
pytest tests/ -v                          # all diabetes tests (unit + API + retrieval)
pytest tests/ -v -m "not legacy"          # exclude legacy generic tests
pytest tests/security -v -m security       # OWASP Top 10 only
pytest tests/test_diabetes_retrieval.py -v # 15 diabetes retrieval Qs
```

- **Generic non-diabetes tests removed:** Original `retrieval_evaluation.json` (20 Qs on physical activity, sodium, sleep, water/sanitation) was **pruned**. The diabetes-specific set is `data/evaluation/diabetes_retrieval_evaluation.json` (15 Qs) and `diabetes_benchmark_questions.json` (5 multi-hop diabetes Qs) referencing only `WHO_Classification_Diabetes_2019.pdf`, `WHO_HEARTS_D_Diabetes_2020.pdf`, `WHO_PEN_2020_NCD.pdf`, `ADA_SoC_2024`. Legacy files kept as `*legacy.json` with `pytest.mark.legacy` (skipped by default).
- **Coverage:** `test_glucose_tracker.py` (fasting/postprandial thresholds 126/200, hypo 70, critical 300, escalation 3×high), `test_soap_summary.py` (4-section SOAP, markdown, no-log edge), `test_api_diabetes.py` (RAG citation, glucose roundtrip, validation 422, SOAP, guidelines/health).

### Backend — Security (OWASP Top 10 2021)

```bash
pytest tests/security/test_owasp_top10.py -v
```

| OWASP | Test |
|-------|------|
| A01 Broken Access Control | `test_a01_user_isolation` — user_id isolation |
| A02 Cryptographic Failures | `test_a02_no_secrets_in_health` |
| A03 Injection | `test_a03_prompt_injection_blocked`, `test_a03_sql_injection_glucose_notes`, `test_a03_xss_notes_escaped` |
| A04 Insecure Design | `test_a04_escalation_design_critical / _three_high` |
| A05 Security Misconfiguration | `test_a05_cors_misconfiguration` (documents wildcard `allow_origins=["*"]` + `allow_credentials=True` → xfail until env allowlist) |
| A06 Vulnerable Components | `test_a06_no_high_vulns_in_lockfiles` |
| A07 Auth Failures | `test_a07_soap_requires_auth` (xfail placeholder, no JWT yet) |
| A08 Data Integrity | `test_a08_file_fingerprint_detects_tampering` |
| A09 Logging & Monitoring | `test_a09_logging_present` (langsmith/timings) |
| A10 SSRF | `test_a10_ssrf_crawler_rejects_private_ips` |

### Frontend — Playwright E2E

```bash
cd frontend
npm install  # includes @playwright/test
npx playwright install chromium
npm run test:e2e        # headless
npm run test:e2e:ui     # UI mode
```

- Config `frontend/playwright.config.ts` (webServer `npm run dev` on 3000, baseURL, trace on retry).
- Specs `frontend/e2e/overview.spec.ts` (hero stats 7.3%/20.2%, 5-criteria, 3 cards, nav), `tracker.spec.ts` (form → classification badge, chart ReferenceLines 126/200/70, KPI alert, escalation banner, RAG trend mock), `previsit.spec.ts` (SOAP 4 sections, markdown download), `api-playground.spec.ts` (audit trail, top_k slider, raw JSON, error handling).

---

## 8. LangSmith Tracing, Prompt Versioning & Observability

Every `rag_chat()` call creates one root trace with `retrieve_context`, `rerank_contexts`, `generate_answer` spans. New diabetes queries use `tone=diabetes` and `DIABETES_STRICT_PROMPT` (versioned in Hub as `medical-support-diabetes-strict`).

- One-time sync: `cd src && python prompt_manager.py` creates 7 repos (`strict|friendly|balanced|diabetes|soap|who_rag|evaluation`).
- At runtime `prompt_manager.get_system_prompt(tone)` pulls the latest Hub version (TTL `PROMPT_HUB_CACHE_TTL_SECONDS`), falls back to local constants offline.
- User feedback via 👍/👎 buttons attached to trace (`client.create_feedback`).

---

## 9. Purpose

This project demonstrates:
* Production-style **RAG system design** (hybrid retrieval + reranking + CAG)
* Practical **agentic workflows** + **3-track diabetes assistant**
* Context engineering & **auditable RAG** evaluation
* **WHO-RAG Infrastructure API** for B2B integration
* It aligns with **LLM-powered Customer Support Agent (Advanced RAG & Agentic System)** and serves as a foundation for real-world diabetes support.

---

## 10. Contributing

* Open issues/PRs
* Run tests before PR: `pytest tests/ -v && cd frontend && npm run test:e2e`
* Follow clean architecture & reproducibility

---

## 11. References

[^vtv]: VTV — Diabetes undiagnosed rate: https://suckhoe.vtv.vn/suc-khoe/ty-le-nguoi-mac-dai-thao-duong-chua-duoc-chan-doan-tai-viet-nam-hien-tai-la-hon-60-20241109143256325.htm
[^tuoitre]: Tuoi Tre — ~7M people with diabetes in Vietnam: https://tuoitre.vn/khoang-7-trieu-nguoi-viet-dang-mac-dai-thao-duong-20240701080303435.htm
[^sytbn]: Bac Ninh Health Dept — 3.5M → 6.3M by 2045: https://syt.bacninh.gov.vn/news/-/details/22511/hon-3-5-trieu-nguoi-viet-mac-ai-thao-uong-va-se-tang-len-6-3-trieu-vao-nam-2045
[^vtv-undiagnosed]: VTV — World Diabetes Day 2025: https://vtv.vn/ngay-the-gioi-phong-chong-dai-thao-duong-2025-chung-tay-hanh-dong-vi-suc-khoe-cong-dong-100251113233503187.htm
[^baodautu]: Dau Tu — Diabetes alert (Prof. Tran Huu Dang): https://baodautu.vn/bao-dong-ty-le-mac-benh-dai-thao-duong-o-nguoi-viet-d230000.html
[^tapchi2022]: Vietnam Journal of Medicine — Follow-up adherence, Univ. of Public Health Clinic 2022: https://tapchiyhocvietnam.vn/index.php/vmj/article/view/6421
[^tapchi2025]: Vietnam Journal of Medicine — Type 2 diabetes + kidney complication, Thanh Nhan Hospital 2025: https://tapchiyhocvietnam.vn/index.php/vmj/article/view/16281
[^vista2023]: STI/VISTA — Type 2 diabetes adherence, Thong Nhat Hospital 2023: https://sti.vista.gov.vn/publication/view/nghien-cuu-su-tuan-thu-dieu-tri-cua-nguoi-benh-dai-thao-duong-type-2-dieu-tri-ngoai-tru-tai-benh-vien-thong-nhat-nam-2023-dddd5a1294a2e26a9f9c922233e890a0-381685.html
[^daithaoduong]: Diabetes .com — Follow-up schedule: https://daithaoduong.com/lich-trinh-theo-doi-benh-tieu-duong/
[^pharmacity]: Pharmacity — FreeStyle Libre: https://www.pharmacity.vn/bo-cam-bien-may-do-duong-huyet-nhanh-freestyle-libre.html
[^nhathuocHN]: Hanoi Drug Pharmacy — FreeStyle Libre 2–4M VND: https://nhathuocduochanoi.com.vn/san-pham/may-do-duong-huyet-freestyle-libre-cham-soc-suc-khoe-benh-tieu-duong.html
[^vjst]: VJST — Jio Health $20M: https://vjst.vn/20-trieu-usd-duoc-rot-vao-start-up-viet-jio-health-21832.html
[^baodautu-yte]: Dau Tu — Health startup $27M (Doctor Anywhere): https://baodautu.vn/startup-y-te-hung-dong-von-khung-d120819.html
[^medpro]: Medpro — Telemedicine Vietnam (TechSci 2023): https://medpro.vn/tin-tuc/telemedicine-kham-benh-tu-xa-la-gi-va-no-hoat-dong-the-nao-
[^vneconomy]: VnEconomy — Online healthcare potential: https://vneconomy.vn/tiem-nang-tu-thi-truong-cham-soc-suc-khoe-truc-tuyen
[^bhyt2025]: National insurance reimburses telehealth from 1/7/2025 (press synthesis; verify via MOH portal)
[^tuoitre-bs]: Tuoi Tre — Target 15 doctors/10k: https://tuoitre.vn/nld/viet-nam-dat-muc-tieu-15-bac-si-van-dan-196240228205038154.htm
[^nld-bs]: Nguoi Lao Dong — 14 doctors/10k achieved: https://nld.com.vn/viet-nam-dat-14-bac-si-tren-10000-dan-196241224105553454.htm
[^skds-ai]: Suc Khoe & Doi Song — AI as doctor assistant: https://suckhoedoisong.vn/tri-tue-nhan-tao-ai-tro-thu-dac-luc-cua-bac-si-trong-kham-chua-benh-quan-ly-benh-an-dien-tu-169250720153923241.htm
[^docosan]: Docosan — DiaB app: https://www.docosan.com/blog/noi-tiet/ung-dung-diab/
[^fptmedicare]: FPT MediCare — Glucose tracking app: https://fptmedicare.vn/ung-dung-theo-doi-duong-huyet-thong-minh/
[^who-class]: WHO Classification of Diabetes Mellitus 2019 (IRIS 10665/325182): https://iris.who.int/handle/10665/325182
[^who-hearts]: WHO HEARTS-D Diagnosis & Management of Type 2 Diabetes 2020 (IRIS 10665/331710): https://iris.who.int/handle/10665/331710
[^who-pen]: WHO PEN 2020 (IRIS 10665/334186): https://iris.who.int/handle/10665/334186
