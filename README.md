# Customer Support RAG Chatbot

**LLM-powered Customer Support Agent with Advanced RAG & Agentic System**

This project implements a **task-oriented customer support chatbot** using an **advanced Retrieval-Augmented Generation (RAG) pipeline** designed to be production-oriented, observable, and extensible. The system focuses on **high answer precision, low latency, and reduced hallucination** through multi-stage retrieval, reranking, caching, and structured generation.

---

## 🚀 Project Overview

* Built as a **research-to-production style project** aligned with real-world customer support use cases.
* Combines **hybrid retrieval (dense + sparse)**, **cache-augmented generation (CAG)**, and **agentic routing**.
* Designed to demonstrate best practices in **RAG architecture, context engineering, evaluation, and observability**.

---

## ✨ Key Features

### 🔹 Cache-Augmented Generation (CAG)

* Hybrid cache combining:

  * **Exact-match cache** for repeated or identical queries
  * **Semantic cache (FAISS / Chroma)** for near-duplicate queries
* Achieves **sub-600ms latency on cache hits**.

### 🔹 Adaptive Document Chunking

* Section-based chunking for structured documents
* Semantic chunking to preserve meaning across boundaries
* Reduces retrieval noise while maintaining context integrity

### 🔹 Multi-stage Retrieval Pipeline

1. **Hybrid Search**

   * Dense embeddings (Sentence-Transformers / domain-adapted models)
   * Sparse retrieval (BM25)

2. **Candidate Selection & Reranking**

   * Top-k retrieval
   * Cross-encoder reranking for fine-grained relevance
   * Diversity filtering to reduce redundancy

3. **Context Compression**

   * Removes irrelevant or overlapping content
   * Optimizes prompt context length before generation

### 🔹 Agentic Query Routing

* LLM-based **query rewriting and intent classification**
* Routes requests into:

  * RAG-based knowledge retrieval
  * Structured data queries (SQL, internal APIs)
  * Tool-calling workflows

### 🔹 Reliable Generation

* Enforced **structured outputs using JSON schemas**
* Deterministic formatting for downstream consumption
* Reduced runtime errors and hallucinations

---

## 🧠 Prompt Engineering Layer

Located in `src/prompt_templates.py`, this layer enables **controlled, role-based response generation** and system self-evaluation.

### Implemented Prompts

* **STRICT_SYSTEM_PROMPT**
  Factual, concise, and high-precision responses (suitable for regulated domains).

* **FRIENDLY_SYSTEM_PROMPT**
  Empathetic and user-friendly explanations for general customer support scenarios.

* **BALANCED_SYSTEM_PROMPT**
  Combines clarity and reasoning transparency with structured outputs.

* **EVALUATION_PROMPT**
  Enables an evaluation agent to score responses on:

  * Context relevance
  * Faithfulness
  * Precision
  * Fluency

This design prepares the system for **self-assessing and event-driven agent workflows**.

---

## 📊 RAG Evaluation & Observability

* Built evaluation pipelines using **LLM-as-a-Judge** to measure:

  * Context relevance
  * Answer faithfulness
  * Recall@k
* Added logging and tracing for:

  * Retrieval stages
  * Tool-calling decisions
  * End-to-end latency

Observed results:

* ~**20% improvement in answer precision** through hybrid retrieval and reranking
* Reduced hallucination via multi-stage context filtering

---

## 🧩 Context Engineering (LangChain-aligned)

This project applies **LangChain Context Engineering principles** to manage information flow effectively.

| Step                 | Goal                                      | Implementation                                |
| -------------------- | ----------------------------------------- | --------------------------------------------- |
| Define Context Types | Separate static, dynamic, persistent data | System prompts, retrieved chunks, FAISS cache |
| Write Context        | Persist intermediate reasoning & state    | Logs, agent states                            |
| Select Context       | Avoid irrelevant information              | Hybrid retrieval + metadata filtering         |
| Compress Context     | Handle long documents                     | Context summarization & pruning               |
| Isolate Context      | Prevent cross-task leakage                | Scoped agent prompts                          |
| Manage Lifecycle     | Control context evolution                 | Session-level memory                          |
| Evaluate & Tune      | Improve retrieval quality                 | Evaluation prompts                            |

---

## 🗂 Project Structure

 ```
 src/
 ├── app.py              # Streamlit demo app
 ├── cache.py            # Hybrid CAG (exact + semantic cache)
 ├── config.py           # Model and system configuration
 ├── generator.py        # LLM generation & output formatting
 ├── retriever.py        # Multi-stage retrieval logic
 ├── rag_pipeline.py     # End-to-end pipeline orchestration
 ├── prompt_templates.py# System & evaluation prompts (local fallback)
 ├── prompt_manager.py   # LangSmith Prompt Hub sync + runtime resolution
 ├── observability/      # LangSmith tracing, feedback, eval logging
 ├── models/             # Pydantic schemas for LLM I/O
 notebooks/               # Embedding & pipeline experiments
 data/                    # Raw and processed datasets
 requirements.txt
 ```

---

## ▶️ Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Set environment variables (`.env`):

```bash
# LLM provider: "openai" (default) or "groq"
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # becomes DEFAULT_MODEL
# GROQ_API_KEY=gsk_...            # only needed when PROVIDER=groq

# LangSmith observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=customer-support-rag
```

Run demo:

```bash
python -m src.rag_pipeline
# or
streamlit run src/app.py
```

---

## 🔍 LangSmith Tracing, Prompt Versioning & Observability

The project is fully instrumented with [LangSmith](https://smith.langchain.com) for tracing, prompt lifecycle management, and user feedback.

### Configuration

Environment variables (already supported via `.env`, legacy `LANGCHAIN_*` names also work):

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=customer-support-rag   # any project name
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

If the API key is missing or the Hub is unreachable, the app **keeps working** — prompts fall back to local constants in `src/prompt_templates.py` and tracing silently disables itself.

### Trace structure

Every `rag_chat()` call creates one root trace:

```
[chain] rag_chat                     # metadata: user_id, cache_hit, top_k,
├─ [retriever] retrieve_context      #            stage timings, latency, model
│   ├─ VectorStoreRetriever          # vector_hits / bm25_hits / merged_groups
│   └─ BM25Retriever
├─ [chain] rerank_contexts           # reranker model, candidate counts
└─ [chain] generate_answer           # tone, temperature, prompt_version hash
    └─ [llm] ChatOpenAI              # token usage + latency (auto-instrumented)
```

### Prompt versioning (Prompt Hub)

* One-time sync of all prompts to LangSmith Prompt Hub:

  ```bash
  cd src && python prompt_manager.py
  ```

  Creates `medical-support-strict|friendly|balanced|evaluation` repos.
* At runtime `prompt_manager.get_system_prompt(tone)` **pulls the latest committed version from the Hub** (cached for `PROMPT_HUB_CACHE_TTL_SECONDS`), falling back to local constants when offline.
* Edit a prompt in the LangSmith UI → new commit → running app picks it up within TTL. Every trace records the `prompt_version` hash of the exact prompt text used.

### User feedback

The Streamlit UI adds 👍/👎 buttons plus an optional comment under each answer; feedback is attached to the exact trace run (`client.create_feedback`) together with a deep link "View trace in LangSmith".

### Evaluation logging

`python src/evaluation.py` additionally pushes aggregate Recall@K / Hit@K / MRR results into LangSmith as a `retrieval_evaluation` run so trends appear on dashboards.

---

## 🎯 Purpose

This project is designed to demonstrate:

* Production-style **RAG system design**
* Practical **agentic workflows**
* Context engineering and evaluation best practices

It aligns directly with the **LLM-powered Customer Support Agent (Advanced RAG & Agentic System)** described in the CV and serves as a strong foundation for real-world customer support applications.

---

## 🤝 Contributing

* Open issues or pull requests for improvements
* Add unit tests under `src/`
* Follow clean architecture and reproducibility practices
