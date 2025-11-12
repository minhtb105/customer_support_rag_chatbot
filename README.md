# Customer Support RAG Chatbot

Lightweight Retrieval-Augmented Generation (RAG) project for building a customer-support chatbot using embeddings, semantic retrieval (FAISS/Chromadb), reranking, and a generator LLM. Includes experiments and notebooks for embeddings and pipeline demos.

## Features
- Hybrid Cache-Augmented Generation (CAG) with exact + semantic (FAISS) lookup
- Embedding experiments with Sentence-Transformers / BioBERT
- Cross-encoder reranking for improved context selection
- Pluggable retriever / generator components
- Notebook examples for data preprocessing and pipeline demo

## Quickstart (Windows)
1. Create and activate a virtual environment (recommended Python 3.10+)
   - PowerShell:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - CMD:
     ```
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set environment variables (example)
   ```
   set GROQ_API_KEY=your_api_key_here
   ```

4. Run Jupyter Lab for notebooks:
   ```
   jupyter lab
   ```
   Notebooks are under `notebooks/` (embedding_test.ipynb, rag_pipeline_demo.ipynb, data_preprocessing.ipynb).

5. Run simple pipeline demo (module mode):
   ```
   python -m src.rag_pipeline
   ```
   Or run the app (if used as a service):
   ```
   streamlit run src/app.py
   ```

## Project Layout
- src/
  - app.py — Streamlit app
  - cache.py — CAG hybrid cache implementation (exact + semantic/FAISS)
  - config.py — project constants and model names
  - generator.py — wraps LLM generation & answer formatting
  - retriever.py — context retrieval logic
  - rag_pipeline.py — high-level pipeline (cache → retrieve → rerank → generate)
  - models/ — Pydantic data models (LLM I/O, chunk, etc.)
- notebooks/ — experiments and demos
- data/ — raw and processed datasets
- requirements.txt — pinned dependencies

### 🧠 Prompt Engineering Layer (`src/prompt_templates.py`)

This layer defines structured system prompts that guide the chatbot's reasoning and response behavior.  
The design enables **multi-persona prompt control**, **response evaluation**, and **adaptive tone generation**.

**Implemented templates:**
- **STRICT_SYSTEM_PROMPT** — for factual, concise, and medically accurate responses.  
- **FRIENDLY_SYSTEM_PROMPT** — for empathetic, approachable explanations for general users.  
- **BALANCED_SYSTEM_PROMPT** — combines reasoning transparency with readability, including structured "thinking → final answer" outputs.  
- **EVALUATION_PROMPT** — allows a meta-evaluator agent to rate model responses on *Faithfulness, Recall, Precision,* and *Fluency.*

These templates enable consistent, role-based response control and prepare the system for **self-assessing, event-driven agentic document workflows**.

### 🧭 Why Context Engineering Matters
Efficient context management ensures that large language models operate with  
the *right information, at the right time, in the right format*.  
This table outlines how to apply **LangChain’s Context Engineering framework** to your RAG pipeline.

## 🧠 Context Engineering Checklist (for RAG & Agentic Systems)

This checklist summarizes key best practices from **LangChain Context Engineering** —  
helping your RAG or event-driven agent system manage, select, and evolve context effectively.

| 🧩 Step | 🎯 Goal | ⚙️ Implementation Tips | 🧠 Example in Your Project |
|:--|:--|:--|:--|
| **1. Define Context Types** | Identify what information is static, dynamic, or persistent | - Static: system prompts, global configs<br>- Dynamic: session state, retrieved chunks<br>- Persistent: FAISS store, user memory | Static → system role<br>Dynamic → retrieval results<br>Persistent → FAISS cache |
| **2. Write Context** | Record all reasoning and intermediate steps outside LLM window | - Save scratchpads, reasoning traces<br>- Persist memory/state per agent<br>- Log tool calls & results | LangGraph state + Kafka event logs |
| **3. Select Context** | Provide only relevant information to the model | - Use semantic & metadata filters<br>- Limit `top_k` results<br>- Apply similarity threshold fallback | Hybrid FAISS + exact cache retrieval |
| **4. Compress Context** | Handle long histories or large documents efficiently | - Summarize chat or chunks<br>- Cluster embeddings<br>- Prune noisy info | Summarizer agent for redundant contexts |
| **5. Isolate Context** | Prevent context clash between different agents/tasks | - Give each agent its own system prompt<br>- Pass context via events only when needed | Query Planner, Retriever, and Synthesizer agents use scoped contexts |
| **6. Manage Context Lifecycle** | Control how context evolves, resets, or persists | - Use lifecycle hooks<br>- Reset state between users/sessions<br>- Archive long-term memory | LangGraph memory nodes + FAISS store |
| **7. Evaluate & Tune Context** | Measure and improve context quality | - Score relevance, faithfulness, precision<br>- Use self-eval or Eval Agent prompts | EVALUATION_PROMPT rates retrieved answers |
| **8. Event-Driven Context Flow** | Make agents react dynamically to context updates | - Use Kafka or event bus<br>- Trigger re-indexing or query decomposition<br>- Chain agents by events | Document update → triggers Reindex Agent<br>User query → triggers Query Planner Agent |

---

> 💡 **Purpose:**  
> Apply this checklist to ensure your RAG / Agentic pipeline maintains **context accuracy, scalability, and adaptability** — aligning with LangChain’s *Context Engineering* principles.

## Contributing
- Create issues / PRs for bugs or improvements.
- Add unit tests under `src/` and run locally.
