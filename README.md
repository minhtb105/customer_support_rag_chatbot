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

## Contributing
- Create issues / PRs for bugs or improvements.
- Add unit tests under `src/` and run locally.
