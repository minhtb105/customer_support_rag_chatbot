# AGENTS.md — Diabetes Companion (customer_support_rag_chatbot)

## Quick Start (copy-paste)

```bash
pip install -e .              # openai-only base; local embeddings: pip install .[local]
uvicorn src.api.main:app --reload --port 8000   # backend, http://localhost:8000
cd frontend && npm install && npm run dev       # frontend, http://localhost:3000
pytest -q                                       # backend tests (testpaths=tests, pythonpath=src+.)
npx tsc --noEmit                                # frontend typecheck (run in frontend/)
```

Env (gitignored, never commit): `.env` needs `OPENAI_API_KEY`; `frontend/.env.local` needs `NEXT_PUBLIC_API_URL=http://localhost:8000`.

## Architecture

- `src/api/main.py` — FastAPI composition root (routers mounted here).
- Domain packages: `src/{chat,diabetes,labs,learn,monitors,reviews,scheduling,triage,vitals,voice,auth,admin}/` — each owns `router.py` + store/tracker logic.
- `src/shared/` — shared kernel. Reuse before writing: `text.py:unaccent`, `memory/_extract.py` (single entity extractor), `prompt_templates.build_strict()`, `monitors` use `run_job()`/`stage_version()`.
- `scripts/` — one-shot tooling. Shared helpers: `nutrition/_fetch.py` (download), `nutrition/_parse_io.py`, `nutrition/_ingest.py --src md|json`, `reembed.py --provider local|nvidia`. Keep CLIs working when editing.
- `frontend/app/` — Next.js routes (`tracker`, `previsit`, `monitor/*`, `admin/*`, `expert/*`); shared UI in `frontend/components/` (`AdminTable`, `ChatBox`, `GlucoseChart`, `MonitorList`); API wrappers in `frontend/lib/api.ts` via `apiGet`/`apiPost`.
- `tests/` — pytest; shared fixtures in `tests/_helpers.py` (extend it, don't copy-paste cases).

## Test-DB isolation

- `tests/conftest.py:client` (session fixture) redirects all 5 DBs (glucose/auth/vitals/monitoring/tracing) to temp files — tests never touch dev DBs. Always write API tests against this `client` fixture.
- Why module-namespace patching: trackers do `from config import X_DB_PATH` at import time, so setting `cfg.X_DB_PATH` alone is a no-op. Patch the tracker's own namespace (`bp_tracker.VITALS_DB_PATH`, `glucose_tracker.GLUCOSE_DB_PATH`, …) **before** importing the app — see `conftest.py` lines 39-60.
- Auth helpers: `make_user`, `auth_header`, `doctor_user`, `admin_client` — reuse them instead of hand-rolling register/login.
- Re-init pattern: after redirecting a DB path, call the tracker's `init_*_db()` (e.g. `init_bp_db()`, `init_glucose_db()`).

## Benchmark workflow

```bash
python scripts/run_benchmarks.py [--skip-judge] [--latency-n 10]   # B1-B5 RAG over data/evaluation/
BENCH_STRATEGIES=hybrid_section_semantic BENCH_WINNER=<name> python scripts/run_benchmarks.py
```
- B1 chunking (Recall@5/Hit@5/MRR) → B2 retriever (bm25/vector/hybrid) → B3 rerank → B4 latency P50/P95 → B5 LLM-judge (faithfulness/precision/recall/fluency 0-5). Results land in `data/evaluation/results_benchmark_<ts>.json`.
- B6-B8 (triage/labs/solvers, pure functions, 0 LLM calls) are opt-in via `--services`; gold sets live in `data/evaluation/`. B5 judge follows `LLM_PROVIDER` (`openai` default; `nvidia` uses `NVIDIA_JUDGE_MODEL`); `--skip-judge` for offline runs (`HF_HUB_OFFLINE=1` is set by the script).

## Conventions

- Refactor with aliases: keep old public names (functions, classes, routes, schemas) importable as thin wrappers. Deletion over addition otherwise.
- Conventional commits: `feat:`, `fix:`, `chore:`, `refactor:` (one-line subject).
- `WORKFLOW_STATE.md` is agent scratch — never commit it.
- No new deps for what stdlib/httpx/already-installed covers. `requirements.txt` mirrors `pyproject.toml` base.

## Gotchas

- Shell is Windows PowerShell 5.1: no `&&` (use `; if ($?) { ... }`), no `head`/`tail` (use `Select-Object -First/-Last`), quote paths with spaces.
- Imports use dual `try: from src.… except: from …` fallbacks — preserve them; keep `PYTHONPATH` including `src`.
- `scripts/reembed_local.py`, `reembed_nvidia.py`, `seed_admin.py` are thin shims — edit the real targets (`reembed.py`, `seed_demo_users.py --force`).
- e2e (`frontend/e2e/`) needs a live backend + seeds; `demo_recorder` is manual-only, not CI.
- Secrets: no real keys on disk or in prompts. `.env` files stay gitignored; README shows only an `sk-proj-...` placeholder.
