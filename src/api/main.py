"""
FastAPI — WHO-RAG Infrastructure API (Hướng C) — composition root.

Endpoint logic lives in per-service routers (src/<service>/router.py);
this file only wires routers, middleware, lifespan and system endpoints.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from src.shared.config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, BASE_DIR
    from src.api.schemas import HealthResponse
    from src.api.deps import _count_pdfs
except ImportError:  # pragma: no cover
    from shared.config import API_TITLE, API_VERSION, API_PREFIX, EMBEDDING_PROVIDER, EMBEDDING_MODEL, BASE_DIR  # type: ignore
    from api.schemas import HealthResponse  # type: ignore
    from api.deps import _count_pdfs  # type: ignore

# Auth imports — lazy to avoid circular
try:
    from src.auth.router import router as auth_router
    from src.reviews.router import router as reviews_router
    AUTH_ENABLED = True
except ImportError:
    try:
        from auth.router import router as auth_router  # type: ignore
        from reviews.router import router as reviews_router  # type: ignore
        AUTH_ENABLED = True
    except Exception as e:
        AUTH_ENABLED = False
        auth_router = None  # type: ignore
        reviews_router = None  # type: ignore
        print(f"[auth] disabled: {e}")

# Admin imports
try:
    from src.admin.tracing_router import router as tracing_router
    from src.admin.prompt_router import router as prompt_router
    ADMIN_ENABLED = True
except ImportError:
    try:
        from admin.tracing_router import router as tracing_router  # type: ignore
        from admin.prompt_router import router as prompt_router  # type: ignore
        ADMIN_ENABLED = True
    except Exception as e:
        ADMIN_ENABLED = False
        tracing_router = None  # type: ignore
        prompt_router = None  # type: ignore
        print(f"[admin] disabled: {e}")

# Monitors imports
try:
    from src.monitors.router import router as monitors_router, guidelines_status_router
    MONITORS_ENABLED = True
except ImportError:
    try:
        from monitors.router import router as monitors_router, guidelines_status_router  # type: ignore
        MONITORS_ENABLED = True
    except Exception as e:
        MONITORS_ENABLED = False
        monitors_router = None  # type: ignore
        guidelines_status_router = None  # type: ignore
        print(f"[monitors] disabled: {e}")

# Chat service (RAG query + SSE stream)
try:
    from src.chat.router import router as chat_router
    CHAT_ENABLED = True
except Exception as e:
    CHAT_ENABLED = False
    chat_router = None  # type: ignore
    print(f"[chat] disabled: {e}")

# Diabetes service (glucose tracking + SOAP)
try:
    from src.diabetes.router import router as diabetes_router
    DIABETES_ENABLED = True
except Exception as e:
    DIABETES_ENABLED = False
    diabetes_router = None  # type: ignore
    print(f"[diabetes] disabled: {e}")

# Triage service (smart triage + scheduling + doctor queue)
try:
    from src.triage.router import router as triage_router
    TRIAGE_ENABLED = True
except Exception as e:
    TRIAGE_ENABLED = False
    triage_router = None  # type: ignore
    print(f"[triage] disabled: {e}")

# Labs service (lab-report Q&A)
try:
    from src.labs.router import router as labs_router
    LABS_ENABLED = True
except Exception as e:
    LABS_ENABLED = False
    labs_router = None  # type: ignore
    print(f"[labs] disabled: {e}")

# Vitals service (BP / respiratory / mood)
try:
    from src.vitals.router import router as vitals_router
    VITALS_ENABLED = True
except Exception as e:
    VITALS_ENABLED = False
    vitals_router = None  # type: ignore
    print(f"[vitals] disabled: {e}")

# Lifespan: dev thread for monitors (prod uses sidecar python -m src.monitors.scheduler)
try:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Dev: background thread for guideline/safety checks
        thread = None
        try:
            from src.monitors.scheduler import start_background_thread

            # Only start if not disabled via env
            if os.getenv("MONITOR_SCHEDULER_DISABLE", "false").lower() not in ("true", "1", "yes"):
                thread = start_background_thread()
                if thread:
                    print(f"[monitors] scheduler background thread started: {thread.name}")
        except Exception as e:
            print(f"[monitors] scheduler lifespan start failed: {e}")
        yield
        # No explicit stop needed (daemon thread)

except Exception as _lifespan_err:
    # Fallback without lifespan (e.g., import error)
    lifespan = None  # type: ignore
    print(f"[monitors] lifespan setup failed: {_lifespan_err}")

if lifespan is not None:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description="WHO-RAG Infrastructure API — Hướng C: lớp truy vấn y tế đáng tin cậy cho app bên thứ 3. (Auth + HILT + Local Tracing)",
        lifespan=lifespan,
    )
else:
    app = FastAPI(title=API_TITLE, version=API_VERSION, description="WHO-RAG Infrastructure API — Hướng C: lớp truy vấn y tế đáng tin cậy cho app bên thứ 3. (Auth + HILT + Local Tracing)")

# CORS — env-driven allowlist
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_URL", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- include routers ----------
if AUTH_ENABLED and auth_router is not None:
    app.include_router(auth_router)
    app.include_router(reviews_router)  # type: ignore
if ADMIN_ENABLED and tracing_router is not None:
    app.include_router(tracing_router)
    app.include_router(prompt_router)
if MONITORS_ENABLED and monitors_router is not None:
    app.include_router(monitors_router)
    app.include_router(guidelines_status_router)  # type: ignore
if CHAT_ENABLED and chat_router is not None:
    app.include_router(chat_router)
if DIABETES_ENABLED and diabetes_router is not None:
    app.include_router(diabetes_router)
if TRIAGE_ENABLED and triage_router is not None:
    app.include_router(triage_router)
if LABS_ENABLED and labs_router is not None:
    app.include_router(labs_router)
if VITALS_ENABLED and vitals_router is not None:
    app.include_router(vitals_router)

# ---------- health ----------
@app.get("/health", tags=["system"])
@app.get(f"{API_PREFIX}/health", tags=["system"])
def health():
    total, _ = _count_pdfs()
    legacy_ready = (BASE_DIR / "embeddings" / "pdf_db" / "structure").exists()
    openai_ready = (BASE_DIR / "embeddings" / "pdf_db_openai" / "structure").exists()
    if EMBEDDING_PROVIDER == "openai":
        vector_ready = openai_ready or legacy_ready
    else:
        vector_ready = legacy_ready
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
        pdf_count=total,
        vector_db_ready=vector_ready,
    ).model_dump()

# ---------- Root ----------
@app.get("/", tags=["system"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "query": f"POST {API_PREFIX}/query (auth required)",
        "query_stream": f"POST {API_PREFIX}/query/stream SSE (auth required) — events: metadata, token, done, error",
        "glucose": f"POST {API_PREFIX}/glucose (auth)",
        "bp": f"POST {API_PREFIX}/bp (auth)",
        "respiratory": f"POST {API_PREFIX}/respiratory (auth)",
        "mood": f"POST {API_PREFIX}/mood (auth)",
        "vitals": f"POST {API_PREFIX}/vitals (auth)",
        "soap": f"POST {API_PREFIX}/soap/generate (auth)",
        "auth": f"POST {API_PREFIX}/auth/login, /register, /me",
        "reviews": f"GET {API_PREFIX}/reviews (expert/admin), POST {API_PREFIX}/reviews/{{id}}/decide",
        "notifications": f"GET {API_PREFIX}/auth/notifications",
        "monitors": f"GET {API_PREFIX}/monitors/status (auth), POST {API_PREFIX}/monitors/check/*, GET {API_PREFIX}/monitors/guidelines, GET {API_PREFIX}/monitors/alerts",
    }
