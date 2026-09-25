"""
Local prompt versioning — thay Hub, lưu trong tracing.db
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    from src.shared.observability.tracing_db import get_active_prompt, create_prompt, approve_prompt, list_prompts
    from src.shared.observability.local_tracing import short_hash
except ImportError:
    from shared.observability.tracing_db import get_active_prompt, create_prompt, approve_prompt, list_prompts  # type: ignore
    from shared.observability.local_tracing import short_hash  # type: ignore

try:
    from shared.prompt_templates import (
        BALANCED_SYSTEM_PROMPT, DIABETES_STRICT_PROMPT, HYPERTENSION_STRICT_PROMPT,
        RESPIRATORY_STRICT_PROMPT, MENTAL_HEALTH_STRICT_PROMPT, EVALUATION_PROMPT,
        FRIENDLY_SYSTEM_PROMPT, SOAP_PROMPT, STRICT_SYSTEM_PROMPT, WHO_RAG_AUDIT_PROMPT,
        GUIDELINE_DIFF_PROMPT, SAFETY_SUMMARY_PROMPT,
    )
except ImportError:
    from src.shared.prompt_templates import (
        BALANCED_SYSTEM_PROMPT, DIABETES_STRICT_PROMPT, HYPERTENSION_STRICT_PROMPT,
        RESPIRATORY_STRICT_PROMPT, MENTAL_HEALTH_STRICT_PROMPT, EVALUATION_PROMPT,
        FRIENDLY_SYSTEM_PROMPT, SOAP_PROMPT, STRICT_SYSTEM_PROMPT, WHO_RAG_AUDIT_PROMPT,
        GUIDELINE_DIFF_PROMPT, SAFETY_SUMMARY_PROMPT,
    )

logger = logging.getLogger(__name__)

PROMPT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "strict": {"description": "Cautious medical assistant", "local": STRICT_SYSTEM_PROMPT},
    "friendly": {"description": "Friendly medical assistant", "local": FRIENDLY_SYSTEM_PROMPT},
    "balanced": {"description": "Balanced medical assistant", "local": BALANCED_SYSTEM_PROMPT},
    "diabetes": {"description": "Diabetes specialist — WHO/ADA/BYT", "local": DIABETES_STRICT_PROMPT},
    "hypertension": {"description": "Hypertension specialist — AHA/ACC 2025", "local": HYPERTENSION_STRICT_PROMPT},
    "respiratory": {"description": "Asthma/COPD — GOLD 2024 + GINA", "local": RESPIRATORY_STRICT_PROMPT},
    "mental": {"description": "Mental health safe — WHO mhGAP", "local": MENTAL_HEALTH_STRICT_PROMPT},
    "soap": {"description": "SOAP pre-visit summary", "local": SOAP_PROMPT},
    "who_rag": {"description": "WHO-RAG Infrastructure audit", "local": WHO_RAG_AUDIT_PROMPT},
    "evaluation": {"description": "LLM-as-judge rubric", "local": EVALUATION_PROMPT},
    "guideline_diff": {"description": "Guideline change summarizer (VI)", "local": GUIDELINE_DIFF_PROMPT},
    "safety_summary": {"description": "Drug safety alert summarizer (VI)", "local": SAFETY_SUMMARY_PROMPT},
}

try:
    from src.shared.config import PROMPT_ACTIVE_CACHE_TTL_SECONDS
except ImportError:
    from shared.config import PROMPT_ACTIVE_CACHE_TTL_SECONDS  # type: ignore

_cache: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

def _seed_if_empty():
    for tone, spec in PROMPT_REGISTRY.items():
        active = get_active_prompt(tone)
        if not active:
            create_prompt(tone, spec["local"], spec["description"], created_by="system", status="active")
            # also mark active
            import hashlib
            v = hashlib.sha256(spec["local"].encode()).hexdigest()[:8]
            approve_prompt(tone, v)

# ensure seeded on import (best effort)
try:
    _seed_if_empty()
except Exception as e:
    logger.warning(f"prompt seed failed: {e}")

def _cached(tone: str) -> Optional[str]:
    with _lock:
        entry = _cache.get(tone)
        if not entry:
            return None
        if (time.time() - entry["fetched_at"]) < PROMPT_ACTIVE_CACHE_TTL_SECONDS:
            return entry["text"]
    return None

def get_system_prompt(tone: str = "balanced", refresh: bool = False) -> str:
    spec = PROMPT_REGISTRY.get(tone)
    if spec is None:
        logger.warning(f"Unknown tone {tone}, fallback balanced")
        return get_system_prompt("balanced", refresh=refresh)
    if not refresh:
        c = _cached(tone)
        if c:
            return c
    active = get_active_prompt(tone)
    if active:
        text = active["text"]
        with _lock:
            _cache[tone] = {"text": text, "fetched_at": time.time(), "version": active["version"]}
        return text
    # fallback local
    logger.info(f"Using local fallback for {tone}")
    return spec["local"]

def get_prompt_version(tone: str) -> str:
    text = get_system_prompt(tone)
    return short_hash(text)

def get_evaluation_prompt() -> str:
    return get_system_prompt("evaluation")

# Compatibility shim for old callers
def push_all_prompts(commit_message: str = "seed") -> Dict[str, bool]:
    try:
        _seed_if_empty()
        return {k: True for k in PROMPT_REGISTRY}
    except Exception as e:
        return {k: False for k in PROMPT_REGISTRY}
